import os
from abc import ABC
from collections import defaultdict
import pickle

import pandas as pd
from tqdm import tqdm
import logging
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from nlu_transformer.inference import InferenceJoint

from nlu_transformer.trainer import TrainerBase
from nlu_transformer.dataset import *
from nlu_transformer.model import *
from nlu_transformer.utils.io import read_file

logger = logging.getLogger(__name__)


class TrainerJointModel(TrainerBase, ABC):
    def __init__(self, args, model: JointModel, is_train_continue: bool = False,
                 datasource: DataSource = None, epoch_report: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.max_seq_length = self.model.roberta.config.max_position_embeddings
        self.train_dataset = datasource.train_dataset
        self.val_dataset = datasource.dev_dataset
        self.pad_token_id = datasource.pad_token_id
        self.test_sentences = read_file(self.args.path_file_predict)

        self.list_intents = self.train_dataset.processor.list_intents
        self.list_slots = self.train_dataset.processor.list_slots
        self.train_loader = self.make_dataloader(self.train_dataset)
        if self.val_dataset is not None:
            self.val_loader = self.make_dataloader(self.val_dataset)
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
        self.has_save_model = False

        if not is_train_continue:
            self.best_acc = -0.01
        else:
            self.best_acc = kwargs.get('best_acc')
            optimizer_state = kwargs.get('optimizer_state')
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)

        self.num_training_steps = len(self.train_loader) // self.args.num_accumulate_gradient * self.args.n_epochs
        self.epoch_report = epoch_report
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )

        self.config = {
            'hyperparameter': {
                'n_epochs': self.args.n_epochs,
                'batch_size': self.args.batch_size,
                'num_accumulate_gradient': self.args.num_accumulate_gradient,
                'num_training_steps': self.num_training_steps,
                'num_warmup_steps': self.args.num_warmup_steps,
                'max_seq_length': self.max_seq_length
            },
            'model': {
                'n_intent_label': len(self.list_intents),
                'n_slot_label': len(self.list_slots),
                'dropout': self.args.dropout,
                'use_crf': self.args.use_crf,
                'intent_loss_coef': self.args.intent_loss_coef,
                'pad_token_label_id': self.args.pad_token_label_id,
                'use_intent_context_concat': self.args.use_intent_context_concat,
                'use_intent_context_attn': self.args.use_intent_context_attn,
                'attention_embedding_size': self.args.attention_embedding_size,
                'contribution_coef': {
                    'intent_coef': self.model.contribution_intent_loss_level.detach().cpu().numpy().tolist(),
                    'slot_coef': self.model.contribution_slot_loss_level.detach().cpu().numpy().tolist()
                },
            },
            'data': {
                'list_intents': self.list_intents,
                'list_slots': self.list_slots,
                'pad_token_segment_id': self.args.pad_token_segment_id,
                'pad_token_label_id': self.args.pad_token_label_id
            },
            'pretrained': {
                'pretrained_name': self.args.pretrained_model_name,
                'pad_token_id': self.pad_token_id,
            },
            'device': self.device.__str__(),
        }

        if os.path.exists(self.args.path_save_model) is False:
            os.makedirs(self.args.path_save_model, exist_ok=True)

    def make_dataloader(self, dataset: JointAtisDataset):
        return DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=JointAtisCollate(
                pad_token_id=self.pad_token_id,
                pad_token_segment_id=self.args.pad_token_segment_id,
                pad_token_label_id=self.args.pad_token_label_id
            )
        )

    def evaluate(self, loader: DataLoader, **kwargs):
        logger.info('************START EVALUATE************')
        self.model.eval()
        val_loss = 0
        list_intent_preds, list_intent_truth, list_slot_preds, list_slot_truth = [], [], [], []
        all_slot_truth, all_slot_preds = [], []
        for step, sample in tqdm(enumerate(loader), total=len(loader)):
            for key in sample.keys():
                sample[key] = sample[key].to(self.device)
            outputs = self.model(
                sample['input_ids'],
                sample['attention_mask'],
                sample['token_type_ids'],
                sample['intent_label_ids'],
                sample['slot_label_ids'],
                sample['all_slot_mask']
            )
            loss, (intent_logits, slot_logits) = outputs[:2]
            val_loss += loss.item()

            # intent
            intent_preds = torch.argmax(intent_logits, dim=1)
            intent_truth = sample['intent_label_ids']
            list_intent_preds.extend(intent_preds.detach().cpu().numpy().tolist())
            list_intent_truth.extend(intent_truth.detach().cpu().numpy().tolist())

            # slot
            slot_preds = torch.argmax(slot_logits, dim=-1).detach().cpu().numpy()
            slot_truth = sample['slot_label_ids'].detach().cpu().numpy()

            for i in range(slot_logits.size(0)):
                temp_truth, temp_pred = [], []
                for j in range(slot_logits.size(1)):
                    if slot_truth[i][j] != self.args.pad_token_label_id:
                        temp_truth.append(self.list_slots[slot_truth[i][j]])
                        temp_pred.append(self.list_slots[slot_preds[i][j]])
                list_slot_truth.append(temp_truth)
                list_slot_preds.append(temp_pred)

                all_slot_truth.extend(temp_truth)
                all_slot_preds.extend(temp_pred)

        val_loss /= len(loader)
        list_intent_truth = [self.list_intents[idx] for idx in list_intent_truth]
        list_intent_preds = [self.list_intents[idx] for idx in list_intent_preds]

        if self.epoch_current > self.epoch_report:
            print(classification_report(list_intent_truth, list_intent_preds))
            print(classification_report(all_slot_truth, all_slot_preds))

        cnt_correct = 0
        for i in range(len(list_intent_preds)):
            if list_intent_preds[i] == list_intent_truth[i] and list_slot_truth[i] == list_slot_preds[i]:
                cnt_correct += 1
        return val_loss, cnt_correct / len(list_intent_preds)

    def get_result(self, path):
        logger.info("Start inference")
        infer = InferenceJoint(model=self.model,
                               config_architecture=self.config)
        intent_preds, slot_preds = infer.inference(self.test_sentences)
        df = pd.DataFrame({'intent': intent_preds, 'slot': slot_preds})
        df.to_csv(f"{path}/results.csv", index=False, header=False)
        os.system(f"zip -j {path}/results.zip {path}/results.csv")
        logger.info(f"Save results.zip file done in {path}")

    def save_model(self, epoch: int, **kwargs):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if self.args.is_save_best:
            path_save = self.args.path_save_model
        else:
            path_save = f"{self.args.path_save_model}/epoch_{epoch}"
        model_to_save.save_pretrained(f"{path_save}")
        logger.info("Save model done")
        self.config['best_acc'] = self.best_acc
        with open(f"{self.args.path_save_model}/config_architecture.json", 'w') as file:
            json.dump(self.config, file, indent=4, ensure_ascii=False)
        logger.info("Save config architecture done")
        self.get_result(path_save)

        # torch.save(self.optimizer, f"{self.args.path_save_model}/optimizer.pth")
        # logger.info("Save optimizer done")

    def train_one_epoch(self, **kwargs):
        train_loss = 0
        global_steps = 0
        self.model.train()
        for step, sample in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            for key in sample.keys():
                sample[key] = sample[key].to(self.device)
            outputs = self.model(
                sample['input_ids'],
                sample['attention_mask'],
                sample['token_type_ids'],
                sample['intent_label_ids'],
                sample['slot_label_ids'],
                sample['all_slot_mask']
            )
            loss = outputs[0]
            if self.args.num_accumulate_gradient > 1:
                loss = loss / self.args.num_accumulate_gradient
            train_loss += loss.item()
            loss.backward()
            if (step + 1) % self.args.num_accumulate_gradient == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_clip_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                global_steps += 1

        return train_loss / global_steps

    def fit(self, **kwargs):
        logger.info('************START TRAINING************')
        for epoch in range(self.args.n_epochs):
            logger.info(f"Epoch current: {epoch}")
            self.epoch_current = epoch
            train_loss = self.train_one_epoch()
            if self.val_dataset is not None:
                val_loss, val_acc = self.evaluate(self.val_loader)
                if val_acc > self.best_acc:
                    if epoch >= self.args.epoch_save:
                        self.has_save_model = True
                        self.save_model(epoch)
                    self.best_acc = val_acc

                logger.info(f"Train loss:{train_loss} --- Val loss: {val_loss} --- "
                            f"Val acc: {val_acc} --- Best acc: {self.best_acc}")
            else:
                logger.info(f"Train loss: {train_loss}")

        if not self.has_save_model:
            self.save_model(self.args.n_epochs)
