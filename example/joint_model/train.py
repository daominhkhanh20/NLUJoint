import argparse
import pickle
import torch
from transformers import RobertaConfig
from nlu_transformer.trainer import TrainerJointModel
from nlu_transformer.utils import *
from nlu_transformer.model import JointModel
from nlu_transformer.dataset import DataSource
from nlu_transformer.inference import InferenceJoint


parser = argparse.ArgumentParser()
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--use_intent_context_concat', default=True, type=lambda x: str(x).lower() == 'true')
parser.add_argument('--use_intent_context_attn', default=False, type=lambda x: str(x).lower() == 'true')
parser.add_argument('--attention_embedding_size', type=int, default=200)
parser.add_argument('--use_crf', default=False, type=lambda x: str(x).lower() == 'true')
parser.add_argument('--intent_loss_coef', type=float, default=0.2)
parser.add_argument('--pretrained_model_name', type=str, default='vinai/phobert-base')
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--num_warmup_steps', type=int, default=50)
parser.add_argument('--continue_train', default=False, type=lambda x: str(x).lower() == 'true')
parser.add_argument('--path_pretrained', type=str, default=None)
parser.add_argument('--path_save_model', type=str, default='models')
parser.add_argument('--path_folder_data', type=str, default='assets/data/bkai')
parser.add_argument('--path_test_set', type=str, default='assets/data/test_set')
parser.add_argument('--is_train_aug', default=False, type=lambda x: x.lower() == 'true')
parser.add_argument('--is_relabel', default=False, type=lambda x: x.lower() == 'true')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_seq_len', type=int, default=256)
parser.add_argument('--pad_token_label_id', type=float, default=0)
parser.add_argument('--pad_token_segment_id', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--epoch_report', type=int, default=100)
parser.add_argument('--num_accumulate_gradient', type=int, default=1)
parser.add_argument('--max_clip_norm', type=float, default=1.0)
parser.add_argument('--epoch_save', type=int, default=30)
parser.add_argument('--update_loss', default=False, type=lambda x: x.lower() == 'true')
parser.add_argument('--loss_acc_coef', type=float, default=1.0)
parser.add_argument('--path_file_predict', type=str, default='assets/data/bkai/test/seq.in')
parser.add_argument('--is_save_best', default=False, type=lambda x: x.lower() == 'true')
parser.add_argument('--merge_train_dev', default=False, type=lambda x: x.lower() == 'true')


args = parser.parse_args()
nlu_datasource = DataSource.create_datasource_from_dataset(
    args=args,
    path_folder_data=args.path_folder_data,
    max_seq_len=args.max_seq_len,
    is_train_aug=args.is_train_aug,
    merge_train_dev=args.merge_train_dev
)

if args.continue_train:
    optimizer_state = torch.load(f"{args.path_save_model}/optimizer.pth")
    model = JointModel.from_pretrained(
        args.path_save_model,
        n_intent_label=nlu_datasource.n_intents,
        n_slot_label=nlu_datasource.n_slots,
        update_loss=args.update_loss,
        loss_acc_coef=args.loss_acc_coef
    )

else:
    optimizer_state = None
    config = RobertaConfig.from_pretrained(args.pretrained_model_name)
    model = JointModel.from_pretrained(
        args.pretrained_model_name,
        config=config,
        args=args,
        n_intent_label=nlu_datasource.n_intents,
        n_slot_label=nlu_datasource.n_slots,
        contribution_intent_loss_level=nlu_datasource.contribution_intent_loss_level,
        contribution_slot_loss_level=nlu_datasource.contribution_slot_loss_level,
        update_loss=args.update_loss,
        loss_acc_coef=args.loss_acc_coef
    )

if not args.merge_train_dev:
    assert nlu_datasource.train_dataset.processor.list_intents == nlu_datasource.dev_dataset.processor.list_intents
    assert nlu_datasource.train_dataset.processor.list_slots == nlu_datasource.dev_dataset.processor.list_slots

trainer = TrainerJointModel(
    args=args,
    model=model,
    is_train_continue=args.continue_train,
    datasource=nlu_datasource,
    epoch_report=args.epoch_report,
    optimizer_state=optimizer_state
)

trainer.fit()