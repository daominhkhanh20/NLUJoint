import os
import logging
from typing import List, Optional

from torch import Tensor
from transformers import AutoTokenizer
from torch.utils.data import Dataset

from nlu_transformer.dataset import JointAtisDataset, JointAtisDatasetMeanTeacher
from nlu_transformer.processor.make_input import InputFeature, JointProcessor

logger = logging.getLogger(__name__)


class DataSource(object):
    def __init__(self, train_dataset: JointAtisDataset = None,
                 dev_dataset: JointAtisDataset = None,
                 max_seq_len: int = None,
                 pad_token_id: int = None,
                 n_intents: int = None,
                 n_slots: int = None,
                 contribution_intent_loss_level: Tensor = None,
                 contribution_slot_loss_level: Tensor = None
                 ):
        super(DataSource, self).__init__()
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.n_intents = n_intents
        self.n_slots = n_slots
        self.contribution_intent_loss_level = contribution_intent_loss_level
        self.contribution_slot_loss_level = contribution_slot_loss_level

    @classmethod
    def create_datasource_from_dataset(cls, args=None,
                                       path_folder_data: str = None,
                                       max_seq_len: int = None,
                                       is_train_aug: bool = False,
                                       merge_train_dev: bool = False):
        logger.info(f"We use pretrained tokenizer: {args.pretrained_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
        pad_token_id = tokenizer.pad_token_id
        list_folder = os.listdir(path_folder_data)
        n_intents, n_slots = 0, 0
        if 'train' in list_folder:
            if is_train_aug:
                mode = 'train_aug'
            else:
                mode = 'train'

            logger.info(f"Use folder data '{mode}' for training")
            train_dataset = JointAtisDataset(
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                path_folder_data=path_folder_data,
                mode=mode,
                is_relabel=args.is_relabel,
                pad_token_label_id=args.pad_token_label_id,
                merge_train_dev=merge_train_dev
            )
            n_intents = len(train_dataset.processor.list_intents)
            n_slots = len(train_dataset.processor.list_slots)
        else:
            train_dataset = None

        if not merge_train_dev and 'dev' in list_folder:
            dev_dataset = JointAtisDataset(
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                path_folder_data=path_folder_data,
                is_relabel=args.is_relabel,
                mode='dev',
                pad_token_label_id=args.pad_token_label_id
            )
        else:
            dev_dataset = None
            logger.info(f"No dev dataset")

        return cls(train_dataset=train_dataset, dev_dataset=dev_dataset,
                   max_seq_len=max_seq_len, pad_token_id=pad_token_id,
                   n_intents=n_intents, n_slots=n_slots,
                   contribution_intent_loss_level=train_dataset.processor.contribution_intent_loss_level,
                   contribution_slot_loss_level=train_dataset.processor.contribution_slot_loss_level)


class DataSourceMeanTeacher(object):
    def __init__(self, train_processor: JointProcessor,
                 val_dataset: JointAtisDataset,
                 pad_token_id: int = None,
                 contribution_intent_loss_level: Tensor = None,
                 contribution_slot_loss_level: Tensor = None
                 ):
        super(DataSourceMeanTeacher, self).__init__()
        self.train_processor = train_processor
        self.val_dataset = val_dataset
        self.pad_token_id = pad_token_id
        self.contribution_intent_loss_level = contribution_intent_loss_level
        self.contribution_slot_loss_level = contribution_slot_loss_level

    @classmethod
    def create_attribution_for_mean_teacher(cls, args=None,
                                            path_folder_data: str = None,
                                            max_seq_len: int = None,
                                            ):
        logger.info(f"We use pretrained tokenizer: {args.pretrained_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
        pad_token_id = tokenizer.pad_token_id
        list_folder = os.listdir(path_folder_data)

        if 'train' in list_folder:
            train_processor = JointProcessor(
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                path_folder_data=path_folder_data,
                mode='train',
                pad_token_label_id=args.pad_token_label_id
            )
        else:
            train_processor = None

        if 'dev' in list_folder:
            dev_dataset = JointAtisDataset(
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                path_folder_data=path_folder_data,
                is_relabel=args.is_relabel,
                mode='dev',
                pad_token_label_id=args.pad_token_label_id
            )
        else:
            dev_dataset = None

        return cls(train_processor=train_processor, val_dataset=dev_dataset, pad_token_id=pad_token_id,
                   contribution_intent_loss_level=train_processor.contribution_intent_loss_level,
                   contribution_slot_loss_level=train_processor.contribution_slot_loss_level)

    def make_train_dataset(self, list_train_features: List[InputFeature],
                           list_index_accept: List = None):
        if list_index_accept is None:
            list_index_accept = [i for i in range(len(list_train_features))]

        return JointAtisDatasetMeanTeacher(
            features=list_train_features,
            list_index_accept=list_index_accept,
        )


    # @classmethod
    # def create_datasource_from_dataset(cls, args=None,
    #                                    path_folder_data: str = None,
    #                                    max_seq_len: int = None,
    #                                    is_train_aug: bool = False,
    #                                    merge_train_dev: bool = False):
    #     logger.info(f"We use pretrained tokenizer: {args.pretrained_model_name}")
    #     tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    #     pad_token_id = tokenizer.pad_token_id
    #     list_folder = os.listdir(path_folder_data)
    #     n_intents, n_slots = 0, 0
    #     if 'train' in list_folder:
    #         if is_train_aug:
    #             mode = 'train_aug'
    #         else:
    #             mode = 'train'
    #
    #         logger.info(f"Use folder data '{mode}' for training")
    #         train_dataset = JointAtisDataset(
    #             tokenizer=tokenizer,
    #             max_seq_len=max_seq_len,
    #             path_folder_data=path_folder_data,
    #             mode=mode,
    #             is_relabel=args.is_relabel,
    #             pad_token_label_id=args.pad_token_label_id,
    #             merge_train_dev=merge_train_dev
    #         )
    #         n_intents = len(train_dataset.processor.list_intents)
    #         n_slots = len(train_dataset.processor.list_slots)
    #     else:
    #         train_dataset = None
    #
    #     if not merge_train_dev and 'dev' in list_folder:
    #         dev_dataset = JointAtisDataset(
    #             tokenizer=tokenizer,
    #             max_seq_len=max_seq_len,
    #             path_folder_data=path_folder_data,
    #             is_relabel=args.is_relabel,
    #             mode='dev',
    #             pad_token_label_id=args.pad_token_label_id
    #         )
    #     else:
    #         dev_dataset = None
    #         logger.info(f"No dev dataset")
    #
    #     return cls(train_dataset=train_dataset, dev_dataset=dev_dataset,
    #                max_seq_len=max_seq_len, pad_token_id=pad_token_id,
    #                n_intents=n_intents, n_slots=n_slots,
    #                contribution_intent_loss_level=train_dataset.processor.contribution_intent_loss_level,
    #                contribution_slot_loss_level=train_dataset.processor.contribution_slot_loss_level)
