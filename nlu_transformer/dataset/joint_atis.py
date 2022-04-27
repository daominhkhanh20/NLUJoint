from typing import List
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from nlu_transformer.processor import JointProcessor
from nlu_transformer.processor.make_input import InputFeature


class JointAtisCollate:
    def __init__(self, pad_token_id: int = 0, pad_token_segment_id: int = 0, pad_token_label_id: int = 0):
        self.pad_token_id = pad_token_id
        self.pad_token_segment_id = pad_token_segment_id
        self.pad_token_label_id = pad_token_label_id

    def __call__(self, batchs):
        input_ids, attention_mask, token_type_ids, intent_label_ids, slot_label_ids, all_slot_mask, list_index = [], [], [], [], [], [], []
        for batch in batchs:
            input_ids.append(torch.tensor(batch['feature'].input_ids, dtype=torch.long))
            attention_mask.append(torch.tensor(batch['feature'].attention_mask, dtype=torch.long))
            token_type_ids.append(torch.tensor(batch['feature'].token_type_ids, dtype=torch.long))
            intent_label_ids.append(batch['feature'].intent_label_ids)
            slot_label_ids.append(torch.tensor(batch['feature'].slot_label_ids, dtype=torch.long))
            all_slot_mask.append(torch.tensor(batch['feature'].all_slot_masks, dtype=torch.long))
            list_index.append(batch['idx'])

        data = {
            'input_ids': pad_sequence(input_ids, padding_value=self.pad_token_id, batch_first=True),
            'attention_mask': pad_sequence(attention_mask, padding_value=0, batch_first=True),
            'token_type_ids': pad_sequence(token_type_ids, padding_value=self.pad_token_segment_id, batch_first=True),
            'intent_label_ids': torch.tensor(intent_label_ids, dtype=torch.long),
            'slot_label_ids': pad_sequence(slot_label_ids, padding_value=self.pad_token_label_id, batch_first=True),
            'all_slot_mask': pad_sequence(all_slot_mask, padding_value=self.pad_token_label_id, batch_first=True),
            'list_index': list_index
        }
        return data


class JointAtisDataset(Dataset):
    def __init__(self, tokenizer, max_seq_len: int = 512,
                 path_folder_data: str = 'assets/data/atis/syllable-level',
                 is_relabel: bool = False,
                 mode: str = 'train', pad_token_label_id: int = 0, merge_train_dev: bool = False,
                 ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.path_folder_data = path_folder_data
        self.mode = mode
        self.pad_token_label_id = pad_token_label_id
        self.processor = JointProcessor(
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            path_folder_data=self.path_folder_data,
            mode=self.mode,
            is_relabel=is_relabel,
            pad_token_label_id=self.pad_token_label_id,
            merge_train_dev=merge_train_dev
        )
        self.features = self.processor.convert_example_to_features()

    def __len__(self):
        return len(self.processor.data)

    def __getitem__(self, idx):
        return {
            'feature': self.features[idx],
            'idx': idx
        }


class JointAtisDatasetMeanTeacher(Dataset):
    def __init__(self, features: List[InputFeature], list_index_accept: List[int]):
        self.features = features
        self.list_index_accept = list_index_accept

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'feature': self.features[idx],
            'idx': self.list_index_accept[idx]
        }
