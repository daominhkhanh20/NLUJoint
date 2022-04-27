import copy
import json
import pandas as pd
import os
import glob
from tqdm import tqdm
import logging
from collections import Counter
from typing import List

from nlu_transformer.utils.io import get_list_slot_labels, get_list_intent_labels, read_file
from nlu_transformer.utils.process import make_contribution_loss_level

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, words: list, intent_label: int = None, tag_label: list = None):
        self.words = words
        self.text = " ".join(words)
        self.intent_label = intent_label
        self.tag_label = tag_label

    def __repr__(self):
        return str(self.to_json())

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)


class InputFeature(object):
    def __init__(self, input_ids: list, attention_mask: list,
                 token_type_ids: list, intent_label_ids: int,
                 slot_label_ids: list, all_slot_masks: list):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_ids = intent_label_ids
        self.slot_label_ids = slot_label_ids
        self.all_slot_masks = all_slot_masks

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class JointProcessor(object):
    def __init__(self, tokenizer, max_seq_len: int, path_folder_data: str, mode: str, is_relabel: bool = False,
                 pad_token_label_id: int = 0, merge_train_dev: bool = False, list_index_accept: List = None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.path_folder_data = path_folder_data
        self.pad_token_label_id = pad_token_label_id
        self.list_index_accept = list_index_accept

        if is_relabel:
            file_csv = f"relabel_{mode}.csv"
        else:
            file_csv = f"{mode}.csv"
        if os.path.exists(os.path.join(self.path_folder_data, mode, file_csv)):
            self.path_folder_data = os.path.join(self.path_folder_data, mode)
            logger.info(f"Start loading data in {os.path.join(self.path_folder_data, file_csv)}")
            path_intent = glob.glob(f'{self.path_folder_data}/intent*.txt')[0]
            self.list_intents = get_list_intent_labels(path_intent)
            path_slot = glob.glob(f'{self.path_folder_data}/slot*.txt')[0]
            self.list_slots = get_list_slot_labels(path_slot)
            self.data = pd.read_csv(self.path_folder_data + f'/{file_csv}')

            if merge_train_dev:
                if is_relabel:
                    df_dev = pd.read_csv(f"{path_folder_data}/dev/relabel_dev.csv")
                else:
                    df_dev = pd.read_csv(f"{path_folder_data}/dev/dev.csv")
                self.data = pd.concat([self.data, df_dev], sort=False)

        else:
            # load bkai data
            self.path_folder_data = os.path.join(self.path_folder_data, mode)
            self.path_intents = os.path.join(self.path_folder_data, 'intent_label.txt')
            self.path_slots = os.path.join(self.path_folder_data, 'slot_label.txt')

            self.list_intents = get_list_intent_labels(self.path_intents)
            self.list_slots = get_list_slot_labels(self.path_slots)
            if not os.path.exists(os.path.join(self.path_folder_data, 'seq.in')):
                raise Exception("File seq.in not found")

            if not os.path.exists(os.path.join(self.path_folder_data, 'seq.out')):
                raise Exception("File seq.out not found")

            if not os.path.exists(os.path.join(self.path_folder_data, 'label')):
                raise Exception("File label not found")

            df = {
                'text': read_file(os.path.join(self.path_folder_data, 'seq.in')),
                'intent': read_file(os.path.join(self.path_folder_data, 'label')),
                'tag': read_file(os.path.join(self.path_folder_data, 'seq.out'))
            }
            self.data = pd.DataFrame(df)

        self.tags = self.data.tag.apply(lambda x: x.strip().split(" ")).values.tolist()
        self.texts = self.data.text.values.tolist()
        if is_relabel:
            self.intents = self.data.relabel_intent.values.tolist()
        else:
            self.intents = self.data.intent.values.tolist()
        self.contribution_intent_loss_level = make_contribution_loss_level(self.intents, self.list_intents,
                                                                           type_input="Intent")
        self.contribution_slot_loss_level = make_contribution_loss_level(self.tags, self.list_slots, type_input="Tag")

    def create_example(self):
        list_examples = []
        logger.info("Time for making example")
        cnt = 0
        slot_cnt = []
        for idx, (text, intent, slot) in tqdm(enumerate(zip(self.texts, self.intents, self.tags)),
                                              total=len(self.tags)):
            words = text.split()
            if intent in self.list_intents:
                intent_label = self.list_intents.index(intent)
            else:
                logger.info(f"{intent} not exist in list intents")
                intent_label = self.list_intents.index('UNK')

            slot_labels = []
            for value in slot:
                if value in self.list_slots:
                    slot_labels.append(self.list_slots.index(value))
                else:
                    logger.info(f"{value} not exist in list slots")
                    slot_labels.append(self.list_slots.index('UNK'))

            assert len(words) == len(slot_labels)
            slot_cnt.extend([self.list_slots[value] for value in slot_labels])
            # if cnt < 10:
            #     print(words)
            #     print(slot_labels)
            #     print(" ".join([self.list_slots[value] for value in slot_labels]))
            #     cnt += 1

            temp_example = InputExample(words, intent_label, slot_labels)
            list_examples.append(temp_example)

        return list_examples

    def convert_example_to_features(self):
        list_examples = self.create_example()
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        count_special_token = 2
        features = []
        logger.info("Time for convert example to feature")
        cnt = 0
        for idx, example in tqdm(enumerate(list_examples), total=len(list_examples)):

            input_ids, token_type_ids, slot_label_ids, all_slot_masks = [], [], [], []
            for index, word in enumerate(example.words):
                outs = self.tokenizer(word, add_special_tokens=False)
                if 'token_type_ids' not in outs:
                    outs = self.tokenizer(word, add_special_tokens=False, return_token_type_ids=True)
                input_ids.extend(outs['input_ids'])
                token_type_ids.extend(outs['token_type_ids'])
                temp_slot_label = [example.tag_label[index]] + (len(outs['input_ids']) - 1) * [self.pad_token_label_id]
                slot_label_ids.extend(temp_slot_label)
                all_slot_masks.extend(
                    [self.pad_token_label_id + 1] + [self.pad_token_label_id] * (len(outs['input_ids']) - 1))

            if len(input_ids) > self.max_seq_len - count_special_token:
                n_element = self.max_seq_len - count_special_token
                input_ids = input_ids[:n_element]
                token_type_ids = token_type_ids[:n_element]
                slot_label_ids = slot_label_ids[:n_element]
                all_slot_masks = all_slot_masks[:n_element]

            input_ids = [cls_token_id] + input_ids + [sep_token_id]
            attention_mask = [1] * len(input_ids)
            token_type_ids = token_type_ids + [token_type_ids[0]] * count_special_token
            slot_label_ids = [self.pad_token_label_id] + slot_label_ids + [self.pad_token_label_id]
            all_slot_masks = [self.pad_token_label_id] + all_slot_masks + [self.pad_token_label_id]

            # if cnt < 5:
            #     print('\n')
            #     print(example.words)
            #     print(example.intent_label)
            #     print(example.tag_label)
            #     print(input_ids)
            #     print(attention_mask)
            #     print(token_type_ids)
            #     print(slot_label_ids)
            #     print(all_slot_masks)
            #     print('\n')
            #     cnt += 1

            assert len(slot_label_ids) == len(token_type_ids)
            features.append(
                InputFeature(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    intent_label_ids=int(example.intent_label),
                    slot_label_ids=slot_label_ids,
                    all_slot_masks=all_slot_masks
                )
            )
        return features
