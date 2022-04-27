import os
import json
from typing import Optional, List, Dict
from collections import defaultdict
import re
import random

import pandas as pd
from tqdm import tqdm
from nlu_transformer.utils.io import get_list_intent_labels, get_list_slot_labels, read_file


class ArgumentData(object):
    def __init__(self, intents: List[str] = None,
                 slots: List[List[str]] = None,
                 tags_to_words: Dict = None,
                 chunk_texts: List[List[str]] = None,
                 chunk_slots: List[List[str]] = None,
                 original_slots: List[List[str]] = None,
                 max_argument_sentence: int = 20
                 ):
        self.intents = intents
        self.slots = slots
        self.tags_to_words = tags_to_words
        self.chunk_texts = chunk_texts
        self.chunk_slots = chunk_slots
        self.original_slots = original_slots
        self.max_argument_sentence = max_argument_sentence

    @classmethod
    def chunk_one_sentence(self, sentence: str, slot: str):
        words = sentence.split(" ")
        slot_copy = slot.split(" ")
        slot = [slot_word[2:] if slot_word not in ['PAD', 'O', 'UNK'] else slot_word for slot_word in slot.split(" ")]
        chunk_word, chunk_slot, original_slots = [], [], []
        current_index = 0
        for i in range(len(words)):
            if i == 0:
                chunk_slot.append(slot[i])
                chunk_word.append(words[i])
                original_slots.append(slot_copy[i])
            else:
                if slot[i] == slot[i - 1]:
                    chunk_word[current_index] = chunk_word[current_index] + f" {words[i]}"
                    original_slots[current_index] = original_slots[current_index] + f" {slot_copy[i]}"
                else:
                    chunk_word.append(words[i])
                    chunk_slot.append(slot[i])
                    original_slots.append(slot_copy[i])
                    current_index += 1

        assert len(chunk_word) == len(original_slots)
        assert len(chunk_word) == len(chunk_slot)
        return chunk_word, chunk_slot, original_slots

    @classmethod
    def init_attribute(cls, path_folder_data, remove_number: bool = False,
                       max_argument_sentence: int = 20, is_relabel: bool = False):
        path_intents = os.path.join(path_folder_data, 'label')
        if os.path.exists(path_intents) is False:
            raise Exception("Path intents not found")

        path_slots = os.path.join(path_folder_data, 'seq.out')
        if os.path.exists(path_slots) is False:
            raise Exception("Path slots not found")

        path_sentences = os.path.join(path_folder_data, 'seq.in')
        if os.path.exists(path_sentences) is False:
            raise Exception("Path sentence not found")

        if is_relabel:
            data = pd.read_csv(f"{path_folder_data}/relabel_train.csv")
            sentences = data['text'].values.tolist()
            intents = data['relabel_intent'].values.tolist()
            slots = data['tag'].values.tolist()
        else:
            sentences = read_file(path_sentences)
            intents = read_file(path_intents)
            slots = read_file(path_slots)

        tags_to_words = {}
        chunk_texts, chunk_slots, original_slots = [], [], []

        for sent, slot, intent in tqdm(zip(sentences, slots, intents), total=len(sentences)):
            chunk_text, chunk_slot, original_slot = cls.chunk_one_sentence(sent, slot)
            chunk_texts.append(chunk_text)
            chunk_slots.append(chunk_slot)
            original_slots.append(original_slot)
            if intent not in tags_to_words:
                tags_to_words[intent] = {}
            for idx in range(len(chunk_text)):
                temp_slot = chunk_slot[idx]
                if temp_slot not in ['PAD', 'O', 'UNK']:
                    if temp_slot not in tags_to_words[intent]:
                        tags_to_words[intent][temp_slot] = {'words': [],
                                                            'map_word_to_slot': {},
                                                            'max_number': -100}

                    temp_word = chunk_text[idx].strip()
                    if remove_number and bool(re.search(r"\d", temp_word)):
                        all_numbers = [int(value) for value in re.findall(r"\d+", temp_word)]
                        if len(all_numbers) > 0:
                            max_number = max(all_numbers)
                            tags_to_words[temp_slot]['max_number'] = max(max_number,
                                                                         tags_to_words[temp_slot]['max_number'])

                        tags_to_words[intent][temp_slot]['has_number'] = True
                        temp_word = re.sub('\d+', " ", temp_word).strip()

                    if temp_word != "" and temp_word not in tags_to_words[intent][temp_slot]['words']:
                        tags_to_words[intent][temp_slot]['words'].append(temp_word)
                        tags_to_words[intent][temp_slot]['map_word_to_slot'][temp_word] = original_slot[idx]

        return cls(intents, slots, tags_to_words, chunk_texts, chunk_slots, original_slots, max_argument_sentence)

    def generate_data(self, path_save_data: str, write_csv: bool = False):
        if os.path.exists(path_save_data) is False:
            os.makedirs(path_save_data, exist_ok=True)

        final_intents, final_slots, final_sentences = [], [], []
        for idx, (chunk_text, chunk_slot) in tqdm(enumerate(zip(self.chunk_texts, self.chunk_slots)),
                                                  total=len(self.chunk_texts)):
            original_sentence = " ".join(chunk_text)
            final_sentences.append(original_sentence)
            final_intents.append(self.intents[idx])
            final_slots.append(" ".join(self.original_slots[idx]))
            cnt = 0
            if len(chunk_text) == 1:
                continue
            while cnt < self.max_argument_sentence:
                new_sentence = ""
                new_slot = ""
                for i in range(len(chunk_text)):
                    temp_slot = chunk_slot[i]
                    if temp_slot not in ['PAD', 'O', 'UNK']:
                        all_candidates = self.tags_to_words[self.intents[idx]][temp_slot]['words']
                        random_candidate = random.choice(all_candidates)
                        new_sentence += f"{random_candidate} "
                        new_slot += f"{self.tags_to_words[self.intents[idx]][temp_slot]['map_word_to_slot'][random_candidate]} "
                    else:
                        new_sentence += f"{chunk_text[i]} "
                        new_slot += f"{self.original_slots[idx][i]} "

                new_sentence = new_sentence.strip()
                new_slot = new_slot.strip()
                if new_sentence != original_sentence:
                    cnt += 1
                    final_sentences.append(new_sentence)
                    final_slots.append(new_slot)
                    final_intents.append(self.intents[idx])

        if write_csv:
            df = pd.DataFrame({'text': final_sentences, 'relabel_intent': final_intents, 'tag': final_slots})
            df.to_csv(f"{path_save_data}/relabel_train_aug.csv", index=False)
        else:
            self.save_file(final_sentences, f"{path_save_data}/seq.in")
            self.save_file(final_intents, f"{path_save_data}/label")
            self.save_file(final_slots, f"{path_save_data}/seq.out")

    def save_file(self, data, path):
        with open(path, 'w') as file:
            for sent in data:
                file.write(f"{sent}\n")

    def save_json(self, data, path_save):
        with open(path_save, 'w') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)


aug = ArgumentData.init_attribute(path_folder_data='assets/data/bkai/train', remove_number=False,
                                  max_argument_sentence=50, is_relabel=True)
# for slot in aug.chunk_slots:
#     print(slot)
aug.generate_data(path_save_data='assets/data/bkai/train_aug', write_csv=True)
os.system('cp assets/data/bkai/train/intent_label.txt assets/data/bkai/train_aug/intent_label.txt')
os.system('cp assets/data/bkai/train/slot_label.txt assets/data/bkai/train_aug/slot_label.txt')

aug.save_json(aug.tags_to_words, 'tag_to_words.json')

#
# aug1 = ArgumentData.init_attribute(path_folder_data='assets/data/bkai/train', remove_number=True)
# aug1.save_json(aug1.tags_to_words, 'tag_to_words1.json')


# aug = ArgumentData()
# all_sentece = read_file('assets/data/bkai/train/seq.in')
# all_slots = read_file('assets/data/bkai/train/seq.out')
# chunk_texts, chunk_slots = [], []

# chunk_text, chunk_slot, original_slot = aug.chunk_one_sentence(
#     sentence='tôi đang cần giảm mức độ đèn bóng trần phòng thờ 8',
#     slot='O O O O O O O O O B-roomroom I-roomroom I-roomroom'
# )
#
# print(chunk_text)
# print(chunk_slot)
# print(original_slot)


# for sent, slot in tqdm(zip(all_sentece, all_slots), total=len(all_sentece)):
#     chunk_text, chunk_slot, original_slot = aug.chunk_one_sentence(
#         sentence=sent,
#         slot=slot
#     )
#     chunk_texts.append(chunk_text)
#     chunk_slots.append(chunk_slot)

# df = pd.DataFrame({'text': all_sentece, 'chunk_text': chunk_texts, 'slots': all_slots, 'chunk_slot': chunk_slots})
# df.to_csv('temp.csv', index=False)
