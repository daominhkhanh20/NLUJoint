import os
from typing import Optional, Union, List, Dict
from underthesea import word_tokenize
from collections import Counter, defaultdict
import numpy as np
import re
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from nlu_transformer.utils.process import remove_number

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class RelabelByCluster:
    def __init__(self, path_folder_data: str = 'assets/data/bkai',
                 path_save_data: str = 'assets/data/bkai_relabel',
                 top_k_common: int = 20,
                 column_name_relabel: str = 'text_copy'):
        self.top_word_common_by_label = None
        self.path_folder_data = path_folder_data
        self.path_save_data = path_save_data
        self.top_k_common = top_k_common
        self.column_name_relabel = column_name_relabel
        if os.path.exists(self.path_save_data) is False:
            os.makedirs(self.path_save_data)

    def convert_word_counter(self, corpus_sent_by_label: Dict):
        top_word_common_by_label = dict()
        for key, value in corpus_sent_by_label.items():
            counter_word = Counter(value['text'].split(" "))
            top_words = counter_word.most_common(self.top_k_common)
            top_word_common_by_label[key] = {}

            for word, word_counter in top_words:
                if word != 'phần_trăm':
                    top_word_common_by_label[key][word] = word_counter / value['count']
                else:
                    top_word_common_by_label[key][word] = 2 * word_counter / value['count']
        return top_word_common_by_label

    def make_word_counter_by_label(self):
        df_train = pd.read_csv(f"{self.path_folder_data}/train/train.csv")
        df_train[self.column_name_relabel] = df_train['text'].astype(str).apply(lambda x: remove_number(x))
        df_train[self.column_name_relabel] = df_train[self.column_name_relabel].apply(
            lambda x: word_tokenize(x, format='text'))

        corpus_sent_by_label = dict()
        for idx, row in df_train.iterrows():
            if row['intent'] not in corpus_sent_by_label:
                corpus_sent_by_label[row['intent']] = {
                    'text': f"{row['text_copy']} ",
                    'count': 1
                }
            else:
                corpus_sent_by_label[row['intent']]['text'] += f"{row[self.column_name_relabel]} "
                corpus_sent_by_label[row['intent']]['count'] += 1
        return self.convert_word_counter(corpus_sent_by_label), df_train

    def relabel(self, sent: str, debug_mode: bool = False):
        # sent = word_tokenize(sent, format='text')
        sent = sent.split(" ")
        label, score = None, 0
        if debug_mode:
            print(sent)
        for key, value in self.top_word_common_by_label.items():
            temp_score = 0
            temp = {}
            for word in sent:
                if word in value:
                    temp_score += value[word]
                    temp[word] = value[word]
            if temp_score > score:
                score = temp_score
                label = key

            if debug_mode:
                print(key)
                print(temp)
                print(temp_score)

        if debug_mode:
            print("Predict:", label)
        return label

    def relabel_by_word_counter(self, sent: str = None):
        self.top_word_common_by_label, df = self.make_word_counter_by_label()
        # print(self.top_word_common_by_label)
        if sent is not None:
            self.relabel(sent, debug_mode=True)
        else:
            new_labels, is_diff = [], []
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                label = self.relabel(row[self.column_name_relabel])
                new_labels.append(label)
                if label != row['intent']:
                    is_diff.append(True)
                else:
                    is_diff.append(0)
            df_new = pd.DataFrame({
                'text': df['text'],
                self.column_name_relabel: df[self.column_name_relabel],
                'intent': df['intent'],
                'is_diff': is_diff,
                'relabel_intent': new_labels,
                'tag': df['tag']
            })
            df_new.to_csv(f"{self.path_save_data}/relabel_train.csv", index=False)


class RelabelByML:
    def __init__(self, path_folder_data: str = 'assets/data/bkai',
                 path_save_data: str = 'assets/data/bkai_relabel_ml',
                 text_column_name: str = 'new_text',
                 continuous_replace: bool = False,
                 kernel: str = 'linear'
                 ):
        self.top_word_common_by_label = None
        self.path_folder_data = path_folder_data
        self.path_save_data = path_save_data
        self.text_column_name = text_column_name
        self.continuous_replace = continuous_replace
        self.model = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 1), max_features=None)),
            # ('tfidf', TfidfTransformer()),
            # ('clf', KNeighborsClassifier(n_neighbors=20))
            # ('clf', DecisionTreeClassifier())
            # ('clf', MultinomialNB())
            # ('clf', svm.SVC(kernel='linear', C=1, decision_function_shape='ovo'))
            ('clf', svm.SVC(kernel=kernel, C=1, decision_function_shape='ovo'))

        ])
        self.data = pd.read_csv(f"{self.path_folder_data}/train/train.csv")
        self.data = self.remove_redundant_text(self.data)
        self.data[self.text_column_name] = self.data[self.text_column_name].apply(lambda x: remove_number(x))
        self.all_sentences = self.data[self.text_column_name].values.tolist()
        self.all_labels = self.data['intent'].values.tolist()

        if os.path.exists(self.path_save_data) is False:
            os.makedirs(self.path_save_data)

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

    def remove_redundant_text(self, df: DataFrame):
        text_remove_redundants = []
        word_reject = ['device', 'room']
        for idx, row in df.iterrows():
            chunk_text, chunk_slot, _ = self.chunk_one_sentence(row['text'], row['tag'])
            temp = ""
            for index in range(len(chunk_text)):
                if any(word in chunk_slot[index] for word in word_reject):
                    continue
                else:
                    temp += chunk_text[index] + " "
            text_remove_redundants.append(temp)
        df[self.text_column_name] = text_remove_redundants
        return df

    def fit_model(self, sent):
        list_index_accept = np.where(np.array(self.all_sentences) != sent)[0]
        list_index_reject = np.where(np.array(self.all_sentences) == sent)[0]
        sub_all_sentences = [self.all_sentences[index] for index in list_index_accept]
        sub_all_labels = [self.all_labels[index] for index in list_index_accept]

        self.model.fit(sub_all_sentences, sub_all_labels)
        label = self.model.predict([sent])[0]
        if self.continuous_replace:
            for idx in list_index_reject:
                self.all_labels[idx] = label
        return label

    def relabel(self):

        cnt = 0
        old_labels = None
        old_is_diff = None
        while cnt < 10:
            new_labels, is_diff = [], []
            for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
                current_sent = row[self.text_column_name]
                label = self.fit_model(current_sent)
                new_labels.append(label)
                if label != row['intent']:
                    is_diff.append(True)
                else:
                    is_diff.append(0)

            cnt += 1
            if old_labels is None:
                old_labels = new_labels
                old_is_diff = is_diff
                continue

            if new_labels == old_labels:
                break
            else:
                old_labels = new_labels
                old_is_diff = is_diff

        df = pd.DataFrame({
            'text': self.data['text'],
            self.text_column_name: self.data[self.text_column_name],
            'intent': self.data['intent'],
            'is_diff': old_is_diff,
            'relabel_intent': old_labels,
            'tag': self.data['tag']
        })
        os.makedirs(f"{self.path_save_data}/train", exist_ok=True)
        df.to_csv(f"{self.path_save_data}/train/relabel_train.csv", index=False)
        os.system(f"cp {self.path_folder_data}/train/intent_label.txt {self.path_save_data}/train/intent_label.txt")
        os.system(f"cp {self.path_folder_data}/train/slot_label.txt {self.path_save_data}/train/slot_label.txt")

        # relabel for dev
        df_dev = pd.read_csv(f"{self.path_folder_data}/dev/dev.csv")
        df_dev = self.remove_redundant_text(df_dev)
        self.model.fit(self.all_sentences, self.all_labels)
        new_labels_dev = []
        is_diff = []
        for idx, row in tqdm(df_dev.iterrows(), total=len(df_dev)):
            new_labels_dev.append(self.model.predict([row[self.text_column_name]])[0])
            if row['intent'] != new_labels_dev[-1]:
                is_diff.append(True)
            else:
                is_diff.append(0)
        df_dev = pd.DataFrame({
            'text': df_dev['text'],
            self.text_column_name: df_dev[self.text_column_name],
            'intent': df_dev['intent'],
            'is_diff': is_diff,
            'relabel_intent': new_labels_dev,
            'tag': df_dev['tag']
        })
        os.makedirs(f"{self.path_save_data}/dev", exist_ok=True)
        df_dev.to_csv(f"{self.path_save_data}/dev/relabel_dev.csv")
        os.system(f"cp {self.path_folder_data}/dev/intent_label.txt {self.path_save_data}/dev/intent_label.txt")
        os.system(f"cp {self.path_folder_data}/dev/slot_label.txt {self.path_save_data}/dev/slot_label.txt")

        # copy test folder
        os.system(f"cp -r {self.path_folder_data}/test {self.path_save_data}")




