import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import pickle
import os
from tqdm import tqdm
import argparse

# class RelabelData:
#     def __init__(self, data: Union[DataFrame, str]):
#

# parser = argparse.ArgumentParser()
# parser.add_argument('--sent', type=str, required=True)
# args = parser.parse_args()

data = pd.read_csv('assets/data/bkai/train/train.csv')
all_sentence = data['text'].values.tolist()
labels = data['intent'].values.tolist()

original_label = labels.copy()
# sent = args.sent
# idx = all_sentence.index(sent)
# del all_sentence[idx]
# del labels[idx]
#
# assert sent not in all_sentence
is_diff = []
for idx, sent in enumerate(all_sentence):

    model = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1), max_features=None)),
        ('tfidf', TfidfTransformer()),
        # ('clf', DecisionTreeClassifier())
        ('clf', KNeighborsClassifier())

    ])
    if idx < len(all_sentence) - 1:
        temp_sentences = all_sentence[:idx] + all_sentence[idx + 1:]
        temp_labels = labels[:idx] + labels[idx + 1:]
    else:
        temp_sentences = all_sentence[:idx]
        temp_labels = labels[:idx]

    model.fit(temp_sentences, temp_labels)

    predict_label = model.predict([all_sentence[idx]])[0]
    if predict_label != labels[idx]:
        print(f"{all_sentence[idx]} -- {predict_label} -- {labels[idx]}")
    labels[idx] = predict_label
    if predict_label != original_label[idx]:
        is_diff.append(True)
    else:
        is_diff.append(0)

data = pd.DataFrame({'text': all_sentence, 'is_diff': is_diff, 'ori_intent': original_label, 'pred_intent': labels})
data.to_csv('data.csv', index=False)
# model = Pipeline([
#     ('vect', CountVectorizer(ngram_range=(1, 1), max_features=None)),
#     ('svd', TruncatedSVD(n_components=300, random_state=42)),
#     ('clf', KNeighborsClassifier(n_neighbors=10))
# ])
