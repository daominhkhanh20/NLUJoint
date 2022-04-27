import os
import argparse
import pandas as pd
from nlu_transformer.utils.io import read_file

parser = argparse.ArgumentParser()
parser.add_argument('--path_folder_data', type=str, default='assets/data/bkai')
args = parser.parse_args()

for folder in os.listdir(args.path_folder_data):
    if folder != 'test':
        all_sentences = read_file(os.path.join(args.path_folder_data, folder, 'seq.in'))
        all_intents = read_file(os.path.join(args.path_folder_data, folder, 'label'))
        all_slots = read_file(os.path.join(args.path_folder_data, folder, 'seq.out'))
        pd.DataFrame({'text': all_sentences, 'intent': all_intents, 'tag': all_slots}).to_csv(f"{args.path_folder_data}/{folder}/{folder}.csv", index=False)
