import os

import numpy as np
import pandas as pd
from scipy import stats
import argparse
import ast
from tqdm import tqdm
import numpy
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--path_model', type=str, default='models/bkai')
parser.add_argument('--path_save_result', type=str, default='models')
args = parser.parse_args()


all_results_file = glob.iglob(f"{args.path_model}/**/*.csv", recursive=True)
cnt_sent = 0
data = []
for file in list(all_results_file):
    df = pd.read_csv(file, names=['intent', 'tag'])
    if isinstance(df['tag'][0], str):
        df['tag'] = df['tag'].apply(lambda x: x[1:].split(" "))
    data.append(df)
    cnt_sent = len(df)

final_intents, final_slots = [], []
for idx in range(cnt_sent):
    temp_intents = [df['intent'][idx] for df in data]
    final_intents.append(stats.mode(temp_intents)[0][0])

    temp_slots = [df['tag'][idx] for df in data]
    temp_slots = np.array(temp_slots)
    temp_slots = list(stats.mode(temp_slots, axis=0)[0][0])
    temp_slots = " " + " ".join(temp_slots)
    final_slots.append(temp_slots)


df_final = pd.DataFrame({'intent': final_intents, 'tag': final_slots})
df_final.to_csv(f"{args.path_save_result}/results.csv", index=False, header=False)
os.chdir(args.path_save_result)
os.system(f"zip results.zip results.csv")





