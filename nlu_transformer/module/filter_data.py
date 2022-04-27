import os
import pandas as pd
from typing import Optional, Dict, Text, List
from tqdm import tqdm
import glob


class FilterData:
    def __init__(self, path_folder_data: Optional[Text] = 'assets/data/bkai',
                 has_relabel: bool = True):
        self.path_folder_data = path_folder_data
        self.has_relabel = has_relabel
        self.percentage_word = ['phần trăm']
        self.on_off_word = ['bật', 'tắt']

    def filter_correct_data(self, path_save_data):
        if os.path.exists(path_save_data) is False:
            os.makedirs(path_save_data, exist_ok=True)

        for folder in os.listdir(self.path_folder_data):
            current_path = os.path.join(self.path_folder_data, folder)
            path_save_current = os.path.join(path_save_data, folder)
            if folder != 'test' and 'aug' not in folder:
                os.makedirs(path_save_current, exist_ok=True)
                if self.has_relabel:
                    file_name = f"relabel_{folder}.csv"
                else:
                    file_name = f"{folder}.csv"

                path_list_intent_label = glob.glob(f"{current_path}/intent*.txt")[0]
                path_list_slot_label = glob.glob(f"{current_path}/slot*.txt")[0]

                path_data = os.path.join(current_path, file_name)
                data = pd.read_csv(path_data)
                list_index_accept = []
                for idx, row in tqdm(data.iterrows(), total=len(data)):
                    # is_true = not any(word in row['text'] for word in self.percentage_word) and 'percentage' in row['intent']
                    # if row['text'] == 'mình muốn tăng bóng vách':
                    #     print(is_true)
                    if not any(word in row['text'] for word in self.percentage_word) and 'percentage' in row['intent']:
                        continue
                    list_index_accept.append(idx)

                data = data[data.index.isin(list_index_accept)].reset_index(drop=True)
                data.to_csv(f'{path_save_current}/{file_name}', index=False)
                os.system(f"cp {path_list_intent_label} {path_save_current}")
                os.system(f"cp {path_list_slot_label} {path_save_current}")
            elif folder == 'test':
                os.system(f"cp -r {current_path} {path_save_data}")







