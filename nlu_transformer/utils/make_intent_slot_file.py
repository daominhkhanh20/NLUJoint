import pandas as pd
import os
from tqdm import tqdm

path = '../../assets/data'


def get_list_element(data: list):
    list_element = []
    if len(data) > 0:
        if isinstance(data[0], str):
            list_element = list(set(data))
        elif isinstance(data[0], list):
            temp = set()
            for element in data:
                temp.update(set(element))
            list_element = list(temp)
    list_element.append("UNK")
    return list_element


def write_file(path: str, list_element):
    with open(path, 'w') as file:
        for element in list_element:
            file.write(element + '\n')
    print("write done")


def make_file(path_current: str):
    data = pd.read_csv(path_current + '/train.csv')
    data.tag = data.tag.apply(lambda x: x.split())
    intents = get_list_element(data.intent)
    slots = get_list_element(data.tag)
    write_file(path_current + '/intents.txt', intents)
    write_file(path_current + '/slots.txt', slots)


# for folder in tqdm(os.listdir(path)):
#     path_current = path + '/' + folder
#     if folder == 'comet':
#         make_file(path_current)
#     else:
#         for sub_folder in os.listdir(path_current):
#             path_temp = path_current + '/' + sub_folder
#             make_file(path_temp)


