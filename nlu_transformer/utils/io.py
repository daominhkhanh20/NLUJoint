import json
import os
import glob
from nlu_transformer.utils.process import preprocess_text


def get_list_intent_labels(path):
    return [
        label.strip()
        for label in open(path, 'r', encoding='utf-8')
    ]


def get_list_slot_labels(path):
    return [
        label.strip()
        for label in open(path, 'r', encoding='utf-8')
    ]


def get_config_architecture(model_path):
    model_path = os.path.abspath(model_path)
    architecture_files = glob.glob(model_path + "/*architecture.json")
    if len(architecture_files) == 0:
        raise Exception(f"File config architecture in {model_path} not found")
    else:
        file_config = architecture_files[0]
        with open(f"{file_config}", "r") as file:
            config_architecture = json.load(file)

        return config_architecture


def read_file(path):
    return [
        sentence.strip()
        for sentence in open(path, 'r')
    ]
