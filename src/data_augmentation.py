import json
import logging
import os
import random

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from src.utils import get_slot_labels, is_float, seed_everything

logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
fill_mask_model = pipeline("fill-mask", model="xlm-roberta-base", tokenizer=tokenizer)


def replacement(sentence: str, tag: str, fill_mask_model=None):
    list_tags = tag.split()
    new_sentence = sentence.split().copy()
    length = len(new_sentence)

    index = -1
    count = 0
    while index == -1 or list_tags[index] != "O":
        index = np.random.randint(length)
        count += 1
        if count >= 1000:
            return None

    new_sentence[index] = "<mask>"
    new_sentence = " ".join(new_sentence)

    outputs = fill_mask_model(new_sentence)
    for s in outputs:
        if s["sequence"] != sentence:
            return s["sequence"]


def random_add(sentence: str, tag: str, fill_mask_model=None):
    list_tags = tag.split()
    new_sentence = sentence.split().copy()
    length = len(new_sentence)

    index = np.random.randint(0, length)
    new_sentence.insert(index, "<mask>")
    list_tags.insert(index, "O")
    new_sentence = " ".join(new_sentence)

    outputs = fill_mask_model(new_sentence)
    s = outputs[0]["sequence"]
    return s, " ".join(list_tags)


def construct_dictionary(args, texts, slots):
    tagging_labels = get_slot_labels(args)
    slot_labels = []
    for tag in tagging_labels:
        if "-" not in tag:
            if tag not in slot_labels:
                slot_labels.append(tag)
        else:
            slot = tag[2:]
            if slot not in slot_labels:
                slot_labels.append(slot)
    tagging_dict = {tag: [] for tag in slot_labels}

    num_samples = len(texts)
    for i in tqdm(range(num_samples), total=num_samples, desc="Creating dictionary..."):
        tokens = texts[i].split()
        tags = slots[i].split()
        assert len(tokens) == len(tags)

        word = ""
        for j in range(len(tags)):
            if tags[j].startswith("B-"):
                word += tokens[j] + " "
                # End of entity
                if (
                    j == len(tokens) - 1
                    or tags[j + 1] == "O"
                    or tags[j].replace("B-", "").replace("I-", "")
                    != tags[j + 1].replace("B-", "").replace("I-", "")
                ):
                    entity = tags[j][2:]
                    if entity == "floornumberfloornumber":
                        if not is_float(word.strip()):
                            tagging_dict[entity].append(word.strip())
                            word = ""
                    else:
                        tagging_dict[entity].append(word.strip())
                        word = ""
            elif tags[j].startswith("I-"):
                word += tokens[j] + " "
                if (
                    j == len(tokens) - 1
                    or tags[j + 1] == "O"
                    or tags[j].replace("B-", "").replace("I-", "")
                    != tags[j + 1].replace("B-", "").replace("I-", "")
                ):
                    entity = tags[j][2:]
                    tagging_dict[entity].append(word.strip())
                    word = ""
    tagging_dict = {k: list(set(v)) for k, v in tagging_dict.items()}
    tagging_dict = added_dictionary(tagging_dict)
    tagging_dict = filter_dictionary(tagging_dict)
    with open(args.tagging_dict_path, "w") as json_file:
        json.dump(tagging_dict, json_file, indent=4, ensure_ascii=False)
        json_file.close()


def added_dictionary(tagging_dict):
    tagging_dict["allall"].extend(["tất cả", "toàn bộ"])
    tagging_dict["devicedevice"].extend(["cột đèn thứ 3", "đèn downlight 3", "ốp trần 4"])

    return tagging_dict


def filter_dictionary(tagging_dict):
    for word in ["thông báo", "tăng đến", "chế độ", "hạ thấp", "đúng không"]:
        tagging_dict["commandcommand"].remove(word)
    for word in ["đóng hay mở", "mở hay đóng", "đang đóng", "bật hay tắt"]:
        tagging_dict["statusstatus"].remove(word)
    return tagging_dict


def get_threshold(tags):
    for tag in tags:
        if "allall" in tag or "statusstatus" in tag:
            return 0.4
    for tag in tags:
        if (
            "colorcolor" in tag
            or "commandcommand" in tag
            or "floornumberfloornumber" in tag
            or "sysnumbersysnumber" in tag
        ):
            return 0.45
    return 0.7


def mixup_augmentation(data, slots, intents, dictionary_path):
    with open(dictionary_path, "r", encoding="utf-8") as f:
        dictionary = json.load(f)
        f.close()

    greeting_sentence = []
    new_data, new_slots, new_intents = list(), list(), list()
    for i in tqdm(range(len(data))):
        tokens = data[i].split()
        tags = slots[i].split()

        queue = [(tokens, tags, 0)]
        while queue:
            aug_tokens, aug_tags, start_idx = queue.pop(0)
            threshold_to_augment = get_threshold(aug_tags)
            for j in range(start_idx, len(aug_tags)):
                if aug_tags[j].startswith("B-") and aug_tags[j] != "B-commandcommand":
                    # if tags[j].startswith("B-"):
                    tag_type = aug_tags[j].replace("B-", "")
                    if j == len(aug_tags) - 1:
                        pass
                    else:
                        for k in range(j + 1, len(aug_tags) + 1):
                            if j < k < len(aug_tags) and aug_tags[k].replace("B-", "").replace(
                                "I-", ""
                            ) == aug_tags[j].replace("B-", "").replace("I-", ""):
                                continue
                            elif k == len(aug_tags):
                                new_word = ""
                                while new_word == "" or new_word == " ".join(tokens[j:k]):
                                    new_word = random.choice(dictionary[tag_type]).split()
                                new_tag = ["I-" + tag_type] * len(new_word)
                                new_tag[0] = "B-" + tag_type

                                new_sentence = aug_tokens[:j] + new_word + aug_tokens[k:]
                                new_slot = aug_tags[:j] + new_tag + aug_tags[k:]

                                assert len(new_sentence) == len(new_slot)
                                if random.random() >= 0.85:
                                    new_data.append(" ".join(new_sentence))
                                    new_slots.append(" ".join(new_slot))
                                    new_intents.append(intents[i])

                                break

                            elif aug_tags[k].replace("B-", "").replace("I-", "") != aug_tags[
                                j
                            ].replace("B-", "").replace("I-", ""):
                                new_word = ""
                                while new_word == "" or new_word == " ".join(tokens[j:k]):
                                    new_word = random.choice(dictionary[tag_type]).split()
                                new_tag = ["I-" + tag_type] * len(new_word)
                                new_tag[0] = "B-" + tag_type

                                new_sentence = aug_tokens[:j] + new_word + aug_tokens[k:]
                                new_slot = aug_tags[:j] + new_tag + aug_tags[k:]

                                assert len(new_sentence) == len(new_slot)
                                if random.random() >= threshold_to_augment:
                                    new_data.append(" ".join(new_sentence))
                                    new_slots.append(" ".join(new_slot))
                                    new_intents.append(intents[i])

                                    queue.append((aug_tokens, aug_tags, k))

                                break

        if intents[i] == "greeting":
            greeting_sentence.append(data[i])

    augmented_n_greet = 20
    for i in range(augmented_n_greet):
        s1 = random.choice(greeting_sentence)
        s2 = ""
        while s2 == "" or s2 == s1:
            s2 = random.choice(greeting_sentence)

        full_sentence = s1 + " " + s2
        new_data.append(full_sentence)
        new_slots.append(" ".join(["O"] * len(full_sentence.split())))
        new_intents.append("greeting")

    print("Number of augmented data: ", len(new_data))
    return new_data, new_slots, new_intents


def augment_data(args, texts, slots, intents):
    seed_everything(args)

    logger.info(f"Start augment data, first size is: {len(texts)} samples")
    new_texts, new_slots, new_intents = mixup_augmentation(
        texts, slots, intents, dictionary_path=args.tagging_dict_path
    )

    texts.extend(new_texts)
    slots.extend(new_slots)
    intents.extend(new_intents)

    logger.info(f"Finished augment data, new size is: {len(texts)} samples")
    if not os.path.exists(
        os.path.join(args.data_dir, args.token_level, args.augment_seq_in_path.split("/")[0])
    ):
        os.makedirs(
            os.path.join(args.data_dir, args.token_level, args.augment_seq_in_path.split("/")[0])
        )

    with open(os.path.join(args.data_dir, args.token_level, args.augment_seq_in_path), "w") as f:
        f.write("\n".join(texts))
        f.close()
    with open(os.path.join(args.data_dir, args.token_level, args.augment_seq_out_path), "w") as f:
        f.write("\n".join(slots))
        f.close()
    with open(os.path.join(args.data_dir, args.token_level, args.augment_label_path), "w") as f:
        f.write("\n".join(intents))
        f.close()

    logger.info(f"Successfully Augmented.\bThe number of augmented data: {len(texts)} samples")
