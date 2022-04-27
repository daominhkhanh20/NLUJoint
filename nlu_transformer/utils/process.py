import torch
from typing import List, Union
from collections import Counter
import itertools
import numpy as np
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_contribution_loss_level(values: Union[List[str], List[List[str]]], list_value: List, type_input: str):
    if isinstance(values[0], list):
        values = list(itertools.chain(*values))

    value_counter = Counter(values)
    sum_count = sum(value_counter.values())
    min_count = min(value_counter.values())
    contribution_loss_level = []
    for value in list_value:
        if value_counter[value] != 0:
            contribution_loss_level.append(sum_count / value_counter[value])
        else:
            logger.info(f"'{type_input} {value}' has count = 0")
            contribution_loss_level.append(0)

    contribution_loss_level = np.array(contribution_loss_level)
    norm = np.linalg.norm(contribution_loss_level)
    contribution_loss_level /= norm
    return torch.Tensor(contribution_loss_level).to(device)


def preprocess_text(x: str):
    x = unicodedata.normalize('NFC', x)
    x = x.lower().strip()
    x = re.sub(
        r'[^a-zA-Z__ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ0-9\s]{2,}|'
        r'\s+[-*+-<>\/]\s+|'
        r'^[^a-zA-Z__ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ0-9]\s+?',
        ' ', x)
    x = re.sub('\s+', " ", x).strip()
    return x


def make_input_model(sentence: str, tokenizer,
                     pad_token_label_id: int = 0,
                     pad_token_segment_id: int = 0,
                     max_seq_length: int = 512,
                     ):
    input_ids, token_type_ids, all_slot_mask = [], [], []
    words = sentence.split(" ")
    for word in words:
        outs = tokenizer(word, add_special_tokens=False)
        if 'token_type_ids' not in outs:
            outs = tokenizer(word, add_special_tokens=False, return_token_type_ids=True)
        input_ids.extend(outs['input_ids'])
        token_type_ids.extend(outs['token_type_ids'])
        all_slot_mask.extend(
            [pad_token_label_id + 1] + [pad_token_label_id] * (len(outs['input_ids']) - 1))
    max_count = max_seq_length - 2
    if len(input_ids) > max_count:
        input_ids = input_ids[: max_count]
        token_type_ids = token_type_ids[: max_count]
        all_slot_mask = all_slot_mask[: max_count]

    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    token_type_ids = [pad_token_segment_id] + token_type_ids + [pad_token_segment_id]
    all_slot_mask = [pad_token_label_id] + all_slot_mask + [pad_token_label_id]

    attention_mask = [1] * len(input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.long).view(1, -1)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).view(1, -1)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).view(1, -1)
    all_slot_mask = torch.tensor(all_slot_mask, dtype=torch.long).view(1, -1)
    return (input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)), all_slot_mask.to(device)


def remove_number(sent: str):
    sent = re.sub('[0-9]+', "", sent)
    return re.sub('\s+', " ", sent).strip()