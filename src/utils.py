import logging
import os
import random

import numpy as np
import torch
from transformers import (
    AdamW,
    AutoTokenizer,
    RobertaConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from models import JoinIDSF

MODEL_CLASSES = {
    "xlmr": (XLMRobertaConfig, JoinIDSF, XLMRobertaTokenizer),
    "phobert": (RobertaConfig, JoinIDSF, AutoTokenizer),
}

MODEL_PATH_MAP = {
    "xlmr": "xlm-roberta-base",
    "phobert": "vinai/phobert-base",
}


def get_intent_labels(args):
    intent_path = os.path.join(args.data_dir, args.token_level, args.intent_label_file)
    with open(intent_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        f.close()
    intent_list = [row.strip() for row in data]
    return intent_list


def get_slot_labels(args):
    slot_path = os.path.join(args.data_dir, args.token_level, args.slot_label_file)
    with open(slot_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        f.close()

    slot_list = [row.strip() for row in data]
    return slot_list


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def seed_everything(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False


def get_optimizer(model, args, t_total):
    # Prepare optimizer and schedule (linear warmup and decay)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                param for name, param in param_optimizer if not any(nd in name for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                param for name, param in param_optimizer if any(nd in name for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_training_steps=t_total, num_warmup_steps=args.num_warmup_steps
    )

    return optimizer, scheduler
