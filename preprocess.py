import argparse
import json
import os

import torch
from sentence_transformers import SentenceTransformer

from src.utils import get_intent_labels, is_float


def construct_sentence_embedding(args):
    with open(args.description_path, "r") as json_file:
        description = json.load(json_file)
        json_file.close()

    intent_list = get_intent_labels(args)
    intent2des = {}
    for label in intent_list:
        intent2des[label] = description[label]

    sentence_bert = SentenceTransformer(args.sentence_bert_path)
    intent_embeddings = sentence_bert.encode(
        list(intent2des.values()), convert_to_tensor=True
    ).cpu()
    torch.save(intent_embeddings, args.intent_embedding_path)


def deduplicate_data(sents, intents, slots):
    new_inputs, new_intents, new_slots = [], [], []
    set_sents = list()
    for sent, intent, slot in zip(sents, intents, slots):
        if sent in set_sents:
            continue
        set_sents.append(sent)
        new_inputs.append(sent)
        new_intents.append(intent)
        new_slots.append(slot)
    if len(sents) != len(new_inputs):
        print(f"Before:{len(input)} - Now:{len(new_inputs)}")
    return new_inputs, new_intents, new_slots


def fixed_label(args):
    with open(os.path.join(args.data_dir, args.token_level, args.seq_in_path), "r") as f:
        sents = f.readlines()
        f.close()
    sents = [text.strip() for text in sents]

    with open(os.path.join(args.data_dir, args.token_level, args.seq_out_path), "r") as f:
        slot_labels = f.readlines()
        f.close()
    slot_labels = [slot.strip() for slot in slot_labels]

    with open(os.path.join(args.data_dir, args.token_level, args.label_path), "r") as f:
        intent_labels = f.readlines()
        f.close()
    intent_labels = [label.strip() for label in intent_labels]

    print("Before: Number: ", len(sents))
    new_sents, new_slot_labels, new_labels = [], [], []
    for sent, slot, label in zip(sents, slot_labels, intent_labels):
        # fix intent
        new_label = None
        if (
            "thay đổi" in sent
            and not any(x in sent for x in ["tăng", "giảm", "màu", "xám", "xanh", "trắng", "tím"])
            and "set.level" not in label
        ):
            new_label = "smart.home.set.level"
        elif not any(x in sent for x in ["phần"]) and "bật" in sent and "percentage" in label:
            new_label = "smart.home.device.onoff"
        elif not any(x in sent for x in ["phần"]) and "giảm" in sent and "percentage" in label:
            new_label = "smart.home.decrease.level"
        elif not any(x in sent for x in ["phần"]) and "tăng" in sent and "percentage" in label:
            new_label = "smart.home.increase.level"
        elif (
            "set.level" in label
            and any(x in sent for x in ["bật"])
            and not any(x in sent for x in ["cấp", "mức", "giảm", "tăng"])
        ):
            new_label = "smart.home.device.onoff"
        elif "set.percentage" in label and "phần" not in sent:
            if any(x in sent for x in ["đóng", "bật", "mở"]):
                new_label = "smart.home.device.onoff"
            elif "vui vẻ" in sent:
                new_label = "greeting"
            else:
                new_label = "smart.home.set.level"
        elif "percentage" in label and len(sent.split()) < 5 and "phần" not in sent:
            new_label = "smart.home.set.level"
        else:
            new_label = label

        # fix slot
        if "xuống" in sent or "tăng" in sent or "giảm" in sent:
            if "final-valuesyspercentage" in slot:
                slot = slot.replace("final-valuesyspercentage", "change-valuesyspercentage")

        slot = slot.split()
        # if 'device' in short:
        new_slot = []
        if (
            "bóng" in sent
            and "set.color" in new_label
            and not slot[sent.split().index("bóng")] != "O"
        ):
            idx = sent.split().index("bóng")
            assert slot[idx + 1] == "B-devicedevice"
            for i in range(len(slot)):
                if i == idx:
                    new_slot.append("B-devicedevice")
                elif i == idx + 1:
                    new_slot.append("I-devicedevice")
                else:
                    new_slot.append(slot[i])
        elif (
            "đèn bóng trần" in sent
            and "decrease.level" in new_label
            and slot[sent.split().index("đèn")] == "O"
        ):
            idx = sent.split().index("đèn")
            assert slot[idx : idx + 3] == ["O"] * 3
            for i in range(len(slot)):
                if i == idx:
                    new_slot.append("B-devicedevice")
                elif i == idx + 1 or i == idx + 2:
                    new_slot.append("I-devicedevice")
                else:
                    new_slot.append(slot[i])
        else:
            new_slot = [x for x in slot]

        sent = sent.split()
        for i in range(len(new_slot)):
            if i > 0 and (
                (i == len(new_slot) - 1 and is_float(sent[i]) and is_float(sent[i - 1]))
                or (
                    i < len(new_slot) - 1
                    and is_float(sent[i])
                    and is_float(sent[i - 1])
                    and sent[i + 1] == "phần"
                )
            ):
                if new_slot[i] == "sysnumbersysnumber" and new_slot[i - 1] == "roomroom":
                    new_slot[i] = "floornumberfloornumber"

        sent = " ".join(sent)
        new_slot = " ".join(new_slot)

        if new_label in [
            "smart.home.decrease.percentage",
            "smart.home.set.percentage",
            "smart.home.increase.percentage",
        ]:
            if "percentage" not in new_slot:
                continue

        new_sents.append(sent)
        new_slot_labels.append(new_slot)
        new_labels.append(new_label)

    # de duplicate
    sents, slot_labels, intent_labels = deduplicate_data(new_sents, new_slot_labels, new_labels)
    print("After: Number: ", len(sents))
    assert len(sents) == len(slot_labels) == len(intent_labels)
    if not os.path.exists(
        os.path.join(args.data_dir, args.token_level, args.fixed_seq_in_path.split("/")[0])
    ):
        os.makedirs(
            os.path.join(args.data_dir, args.token_level, args.fixed_seq_in_path.split("/")[0])
        )

    with open(os.path.join(args.data_dir, args.token_level, args.fixed_seq_in_path), "w") as f:
        f.write("\n".join(sents))
        f.close()
    with open(os.path.join(args.data_dir, args.token_level, args.fixed_seq_out_path), "w") as f:
        f.write("\n".join(slot_labels))
        f.close()
    with open(os.path.join(args.data_dir, args.token_level, args.fixed_label_path), "w") as f:
        f.write("\n".join(intent_labels))
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sentence_bert_path", default="pretrained-models/retrieval-miniLM-L12", type=str
    )
    # parser.add_argument("--sentence_bert_path", default="all-MiniLM-L12-v2", type=str)
    parser.add_argument(
        "--intent_embedding_path", default="data/syllable-level/intent_embedding.pt", type=str
    )
    parser.add_argument(
        "--description_path", default="data/syllable-level/description.json", type=str
    )
    parser.add_argument("--fixed_label", action="store_true")
    parser.add_argument("--fixed_seq_in_path", default="fixed_train/seq.in", type=str)
    parser.add_argument("--fixed_seq_out_path", default="fixed_train/seq.out", type=str)
    parser.add_argument("--fixed_label_path", default="fixed_train/label", type=str)

    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--token_level", default="syllable-level", type=str)
    parser.add_argument("--seq_in_path", default="train_dev/seq.in", type=str)
    parser.add_argument("--seq_out_path", default="train_dev/seq.out", type=str)
    parser.add_argument("--label_path", default="train_dev/label", type=str)
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str)
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str)

    args = parser.parse_args()
    construct_sentence_embedding(args)

    if args.fixed_label:
        fixed_label(args)
