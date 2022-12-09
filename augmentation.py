import argparse
import os

from src.data_augmentation import augment_data, construct_dictionary
from src.utils import init_logger


def run_augmentation(args):
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

    construct_dictionary(args, sents, slot_labels)
    augment_data(args, sents, slot_labels, intent_labels)


if __name__ == "__main__":
    init_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tagging_dict_path", default="data/syllable-level/tagging_dict.json", type=str
    )
    parser.add_argument("--augment_seq_in_path", default="augment_train/seq.in", type=str)
    parser.add_argument("--augment_seq_out_path", default="augment_train/seq.out", type=str)
    parser.add_argument("--augment_label_path", default="augment_train/label", type=str)
    parser.add_argument("--seed", type=int, default=42, help="number of generated examples")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--token_level", default="syllable-level", type=str)
    parser.add_argument("--seq_in_path", default="fixed_train/seq.in", type=str)
    parser.add_argument("--seq_out_path", default="fixed_train/seq.out", type=str)
    parser.add_argument("--label_path", default="fixed_train/label", type=str)
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str)
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str)

    args = parser.parse_args()
    run_augmentation(args)
