from nlu_transformer.model import *
from nlu_transformer.utils.io import read_file
from nlu_transformer.inference import InferenceJoint
from nlu_transformer.utils.io import get_config_architecture
from collections import defaultdict
import argparse
import torch
from tqdm import tqdm
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path_save_model', type=str, required=False, default=None)
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--path_folder_test', type=str, default="assets/data/test_set")
parser.add_argument('--epoch', type=int, default=-1)
parser.add_argument('--sent', type=str, required=False)
parser.add_argument('--path_file_predict', type=str, required=False, default='assets/data/bkai/test/seq.in')
parser.add_argument('--is_relabel', default=False, type=lambda x: x.lower() == 'true')
args = parser.parse_args()

config_architecture = get_config_architecture(args.path_save_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_format(sent):
    return " " + sent


if args.epoch == -1:
    path_pretrained = args.path_save_model
else:
    path_pretrained = f"{args.path_save_model}/epoch_{args.epoch}"

model = JointModel.from_pretrained(
    path_pretrained,
    n_intent_label=config_architecture['model']['n_intent_label'],
    n_slot_label=config_architecture['model']['n_slot_label'],
    dropout=config_architecture['model']['dropout'],
    use_intent_context_concat=config_architecture['model']['use_intent_context_concat'],
    use_intent_context_attn=config_architecture['model']['use_intent_context_attn'],
    use_crf=config_architecture['model']['use_crf'],
    intent_loss_coef=config_architecture['model']['intent_loss_coef'],
    pad_token_label_id=config_architecture['model']['pad_token_label_id'],
).to(device)
model.eval()

infer = InferenceJoint(
    model=model,
    config_architecture=config_architecture
)

if args.mode == 'eval':
    if args.is_relabel:
        data = pd.read_csv(f"{args.path_folder_dev}/relabel_dev.csv")
        all_sentences = data['text'].values.tolist()
        intents_truth = data['relabel_intent'].values.tolist()
        slots_truth = data['tag'].values.tolist()
        slots_truth = [convert_format(value) for value in slots_truth]
    else:
        all_sentences = read_file(f"{args.path_folder_dev}/seq.in")
        intents_truth = read_file(f"{args.path_folder_dev}/label")
        slots_truth = read_file(f"{args.path_folder_dev}/seq.out")

    intent_preds, slot_preds = infer.inference(all_sentences)
    assert len(intent_preds) == len(all_sentences)
    # print(all_sentences[64])
    # print(intent_preds[64])
    # print(intents_truth[64])
    # print(slot_preds[64])
    # print(slots_truth[64])

    failed = defaultdict(list)
    index_failed_intents, intent_preds_copy, intents_truth_copy = [], [], []
    index_failed_slots, slots_preds_copy, slots_truth_copy = [], [], []

    for idx in tqdm(range(len(all_sentences)), total=len(all_sentences)):
        if intents_truth[idx] == intent_preds[idx] and slots_truth[idx] == slot_preds[idx]:
            continue
        else:
            if intents_truth[idx] != intent_preds[idx]:
                failed['why_failed'].append('failed_intent')
                index_failed_intents.append(idx)
                intent_preds_copy.append(intent_preds[idx])
                intents_truth_copy.append(intents_truth[idx])

            failed['intent_truth'].append(intents_truth[idx])
            failed['intent_pred'].append(intent_preds[idx])

            failed['text'].append(all_sentences[idx])
            failed['slot_truth'].append(slots_truth[idx])
            failed['slot_pred'].append(slot_preds[idx])
            failed['detail_slot'].append([])
            current_index = len(failed['detail_slot']) - 1
            if slots_truth[idx] != slot_preds[idx]:
                if len(failed['why_failed']) < len(failed['intent_pred']):
                    failed['why_failed'].append('failed_slot')
                else:
                    failed['why_failed'][-1] = 'failed_intent_and_slot'

                index_failed_slots.append(idx)
                slots_preds_copy.append(slot_preds[idx])
                slots_truth_copy.append(slots_truth[idx])
                pred_slot = slot_preds[idx].strip().split(" ")
                truth_slot = slots_truth[idx].strip().split(" ")
                for index, word in enumerate(all_sentences[idx].strip().split(" ")):
                    if pred_slot[index] != truth_slot[index]:
                        failed['detail_slot'][current_index].append({word: {
                            'pred': pred_slot[index],
                            'truth': truth_slot[index]
                        }})

    pd.DataFrame(failed).to_csv(f"{args.path_save_model}/failed.csv", index=False)
    text_failed_intent = [all_sentences[idx] for idx in index_failed_intents]
    pd.DataFrame({'index': index_failed_intents, 'text': text_failed_intent, 'intent_pred': intent_preds_copy,
                  'intent_truth': intents_truth_copy}).to_csv(f"{args.path_save_model}/failed_intent.csv", index=False)
    text_failed_slot = [all_sentences[idx] for idx in index_failed_slots]
    pd.DataFrame({'index': index_failed_slots, 'text': text_failed_slot, 'slot_pred': slots_preds_copy,
                  'slot_truth': slots_truth_copy}).to_csv(f"{args.path_save_model}/failed_slot.csv", index=False)

elif args.mode == 'test':
    if args.sent is not None:
        intents_results, slots_results = infer.inference(args.sent)
        slots_results = " ".join(slots_results)
        print('\n')
        print(f"Sentence: {args.sent}")
        print(f"Intent: {intents_results}")
        print(f"Slot: {slots_results}")
        
elif args.mode == 'evaluate_test':
    infer.evalute_test(path_folder=args.path_folder_test)
