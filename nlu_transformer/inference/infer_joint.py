import pandas as pd
import torch
import os 
from nlu_transformer.utils.io import * 
from nlu_transformer.utils.process import make_input_model
from transformers import AutoTokenizer
from tqdm import tqdm


class InferenceJoint:
    def __init__(self, model, config_architecture: dict, **kwargs):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(config_architecture['pretrained']['pretrained_name'])
        self.pad_token_label_id = config_architecture['data']['pad_token_label_id']
        self.pad_token_segment_id = config_architecture['data']['pad_token_segment_id']
        self.list_intents = config_architecture['data']['list_intents']
        self.list_slots = config_architecture['data']['list_slots']
        self.max_seq_length = config_architecture['hyperparameter']['max_seq_length']
        self.min_threshold_slot_score = kwargs.get('min_threshold_slot_score', 0.8)

    def inference_one_sentence(self, sentence: str):
        input_model, all_slot_mask = make_input_model(
            sentence,
            self.tokenizer,
            pad_token_label_id=self.pad_token_label_id,
            pad_token_segment_id=self.pad_token_segment_id,
            max_seq_length=self.max_seq_length
        )
        outputs = self.model(*input_model)
        intent_logit, slots_logit = outputs[1]

        intent_pred = torch.argmax(intent_logit, dim=-1).item()

        slots_preds = torch.argmax(slots_logit, dim=-1)  # 1 * seq_len
        slot_prob = torch.softmax(slots_logit, dim=-1).squeeze(dim=0)
        # print(slot_prob.size())

        list_slot_preds = []
        list_index = []
        for i in range(all_slot_mask.size(0)):
            for j in range(all_slot_mask.size(1)):
                if all_slot_mask[i][j] != self.pad_token_label_id:

                    list_slot_preds.append(self.list_slots[slots_preds[i][j]])
                    list_index.append(j)

        # print(torch.max(slot_prob[list_index, :], dim=1))

        return self.list_intents[intent_pred], list_slot_preds

    def inference(self, texts: str or list):
        if isinstance(texts, str):
            intent, slots = self.inference_one_sentence(texts)
            return intent, slots
        else:
            list_intents, list_slots = [], []
            for idx, sentence in tqdm(enumerate(texts), total=len(texts)):
                intent, slots = self.inference_one_sentence(sentence)
                assert len(slots) == len(sentence.split(" "))
                slot = " " + " ".join(slots)
                list_intents.append(intent)
                list_slots.append(slot)
            return list_intents, list_slots
    
    def evalute_test(self, path_folder: str):
        if not os.path.exists(os.path.join(path_folder, 'log_failed')):
            os.makedirs(os.path.join(path_folder, 'log_failed'), exist_ok=True)

        for folder_test in os.listdir(path_folder):
            if 'test' not in folder_test:
                continue
            current_path = os.path.join(path_folder, folder_test)
            df_label = pd.read_csv(os.path.join(current_path, 'label.csv'))
            list_sentence, ground_truth_intents, ground_truth_slots = df_label['sent'].values.tolist(),df_label['intent'].values.tolist(), df_label['slot'].values.tolist()
            pred_intents, pred_slots = self.inference(list_sentence)
            cnt = 0
            tmp_sentence, tmp_truth, tmp_pred = [], [], []
            for idx in range(len(pred_intents)):
                if ground_truth_intents[idx].strip() == pred_intents[idx].strip() and ground_truth_slots[idx].strip() == pred_slots[idx].strip():
                    cnt += 1
                else:
                    tmp_sentence.append(list_sentence[idx])
                    if ground_truth_intents[idx].strip() != pred_intents[idx].strip() and ground_truth_slots[idx].strip() != pred_slots[idx].strip():
                        tmp_truth.append(f"{ground_truth_intents[idx].strip()}|{ground_truth_slots[idx].strip()}")
                        tmp_pred.append(f"{pred_intents[idx].strip()}|{pred_slots[idx].strip()}")
                    elif ground_truth_intents[idx].strip() != pred_intents[idx].strip():
                        tmp_truth.append(ground_truth_intents[idx].strip())
                        tmp_pred.append(pred_intents[idx].strip())
                    elif ground_truth_slots[idx].strip() != pred_slots[idx].strip():
                        tmp_truth.append(ground_truth_slots[idx].strip())
                        tmp_pred.append(pred_slots[idx].strip())
                df = pd.DataFrame({"sentence": tmp_sentence, "truth": tmp_truth, "pred": tmp_pred})
                df.to_csv(f"{os.path.join(path_folder, 'log_failed')}/{folder_test}.csv")
            print(f"Accuracy for {folder_test} = {cnt}/{len(pred_intents)} ~ {cnt/ len(pred_intents)}")
        
