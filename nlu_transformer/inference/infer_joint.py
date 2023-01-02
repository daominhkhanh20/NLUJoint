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
                slot = " ".join(slots)
                list_intents.append(intent)
                list_slots.append(slot)
            return list_intents, list_slots
    
    def evalute_test(self, path_folder: str):
        for folder_test in os.listdir(path_folder):
            current_path = os.path.join(path_folder, folder_test)
            list_sentence = read_file(os.path.join(current_path, 'seq.in'))
            ground_truth_intents, ground_truth_slots = load_label(os.path.join(current_path, 'ground_truth.csv'))
            pred_intents, pred_slots = self.inference(list_sentence)
            cnt = 0
            for idx in range(len(pred_intents)):
                if ground_truth_intents[idx] == pred_intents[idx] and ground_truth_slots[idx] == pred_slots[idx]:
                    cnt += 1
            print(f"Accuracy for {folder_test}: {cnt/ len(pred_intents)}")
        
