import argparse
from torch.utils.data import DataLoader
from transformers import RobertaConfig, AutoTokenizer, XLMRobertaConfig, XLMRobertaTokenizer
from nlu_transformer.dataset import *
from nlu_transformer.model import JointModel

parser = argparse.ArgumentParser()
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--use_intent_context_concat', default=True, type=lambda x: str(x).lower() == 'true')
parser.add_argument('--use_intent_context_attn', default=False, type=lambda x: str(x).lower() == 'true')
parser.add_argument('--attention_embedding_size', type=int, default=200)
parser.add_argument('--use_crf', default=False, type=lambda x: str(x).lower() == 'true')
parser.add_argument('--intent_loss_coef', type=float, default=0.2)
parser.add_argument('--pad_token_label_id', type=float, default=0)

args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
dataset = JointAtisDataset(
    tokenizer=tokenizer,
)

dataloader = DataLoader(dataset, batch_size=2, shuffle=False,
                        collate_fn=JointAtisCollate(
                            pad_token_id=tokenizer.pad_token_id
                        ))
sample = next(iter(dataloader))

config = RobertaConfig.from_pretrained('vinai/phobert-base')
model = JointModel.from_pretrained(
    'vinai/phobert-base',
    config=config,
    args=args,
    n_intent_label=len(dataset.processor.list_intents),
    n_slot_label=len(dataset.processor.list_slots)
)

outputs = model(
    sample['input_ids'],
    sample['attention_mask'],
    sample['token_label_ids'],
    sample['intent_label_ids'],
    sample['slot_label_ids'],
)
print(outputs)
