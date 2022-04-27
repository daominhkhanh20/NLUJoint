from nlu_transformer.dataset import *
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
dataset = JointAtisDataset(
    tokenizer=tokenizer,
)

dataloader = DataLoader(dataset, batch_size=2,shuffle=False,
                        collate_fn=JointAtisCollate(
                            pad_token_id=tokenizer.pad_token_id
                        ))
sample = next(iter(dataloader))
print(sample)