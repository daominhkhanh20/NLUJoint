from transformers import AutoTokenizer
from nlu_transformer.processor import JointProcessor

tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

processor = JointProcessor(
    tokenizer=tokenizer,
    max_seq_len=256,
    path_folder_data='assets/data/atis/syllable-level',
    mode='train',
    pad_token_label_id=0
)

features = processor.convert_example_to_features()
print(features[100])
