# Smart Home Intent Detection and Slot Filling

## Members

Nguyen Hoang Dang - 20194423

Do Quoc An - 20194414

Ha Vu Thanh Dat - 20194424

Pham Nhu Thuan - 20194456

## Installation

- Python version == 3.8

```shell
    conda create --name nlu python=3.8 -y
    conda activate nlu
    pip install -e . --upgrade --use-feature=in-tree-build
```

## Folder Tree

Data Folder Tree (Data Folder):

```shell
├── data
    └── syllable-level
        └── train
            └──seq.in
            └──label
            └──seq.out
        └── dev
            └──seq.in
            └──label
            └──seq.out
        └── public_test
            └──seq.in
            └──label
            └──seq.out
        └── private_test
            └──seq.in
            └──label
            └──seq.out
```

## Training sentence embedding model

Usage:

```bash
python3 train_retriever.py [-h] [--data_path CORPUS_PATH]
                       [--label_path LABEL_PATH]
                       [--pretrained_model_path MODEL_PATH]
                       [--max_seq_length MAX_SEQ_LEN]
                       [--epochs NUM_EPOCHS]
                       [--lr LEARNING_RATE]
                       [--warmup_steps NUM_WARMUP_STEPS]
                       [--no_cuda CUDA_OR_NOT]
                       [--seed SEED]
```

## Pre-processing

Usage:

```bash
python3 preprocess.py [-h] [--sentence_bert_path SBERT_PATH]
                       [--intent_embedding_path INTENT_EMBEDDING_PATH]
                       [--description_path DESCRIPTION_PATH]
                       [--data_dir DATASET_PATH]
                       [--token_level TOKEN_LEVEL]
                       [--seq_in_path SEQ_IN_PATH]
                       [--seq_out_path SEQ_OUT_PATH]
                       [--label_path LABEL_PATH]
                       [--intent_label_file INTENT_LABEL_FILE]
                       [--slot_label_file SLOT_LABEL_FILE]
```

## Augmentation

Usage:

```bash
python3 augmentation.py [-h] [--tagging_dict_path TAG_DICTIONARY_PATH]
                       [--data_dir DATASET_PATH]
                       [--token_level TOKEN_LEVEL]
                       [--seq_in_path SEQ_IN_PATH]
                       [--seq_out_path SEQ_OUT_PATH]
                       [--label_path LABEL_PATH]
                       [--intent_label_file INTENT_LABEL_FILE]
                       [--slot_label_file SLOT_LABEL_FILE]
                       [--no_cuda CUDA_OR_NOT]
                       [--seed SEED]
```

## Training and Evaluation

Usage:

```bash
python3 train.py [-h]
                [--model_dir MODEL_DIR]
                [--data_dir DATA_DIR]
                [--intent_label_file INTENT_LABEL_FILE]
                [--slot_label_file SLOT_LABEL_FILE]
                [--rule_file RULE_FILE]
                [--model_type MODEL_TYPE]
                [--tuning_metric TUNING_METRIC]
                [--seed SEED]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--max_seq_len MAX_SEQ_LEN]
                [--learning_rate LEARNING_RATE]
                [--num_train_epochs NUM_TRAIN_EPOCHS]
                [--weight_decay WEIGHT_DECAY]
                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--adam_epsilon ADAM_EPSILON]
                [--max_grad_norm MAX_GRAD_NORM]
                [--max_steps MAX_STEPS]
                [--warmup_steps WARMUP_STEPS]
                [--dropout_rate DROPOUT_RATE]
                [--logging_steps LOGGING_STEPS]
                [--save_steps SAVE_STEPS]
                [--do_train]
                [--do_eval]
                [--do_eval_dev]
                [--no_cuda]
                [--ignore_index IGNORE_INDEX]
                [--intent_loss_coef INTENT_LOSS_COEF]
                [--token_level TOKEN_LEVEL]
                [--early_stopping EARLY_STOPPING]
                [--gpu_id GPU_ID]
                [--use_crf]
                [--pretrained]
                [--pretrained_path PRETRAINED_PATH]
                [--use_rule_based]
                [--use_filter]
                [--attention_embedding_size ATTENTION_EMBEDDING_SIZE]
                [--slot_pad_label SLOT_PAD_LABEL]
                [--embedding_type EMBEDDING_TYPE]
                [--use_attention_mask]
                [--train_type TRAIN_TYPE]
                [--val_type VAL_TYPE]
                [--test_type TEST_TYPE]
```

## Inference

Optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        Input file for prediction
  --output_file OUTPUT_FILE
                        Output file for prediction
  --model_dir MODEL_DIR
                        Path to save, load model
  --batch_size BATCH_SIZE
                        Batch size for prediction
  --no_cuda             Avoid using CUDA when available

Example:

```bash
python3 inference.py --input_file ./data/syllable-level/private_test/seq.in \
                --output_file ./output/results.csv \
                --model_dir ./trained_models \
                --batch_size 64
```

## Pipelines
