import argparse
import os

# ignore warnings
import warnings

from src.data_loader import load_and_cache_examples
from src.trainer import Trainer
from src.utils import MODEL_CLASSES, MODEL_PATH_MAP, init_logger, load_tokenizer, seed_everything

warnings.filterwarnings("ignore")


def main(args):
    init_logger()
    seed_everything(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer, mode=args.train_type)
    dev_dataset = load_and_cache_examples(args, tokenizer, mode=args.val_type)
    test_dataset = load_and_cache_examples(args, tokenizer, mode=args.test_type)

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.do_train:
        trainer.train()

    if args.eval_train:
        trainer.load_model()
        trainer.evaluate("train")

    if args.eval_dev:
        trainer.load_model()
        trainer.evaluate("dev")

    if args.eval_test:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, type=str, help="The name of the task to train")
    parser.add_argument(
        "--model_dir", default="./trained_models", type=str, help="Path to save, load model"
    )
    parser.add_argument("--log_dir", default="./logs", type=str, help="Path to log directory")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument(
        "--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file"
    )
    parser.add_argument(
        "--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file"
    )
    parser.add_argument(
        "--intent_embedding_path", default="data/syllable-level/intent_embedding.pt", type=str
    )

    parser.add_argument(
        "--model_type",
        default="xlmr",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--token_level",
        type=str,
        default="syllable-level",
        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]",
    )

    parser.add_argument(
        "--tuning_metric", default="loss", type=str, help="Metrics to tune when training"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--train_batch_size", default=64, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size", default=64, type=int, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--max_seq_len",
        default=40,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--num_warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--dropout_rate", default=0.4, type=float, help="Dropout for fully-connected layers"
    )

    parser.add_argument(
        "--logging_steps", type=int, default=200, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps", type=int, default=200, help="Save checkpoint every X updates steps."
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--eval_train", action="store_true", help="Whether to run eval on the train set."
    )
    parser.add_argument(
        "--eval_dev", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--eval_test", action="store_true", help="Whether to run eval on the test set."
    )

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument(
        "--ignore_index",
        default=0,
        type=int,
        help="Specifies a target value that is ignored and does not contribute to the input gradient",
    )

    parser.add_argument(
        "--intent_loss_coef", type=float, default=0.5, help="Coefficient for the intent loss."
    )
    parser.add_argument(
        "--contrastive_rate", default=0.001, help="Coefficient of contrastive learning"
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=15,
        help="Number of un-increased validation step to wait for early stopping",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="Select gpu id")

    # CRF option
    parser.add_argument("--use_crf", default=True, action="store_true", help="Whether to use CRF")

    # Init pretrained
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Whether to init model from pretrained base model",
    )
    parser.add_argument(
        "--pretrained_path", default="./trained_models", type=str, help="The pretrained model path"
    )

    parser.add_argument("--train_type", default="train", type=str, help="Train type")
    parser.add_argument("--val_type", default="dev", type=str, help="Eval type")
    parser.add_argument("--test_type", default="test", type=str, help="Eval type")

    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
