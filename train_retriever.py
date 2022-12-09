import argparse
import logging

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader, Dataset

from src.utils import seed_everything


def prepare_data(args):
    with open(args.data_path, mode="r", encoding="utf-8") as f:
        sents = [sent.replace("\n", "") for sent in f.readlines()]
    with open(args.label_path, mode="r", encoding="utf-8") as f:
        labels = [label.replace("\n", "") for label in f.readlines()]

    training_data = []
    window_size = 16
    for i in range(len(sents) - 1):
        for j in range(i + 1, i + window_size):
            if j < len(sents):
                if labels[i] == labels[j]:
                    training_data.append((sents[i], sents[j]))
    return training_data


class MatchingData(Dataset):
    def __init__(self, training_data):
        self.training_data = training_data

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, index):
        return InputExample(texts=[self.training_data[index][0], self.training_data[index][1]])


def train_model(args, training_data):
    train_dataset = MatchingData(training_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = SentenceTransformer("all-MiniLM-L12-v2")

    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        output_path=args.pretrained_model_path,
        optimizer_params={"lr": args.lr},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/syllable-level/train_dev/seq.in", type=str)
    parser.add_argument("--label_path", default="data/syllable-level/train_dev/label", type=str)

    parser.add_argument(
        "--pretrained_model_path", default="pretrained-models/retrieval-miniLM-L12", type=str
    )

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_seq_length", default=32, type=int)

    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("\n******************************************")
    logging.info(f"\nInput Args: {args} ")

    import os

    if not os.path.exists("pretrained-models/"):
        os.mkdir("pretrained-models/")
    else:
        print("Folder is already exist!")

    seed_everything(args)
    training_data = prepare_data(args)
    train_model(args, training_data)
