from typing import List

import torch
from sentence_transformers import SentenceTransformer, util


class Retriever(object):
    def __init__(self, args, model: SentenceTransformer, corpus: List[str] = None):

        self.args = args
        self.model = model
        self.corpus = corpus
        self.embeddings = self.model.encode(
            corpus, convert_to_tensor=True, device=self.args.device, show_progress_bar=True
        )

        self.freeze_model()

    def query(self, text: str, top_k: int = 3):
        query_embedding = self.model.encode(
            text, convert_to_tensor=True, device=self.args.device, show_progress_bar=False
        )

        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k + 3)

        res = []
        for _, idx in zip(top_results[0], top_results[1]):
            if len(res) == top_k:
                break
            if text.strip() != self.corpus[idx].strip():
                res.append(self.corpus[idx])
        return res

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False


def build_retriever(args):
    retrieval_model = SentenceTransformer(args.retrieval_model_path)

    corpus = []
    for line in open(args.corpus_path, "r", encoding="utf-8"):
        corpus.append(line.replace("\n", ""))

    return Retriever(args, retrieval_model, corpus)
