import numpy as np
import sklearn.metrics as sklearn_metrics
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch import Tensor


def cos_sim(a: Tensor, b: Tensor):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_metrics = get_intent_metrics(intent_preds, intent_labels)
    slot_metrics = get_slot_metrics(slot_preds, slot_labels)

    sentence_acc = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    mean_intent_slot = (intent_metrics["intent_f1"] + slot_metrics["slot_f1"]) / 2

    results.update(intent_metrics)
    results.update(slot_metrics)
    results.update(sentence_acc)
    results["mean_intent_slot"] = mean_intent_slot

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds),
        "slot_classification_report": classification_report(labels, preds),
    }


def get_intent_metrics(preds, labels):
    return {
        "intent_accuracy": sklearn_metrics.accuracy_score(labels, preds),
        "intent_f1": sklearn_metrics.f1_score(labels, preds, average="weighted"),
        "intent_classification_report": sklearn_metrics.classification_report(labels, preds),
    }


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = intent_preds == intent_labels

    # Get the slot comparison result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sentence_acc = np.multiply(intent_result, slot_result).mean()
    return {"sentence_frame_accuracy": sentence_acc}


def draw_history(history):
    pass
