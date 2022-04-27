from abc import ABC
import logging
from typing import List

import torch
from torch import Tensor, nn
from transformers import XLMRobertaConfig, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from torchcrf import CRF
from nlu_transformer.model import IntentClassifier, SlotClassifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class JointModel(RobertaPreTrainedModel, ABC):
    def __init__(self, config: XLMRobertaConfig or RobertaConfig,
                 n_intent_label: int,
                 n_slot_label: int,
                 dropout: float = 0.1,
                 use_intent_context_concat: bool = True,
                 use_intent_context_attn: bool = False,
                 attention_embedding_size: int = 200,
                 use_crf: bool = False,
                 intent_loss_coef: float = 0.1,
                 pad_token_label_id: int = 0,
                 contribution_intent_loss_level: Tensor = None,
                 contribution_slot_loss_level: Tensor = None,
                 update_loss: bool = False,
                 loss_acc_coef: float = 1.0,
                 efi_coef: float = 0.2,
                 efs_coef: float = 0.2,
                 **kwargs
    ):
        super(JointModel, self).__init__(config)
        self.n_intent_label = n_intent_label
        self.n_slot_label = n_slot_label
        self.config = config
        self.dropout = dropout
        self.use_crf = use_crf
        self.intent_loss_coef = intent_loss_coef
        self.pad_token_label_id = pad_token_label_id
        self.roberta = RobertaModel(config)
        self.contribution_intent_loss_level = contribution_intent_loss_level
        self.contribution_slot_loss_level = contribution_slot_loss_level
        self.update_loss = update_loss
        self.loss_acc_coef = loss_acc_coef
        self.efi_coef = efi_coef
        self.efs_coef = efs_coef
        self.is_mean_teacher = kwargs.get('is_mean_teacher', False)
        self.intent_classifier = IntentClassifier(
            n_intent_labels=self.n_intent_label,
            input_dims=config.hidden_size,
            dropout=self.dropout
        )

        self.slot_classifier = SlotClassifier(
            input_size=self.config.hidden_size,
            n_intent_labels=self.n_intent_label,
            n_slot_labels=self.n_slot_label,
            use_intent_context_concat=use_intent_context_concat,
            use_intent_context_attn=use_intent_context_attn,
            attention_embedding_size=attention_embedding_size,
            dropout=self.dropout
        )
        if self.is_mean_teacher:
            self.n_sample_train = kwargs.get('n_sample_train')
            self.before_intent_logits = torch.zeros((self.n_sample_train, self.n_intent_label))

        if use_crf:
            self.crf = CRF(num_tags=self.n_slot_label, batch_first=True)

    def forward(self, input_ids: Tensor,
                attention_mask: Tensor,
                token_type_ids: Tensor = None,
                intent_label_ids: Tensor = None,
                slot_label_ids: Tensor = None,
                all_slot_mask: Tensor = None,
                list_index: List = None):
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids
        )
        sequence_output = outputs[0]
        assert sequence_output.size(1) == input_ids.size(1)

        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)

        slot_logits = self.slot_classifier(sequence_output, intent_logits, attention_mask)

        assert slot_logits.size(1) == input_ids.size(1)

        total_loss = 0
        # intent loss
        if intent_label_ids is not None:
            if self.is_mean_teacher:
                intent_logits = self.efi_coef * self.before_intent_logits[list_index] + (1 - self.efi_coef) * intent_logits
            if self.contribution_intent_loss_level is not None:
                intent_loss_fn = nn.CrossEntropyLoss(weight=self.contribution_intent_loss_level)
            else:
                intent_loss_fn = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fn(
                intent_logits.view(-1, self.n_intent_label),
                intent_label_ids.view(-1)
            )
            total_loss += self.intent_loss_coef * intent_loss
            intent_preds = torch.argmax(intent_logits.view(-1, self.n_intent_label), dim=-1)
            if self.update_loss:
                indexs_correct = intent_preds == intent_label_ids.view(-1)

        if slot_label_ids is not None:
            if self.use_crf:
                slot_loss = - self.crf(
                    slot_logits,
                    slot_label_ids,
                    mask=attention_mask.byte(),
                    reduction="mean"
                )
            else:
                if self.contribution_slot_loss_level is not None:
                    slot_loss_function = nn.CrossEntropyLoss(weight=self.contribution_slot_loss_level, ignore_index=self.pad_token_label_id)
                else:
                    slot_loss_function = nn.CrossEntropyLoss(ignore_index=self.pad_token_label_id)

                if attention_mask is not None:
                    active_index_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.n_slot_label)[active_index_loss]
                    active_label_ids = slot_label_ids.view(-1)[active_index_loss]
                    slot_loss = slot_loss_function(
                        active_logits,
                        active_label_ids
                    )
                else:
                    slot_loss = slot_loss_function(
                        slot_logits.view(-1, self.n_slot_label),
                        slot_label_ids.view(-1)
                    )

                if self.update_loss:
                    slot_preds = torch.argmax(slot_logits, dim=-1)
                    cnt = 0
                    for idx in range(slot_preds.size(0)):
                        if indexs_correct[idx]:
                            flag = True
                            for j in range(slot_preds.size(1)):
                                if all_slot_mask[idx][j] != self.pad_token_label_id:
                                    if slot_preds[idx][j] == slot_label_ids[idx][j]:
                                        continue
                                    else:
                                        flag = False
                                        break
                            if flag:
                                cnt += 1

                    loss_acc = 1 - cnt/int(slot_preds.size(0))

            total_loss += (1 - self.intent_loss_coef) * slot_loss
            if self.update_loss:
                total_loss += self.loss_acc_coef * loss_acc

        outputs = ((intent_logits, slot_logits), ) + outputs[2:]
        outputs = (total_loss,) + outputs
        return outputs






