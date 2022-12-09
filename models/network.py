import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel

from src.eval_helper import cos_sim

from .module import IntentClassifier, SlotClassifier


class JoinIDSF(RobertaPreTrainedModel):
    """Joint Intent Detection and Slot Filling Model"""

    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JoinIDSF, self).__init__(config)

        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)

        if args.model_type == "phobert":
            self.roberta = RobertaModel(config)  # Load pretrained phobert
        else:
            self.roberta = XLMRobertaModel(config)  # Load pretrained xlmr

        # GPU or CPU
        if torch.cuda.is_available() and not args.no_cuda:
            device = "cuda"
            torch.cuda.set_device(self.args.gpu_id)
        else:
            device = "cpu"

        self.intent_embedding = torch.load(args.intent_embedding_path).to(device)

        self.dropout = nn.Dropout(p=self.args.dropout_rate)

        # mapping dim of intent embedding to hidden size
        self.intent_projection = nn.Linear(self.intent_embedding.size(1), self.config.hidden_size)

        self.i_projection = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        self.s_projection = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        self.intent_classifier = IntentClassifier(
            config.hidden_size, self.num_intent_labels, args.dropout_rate
        )

        self.slot_classifier = SlotClassifier(
            config.hidden_size, self.num_slot_labels, args.dropout_rate
        )

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

        self.init_weights(self.intent_classifier)
        self.init_weights(self.slot_classifier)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids
    ):
        # last_hidden_state
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_reps = self.dropout(pooled_output)
        # slot_reps = self.dropout(sequence_output)

        intent_reps = F.relu(self.i_projection(intent_reps))
        # slot_reps = F.relu(self.s_projection(slot_reps))

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        intent_embedded = self.intent_projection(self.intent_embedding)
        # matrix distance
        intent_dist = cos_sim(intent_reps, intent_embedded)

        total_loss = 0
        if intent_label_ids is not None:
            # 1. Intent Softmax
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(
                intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
            ) + self.args.contrastive_rate * intent_loss_fct(
                intent_dist.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
            )

            total_loss += self.args.intent_loss_coef * intent_loss

            # 2. Slot Softmax
            if self.args.use_crf:
                slot_loss = self.crf(
                    slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction="mean"
                )
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(
                        slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1)
                    )

            total_loss += (1 - self.args.intent_loss_coef) * slot_loss

            return total_loss, intent_logits, slot_logits

        return intent_logits, slot_logits
