import torch
from torch import Tensor, nn
from nlu_transformer.model import Attention


class IntentClassifier(nn.Module):
    def __init__(self, n_intent_labels: int, input_dims: int = 768, dropout: float = 0.1):
        super(IntentClassifier, self).__init__()
        self.n_intent = n_intent_labels
        self.input_dims = input_dims
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dims, n_intent_labels)

    def forward(self, intent_context: Tensor):
        return self.linear(self.dropout(intent_context))


class SlotClassifier(nn.Module):
    def __init__(self, input_size: int,
                 n_intent_labels: int,
                 n_slot_labels: int,
                 use_intent_context_concat: bool = False,
                 use_intent_context_attn: bool = False,
                 attention_embedding_size: int = 200,
                 dropout: float = 0.1):
        super(SlotClassifier, self).__init__()
        self.input_size = input_size
        self.n_intent_labels = n_intent_labels
        self.n_slot_labels = n_slot_labels
        self.use_intent_context_concat = use_intent_context_concat
        self.use_intent_context_attn = use_intent_context_attn
        self.attention_embedding_size = attention_embedding_size

        if self.use_intent_context_concat:
            self.linear_out = nn.Linear(2 * self.attention_embedding_size, self.attention_embedding_size)

        elif self.use_intent_context_attn:
            self.attention = Attention(dimensions=self.attention_embedding_size)

        else:
            raise NotImplementedError("Method isn't implemented")

        self.output_dim = self.attention_embedding_size
        self.linear_slot = nn.Linear(self.input_size, self.attention_embedding_size)
        if self.use_intent_context_concat or self.use_intent_context_attn:
            self.linear_intent_context = nn.Linear(self.n_intent_labels, self.attention_embedding_size)
            self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(self.attention_embedding_size, self.n_slot_labels)

    def forward(self, feature_embedding: Tensor,
                intent_context: Tensor,
                attention_mask: Tensor = None):
        """
        :param feature_embedding: batch_size * seq_length * d_model
        :param intent_context: batch_size * n_intent_label
        :param attention_mask: batch_size * seq_length
        :return: batch_size * seq_length * n_tag_label
        """
        seq_length = feature_embedding.size(1)
        feature = self.linear_slot(feature_embedding)  # batch_size * seq_length * attention_embedding_size
        intent_context = self.softmax(intent_context)
        intent_context_projector = self.linear_intent_context(intent_context)  # batch_size * attention_embedding_size
        intent_context_projector = intent_context_projector.unsqueeze(dim=1)

        if self.use_intent_context_concat:
            intent_context_projector = intent_context_projector.expand(-1, seq_length, -1)
            outs = torch.cat((feature, intent_context_projector), dim=-1)
            outs = self.linear_out(outs)
        elif self.use_intent_context_attn:
            outs, weights = self.attention(feature, intent_context_projector, attention_mask)
        else:
            raise NotImplementedError("method isn't implemented")
        outs = self.linear(self.dropout(outs))
        return outs








