import torch
from torch import nn, Tensor
import numpy as np


class Attention(nn.Module):
    def __init__(self, dimensions: int):
        super(Attention, self).__init__()
        self.dimensions = dimensions
        self.linear = nn.Linear(self.dimensions * 2, self.dimensions)
        self.dropout = nn.Dropout(0.1)
        self.tanh = nn.Tanh()

    def forward(self, query: Tensor, intent_context: Tensor, attention_mask: Tensor):
        """
        :param query: tensor shape (batch_size * seq_length * d_model)
        :param intent_context: shape (batch_size * 1 * d_model)
        :param attention_mask: shape (batch_size * seq_length)
        :return: batch_size * seq_length * d_model
        """
        attention_scores = torch.bmm(query, intent_context.transpose(1, 2))  # batch_size * seq_length * 1

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(dim=2)
            attention_scores.masked_fill_(attention_mask == 0, -np.inf)
        attention_weights = torch.softmax(attention_scores, dim=1)

        mix_context = torch.bmm(attention_weights, intent_context)  # batch_size * seq_length * d_model
        combined = torch.cat((mix_context, query), dim=2)  # batch_size * seq_length * 2d_model
        outputs = self.tanh(self.linear(combined))  # batch_size * seq_length * d_model
        return outputs, attention_weights

