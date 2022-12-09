import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        self._k_matrix = nn.Linear(input_dim, output_dim)
        self._v_matrix = nn.Linear(input_dim, output_dim)
        self._q_matrix = nn.Linear(input_dim, output_dim)
        self._dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, input_x):
        k_x = self._k_matrix(input_x)
        v_x = self._v_matrix(input_x)
        q_x = self._q_matrix(input_x)

        drop_kx = self._dropout_layer(k_x)
        drop_vx = self._dropout_layer(v_x)
        drop_qx = self._dropout_layer(q_x)

        alpha = F.softmax(torch.matmul(drop_qx.transpose(-2, -1), drop_kx), dim=-1)
        return torch.matmul(drop_vx, alpha)


class Attention(nn.Module):
    """Applies attention mechanism on the `context` using the `query`.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(32, 50, 256)
         >>> context = torch.randn(32, 1, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([32, 50, 256])
         >>> weights.size()
         torch.Size([32, 50, 1])
    """

    def __init__(self, dimensions):
        super(Attention, self).__init__()

        self.dimensions = dimensions
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, query, context, attention_mask):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                over which to apply the attention mechanism.
            output length: length of utterance
            query length: length of each token (1)
        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        # query = self.linear_query(query)

        batch_size, output_len, hidden_size = query.size()
        # query_len = context.size(1)

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        # Compute weights across every context sequence
        # attention_scores = attention_scores.view(batch_size * output_len, query_len)
        if attention_mask is not None:
            # Create attention mask, apply attention mask before softmax
            attention_mask = torch.unsqueeze(attention_mask, 2)
            # attention_mask = attention_mask.view(batch_size * output_len, query_len)
            attention_scores.masked_fill_(attention_mask == 0, -np.inf)
        # attention_scores = torch.squeeze(attention_scores,1)
        attention_weights = self.softmax(attention_scores)
        # attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)
        # from IPython import embed; embed()
        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        # combined = combined.view(batch_size * output_len, 2 * self.dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        # output = self.linear_out(combined).view(batch_size, output_len, self.dimensions)
        output = self.linear_out(combined)

        output = self.tanh(output)
        # output = combined
        return output, attention_weights


class MemoryAttention(nn.Module):
    def __init__(self, n_examples, mask_id=0):
        super(MemoryAttention, self).__init__()

        self.n_examples = n_examples
        self.mask_id = mask_id

    def forward(self, seq_vectors, attention_mask, memory_vectors, memory_attention_mask):
        batch_size, _, hidden_size = seq_vectors.size()

        memory_vectors = memory_vectors.view(batch_size, -1, hidden_size)
        memory_attention_mask = memory_attention_mask.view(batch_size, -1)
        memory_vectors[memory_attention_mask == 0] = 0

        attention_dist = seq_vectors.matmul(memory_vectors.view(batch_size, hidden_size, -1))
        attention_dist = torch.softmax(attention_dist, dim=-1)

        attention_vectors = attention_dist.matmul(memory_vectors)

        return seq_vectors + attention_vectors


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.0):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.0):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
