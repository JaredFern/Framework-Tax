import torch
import torch.nn as nn


class FeedForwardModel(nn.Module):
    def __init__(self, hidden_dims, activation_function=None):
        super(FeedForwardModel, self).__init__()
        self.model = nn.Sequential()
        for i in range(len(hidden_dims)):
            self.model.add_module(
                "hidden_layer_" + str(i),
                nn.Linear(hidden_dims[i], hidden_dims[i]),
            )
            if activation_function is not None:
                self.model.add_module("hidden_layer_activation_" + str(i), nn.ReLU())

    def forward(self, x):
        return self.model(x)


class RnnModel(nn.Module):
    def __init__(
        self,
        hidden_dims,
        dropout,
        bidirectional,
        activation_function,
    ):
        super(RnnModel, self).__init__()
        self.model = nn.Sequential()
        for i in range(len(hidden_dims) - 1):
            self.model.add_module(
                "hidden_layer_" + str(i),
                nn.RNN(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    nonlinearity=activation_function,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    batch_first=True,
                ),
            )

    def forward(self, x):
        return self.model(x)


class LstmModel(nn.Module):
    def __init__(
        self,
        hidden_dims,
        dropout,
        bidirectional,
        activation_function,
    ):
        super(LstmModel, self).__init__()
        self.model = nn.Sequential()
        for i in range(len(hidden_dims) - 1):
            self.model.add_module(
                "hidden_layer_" + str(i),
                nn.LSTM(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    dropout=dropout,
                    bidirectional=bidirectional,
                    batch_first=True,
                ),
            )

    def forward(self, x):
        return self.model(x)


class Conv1DModel(nn.Module):
    def __init__(
        self,
        num_channels,
        kernel_sizes,
        strides,
        paddings,
        dilations,
        groups,
        activation_function=None,
    ):
        self.model = nn.Sequential()
        for i in range(len(self.num_channels) - 1):
            self.model.add_module(
                "hidden_layer_" + str(i),
                nn.Conv1d(
                    num_channels[i],
                    num_channels[i + 1],
                    kernel_sizes[i],
                    strides[i],
                    paddings[i],
                    dilations[i],
                    groups[i],
                ),
            )

    def forward(self, x):
        return self.model(x)


class Conv2DModel(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(Conv2DModel, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride)

    def forward(self, x):
        return self.conv(x)


class LayerNormModel(nn.Module):
    def __init__(self, hidden_dim):
        self.model = nn.Sequential()
        for i in range(len(hidden_dims) - 1):
            self.model.add_module(nn.LayerNorm(hidden_dims))

    def forward(self, x):
        return self.model(x)


# class BertSelfAttention(nn.Module):
#     """Taken from Hugginface bert_modeling.py"""

#     def __init__(self, hidden_size, num_attn_heads, dropout, position_embedding_type=None):
#         super().__init__()
#         if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
#             config, "embedding_size"
#         ):
#             raise ValueError(
#                 f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
#                 f"heads ({config.num_attention_heads})"
#             )

#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)

#         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
#         self.position_embedding_type = position_embedding_type or getattr(
#             config, "position_embedding_type", "absolute"
#         )
#         if (
#             self.position_embedding_type == "relative_key"
#             or self.position_embedding_type == "relative_key_query"
#         ):
#             self.max_position_embeddings = config.max_position_embeddings
#             self.distance_embedding = nn.Embedding(
#                 2 * config.max_position_embeddings - 1, self.attention_head_size
#             )

#         self.is_decoder = config.is_decoder

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         past_key_value=None,
#         output_attentions=False,
#     ):
#         mixed_query_layer = self.query(hidden_states)

#         # If this is instantiated as a cross-attention module, the keys
#         # and values come from an encoder; the attention mask needs to be
#         # such that the encoder's padding tokens are not attended to.
#         is_cross_attention = encoder_hidden_states is not None

#         if is_cross_attention and past_key_value is not None:
#             # reuse k,v, cross_attentions
#             key_layer = past_key_value[0]
#             value_layer = past_key_value[1]
#             attention_mask = encoder_attention_mask
#         elif is_cross_attention:
#             key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
#             value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
#             attention_mask = encoder_attention_mask
#         elif past_key_value is not None:
#             key_layer = self.transpose_for_scores(self.key(hidden_states))
#             value_layer = self.transpose_for_scores(self.value(hidden_states))
#             key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
#             value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
#         else:
#             key_layer = self.transpose_for_scores(self.key(hidden_states))
#             value_layer = self.transpose_for_scores(self.value(hidden_states))

#         query_layer = self.transpose_for_scores(mixed_query_layer)

#         if self.is_decoder:
#             # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
#             # Further calls to cross_attention layer can then reuse all cross-attention
#             # key/value_states (first "if" case)
#             # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
#             # all previous decoder key/value_states. Further calls to uni-directional self-attention
#             # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
#             # if encoder bi-directional self-attention `past_key_value` is always `None`
#             past_key_value = (key_layer, value_layer)

#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

#         if (
#             self.position_embedding_type == "relative_key"
#             or self.position_embedding_type == "relative_key_query"
#         ):
#             seq_length = hidden_states.size()[1]
#             position_ids_l = torch.arange(
#                 seq_length, dtype=torch.long, device=hidden_states.device
#             ).view(-1, 1)
#             position_ids_r = torch.arange(
#                 seq_length, dtype=torch.long, device=hidden_states.device
#             ).view(1, -1)
#             distance = position_ids_l - position_ids_r
#             positional_embedding = self.distance_embedding(
#                 distance + self.max_position_embeddings - 1
#             )
#             positional_embedding = positional_embedding.to(
#                 dtype=query_layer.dtype
#             )  # fp16 compatibility

#             if self.position_embedding_type == "relative_key":
#                 relative_position_scores = torch.einsum(
#                     "bhld,lrd->bhlr", query_layer, positional_embedding
#                 )
#                 attention_scores = attention_scores + relative_position_scores
#             elif self.position_embedding_type == "relative_key_query":
#                 relative_position_scores_query = torch.einsum(
#                     "bhld,lrd->bhlr", query_layer, positional_embedding
#                 )
#                 relative_position_scores_key = torch.einsum(
#                     "bhrd,lrd->bhlr", key_layer, positional_embedding
#                 )
#                 attention_scores = (
#                     attention_scores + relative_position_scores_query + relative_position_scores_key
#                 )

#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         if attention_mask is not None:
#             # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
#             attention_scores = attention_scores + attention_mask

#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)

#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs = self.dropout(attention_probs)

#         # Mask heads if we want to
#         if head_mask is not None:
#             attention_probs = attention_probs * head_mask

#         context_layer = torch.matmul(attention_probs, value_layer)

#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)

#         outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

#         if self.is_decoder:
#             outputs = outputs + (past_key_value,)
#         return outputs


# class LayerNormModel(torch.nn):
#     def __init
# class TransformerEncoderModel(torch.nn):
#     def __init__(
#         self,
#         input_shape,
#         output_shape,
#         hidden_layers,
#         activation_function,
#         batch_first,
#     ):
#         super().__init__(input_shape, output_shape, hidden_layers, activation_function)

#         self.model = nn.Sequential()
#         for i in range(len(self.num_channels) - 1):
#             self.model.add_module(
#                 "hidden_layer_" + str(i),
#                 nn.TransformerEncoderLayer(
#                     model_dim, num_heads, dim_feedforward=,)
#                 ,
#             )
#             self.model.add_module(
#                 "hidden_layer_activation_" + str(i), activation_function
#             )
