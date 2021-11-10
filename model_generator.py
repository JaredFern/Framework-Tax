import torch
import torch.nn as nn


class FeedForwardModel(nn.Module):
    def __init__(self, hidden_dims, activation_function):
        super(FeedForwardModel, self).__init__()
        self.model = nn.Sequential()
        for i in range(len(hidden_dims) - 1):
            self.model.add_module(
                "hidden_layer_" + str(i),
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
            )
            # if activation_function is not None:
            #     self.model.add_module(
            #         "hidden_layer_activation_" + str(i), activation_function
            #     )

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
        activation_function,
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
            self.model.add_module(
                "hidden_layer_activation_" + str(i), activation_function
            )

    def forward(self, x):
        return self.model(x)


class Conv2DdModel(nn.Module):
    def __init__(
        self,
        num_channels,
        kernel_sizes,
        strides,
        paddings,
        dilations,
        groups,
        activation_function,
    ):
        self.model = nn.Sequential()
        for i in range(len(self.num_channels) - 1):
            self.model.add_module(
                "hidden_layer_" + str(i),
                nn.Conv2d(
                    num_channels[i],
                    num_channels[i + 1],
                    kernel_sizes[i],
                    strides[i],
                    paddings[i],
                    dilations[i],
                    groups[i],
                ),
            )
            self.model.add_module(
                "hidden_layer_activation_" + str(i), activation_function
            )

    def forward(self, x):
        return self.model(x)


class MultiheadAttentionModel(nn.Module):
    def __init__(self, embed_dims, kdims, vdims, num_heads):
        super(MultiheadAttentionModel, self).__init__()
        self.model = nn.MultiheadAttention(
            embed_dims, num_heads, kdim=kdims, vdim=vdims, batch_first=False
        )

    def forward(self, x):
        return self.model(x, x, x)


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


# class LayerNormModel(torch.nn):
