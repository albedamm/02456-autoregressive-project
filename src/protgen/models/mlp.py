import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from torch.nn.parameter import Parameter

# Define network
class ProteinMLP(nn.Module):

    def __init__(self, num_features, num_hidden, num_output, embedding_dim=16):
        super(ProteinMLP, self).__init__()

        # The learned representation of input tokens
        self.embedding = nn.Embedding(
            num_embeddings=num_output,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # input layer
        self.W_1 = Parameter(init.xavier_normal_(torch.Tensor(num_hidden, num_features)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))
        # hidden layer
        self.W_2 = Parameter(init.xavier_normal_(torch.Tensor(num_output, num_hidden)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_output), 0))
        # define activation function in constructor
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.embedding(x)
        x = x.flatten(start_dim=1)

        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        x = F.linear(x, self.W_2, self.b_2)
        return x

