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

        # A learnable embedding matrix
        # 23 possible embedding tokens and 16 dimensions
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
        # Each of the 3 input tokens are embedded into a 16-dimensional vector
        x = self.embedding(x)
        # Concatenatig the 3 input vectors into one long vector (3 x 16 = 48 elements)
        # start_dim = 1 ensures that the batch size is preserved
        x = x.flatten(start_dim=1)

        # Transform the 48 input features into N hidden features (could be 64)
        x = F.linear(x, self.W_1, self.b_1)
        # Non-linearity
        x = self.activation(x)
        # Transforms the N hidden features into the 23 possible output tokens
        # Logits that represents a score for how likely a particular amino acid (or special token) is to be the next token in the sequence
        x = F.linear(x, self.W_2, self.b_2)
        return x

