

# some imports

# ! pip install ipywidgets rich seaborn torch datasets transformers tokenizers sentencepiece sacremoses --quiet

# %matplotlib inline

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch import nn
import math
from functools import partial
from pathlib import Path
from tqdm import tqdm
import rich
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import transformers
import tokenizers
# import datasets
# import zipfile
# from huggingface_hub import hf_hub_download

sns.set_theme()

print("imported succesfully")



#####################
### 0. LOAD DATA ####
#####################
train_data = ProteinDataset("data/train.csv")
val_data = ProteinDataset("data/val.csv")
print(train_data)
# ProteinDataset(num_samples=16020, vocab_size=23, num_classes=3, max_length=129)

x, x_shifted, y, length = train_data[0]
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)



#####################
#### 0.1 Utilities (should be transferred later)
#####################


class MultiHeadAttention(nn.Module):
    """
    From DataCamp course:
    'Transformer Model Tutorial in PyTorch: From Theory to Code'
    """

    def __init__(self, d_model, num_heads):
        """
        The class is defined as a subclass of PyTorch's nn.Module.
        d_model: Dimensionality of the input.
        num_heads: The number of attention heads to split the input into.
        The initialization checks if d_model is divisible by num_heads, and then defines the transformation weights for query, key, value, and output.
        """
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Scaled dot-product attention:

        - Calculating attention scores: attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k). Here, the attention scores are calculated by taking the dot product of queries (Q) and keys (K), and then scaling by the square root of the key dimension (d_k).
        - Applying mask: If a mask is provided, it is applied to the attention scores to mask out specific values.
        - Calculating attention weights: The attention scores are passed through a softmax function to convert them into probabilities that sum to 1.
        - Calculating output: The final output of the attention is calculated by multiplying the attention weights by the values (V).

        """
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        """
        This method reshapes the input x into the shape (batch_size, num_heads, seq_length, d_k). It enables the model to process multiple attention heads concurrently, allowing for parallel computation.
        """
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        """
        After applying attention to each head separately, this method combines the results back into a single tensor of shape (batch_size, seq_length, d_model). This prepares the result for further processing.
        """
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        """
        The forward method is where the actual computation happens:

        Apply linear transformations: The queries (Q), keys (K), and values (V) are first passed through linear transformations using the weights defined in the initialization.
        Split heads: The transformed Q, K, V are split into multiple heads using the split_heads method.
        Apply scaled dot-product attention: The scaled_dot_product_attention method is called on the split heads.
        Combine heads: The results from each head are combined back into a single tensor using the combine_heads method.
        Apply output transformation: Finally, the combined tensor is passed through an output linear transformation.
        """
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
