import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    " Builds positional encoding table with one row per position (up to max_seq_len) "
    def __init__(self, d_model, dropout, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) # Dropout layer to prevent overfitting

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        # Compute the div_term for the positional encoding formula
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # Apply sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term) # Even columns
        pe[:, 1::2] = torch.cos(position * div_term) # Odd columns

        # Add a batch dimension
        pe = pe.unsqueeze(0)

        # Register pe as a buffer so it is not considered a model parameter but will be saved and loaded with the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        " Add positional encoding to the input tensor x "
        x = x + self.pe[:, :x.size(1)] # Amino acid sequences are of shape (batch_size, seq_len, d_model)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    " Compute 'Scaled Dot Product Attention' "
    # Get the dimension of the query tensor
    d_k = query.size(-1) # used for scaling
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # every query compared with every key

    # Apply the mask (if provided) to the scores tensor, setting masked positions to negative infinity
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -math.inf)

    # Compute the attention probabilities using softmax
    attn_weights = F.softmax(scores, dim = -1)

    # Apply dropout to the attention probabilities (if provided)
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    return torch.matmul(attn_weights, value), attn_weights


class MultiHeadedAttention(nn.Module):
    """ Implements multi-head attention mechanism as described in "Attention is All You Need" paper """
    
    def __init__(self, n_heads, d_model, dropout=0.1):
        " Take in model size and number of heads "
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0
        
        # Assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn_weights = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        " Implements Multi-Head Attention mechanism "
        batch_size = query.size(0)
        
        if mask is not None:
            # Same mask applied to all n_heads
            mask = mask.unsqueeze(1)

        # 1) Do all the linear projections in batch from d_model => n_heads x d_k 
        query, key, value = [l(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn_weights = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.linears[-1](x)
    

def causal_mask(T, device):
    " Create a causal mask for self-attention to prevent attending to future positions "
    mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0) # (1, T, T)
    
    return mask


def feed_forward(d_model, d_ff, dropout=0.1):
    " Position-wise Feed-Forward Network "
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model)
    )

class DecoderBlock(nn.Module):
    """ Causal (decoder-only) Transformer block """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadedAttention(n_heads, d_model, dropout)
        self.feed_forward = feed_forward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        " Forward pass for the decoder block "
        # Self-attention sublayer
        attn_output = self.self_attn(x, x, x, mask=mask)
        x = x + self.dropout1(attn_output)  # Residual connection
        x = self.norm1(x)  # Layer normalization

        # Feed-forward sublayer
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)  # Residual connection
        x = self.norm2(x)  # Layer normalization
        
        return x
    

class ProteinTransformer(nn.Module):
    """ Causal (decoder-only) Transformer model for protein sequences """
    
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=256, num_layers=3, dropout=0.1, max_seq_len=129):
        super(ProteinTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, return_hidden=False):
        " Forward pass for the ProteinTransformer model "
        # Create a causal mask to prevent attending to future positions
        mask = causal_mask(x.size(1), x.device)

        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)

        # Pass through each decoder block
        for layer in self.layers:
            x = layer(x, mask=mask)

        if return_hidden:
            return x

        # Output layer to produce logits for each token in the vocabulary
        logits = self.output_layer(x)
        
        return logits