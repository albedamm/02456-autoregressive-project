import torch.nn as nn

from protgen.models.transformer import PositionalEncoding, DecoderBlock, causal_mask

# Define number of classes for conditional input and the null label
num_classes = 3
null_label = 3

class ConditionalProteinTransformer(nn.Module):
    """ Causal (decoder-only) Transformer model for protein sequences with conditional input """
    
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=256, num_layers=3, dropout=0.1, max_seq_len=129):
        super(ConditionalProteinTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.label_embedding = nn.Embedding(num_classes + 1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, labels, return_hidden=False):
        " Forward pass for the ConditionalProteinTransformer model "
        # Create a causal mask to prevent attending to future positions
        mask = causal_mask(x.size(1), x.device)

        # Embedding and positional encoding
        x = self.embedding(x)

        # Add label embeddings to the input embeddings
        x = x + self.label_embedding(labels).unsqueeze(1) 

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through each decoder block
        for layer in self.layers:
            x = layer(x, mask=mask)

        if return_hidden:
            return x

        # Output layer to produce logits for each token in the vocabulary
        logits = self.output_layer(x)
        
        return logits