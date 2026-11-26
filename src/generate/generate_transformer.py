import torch
from protgen.models.transformer import ProteinTransformer
from protgen.data import ProteinDataset
from pathlib import Path
import pandas as pd

# Hyperparameter settings
batch_size = 64
d_model = 384
num_heads = 12
num_layers = 4
d_ff = 4 * d_model
dropout = 0.2
max_len = 129
vocab_size = 23

# The number of sequences to generate
num_sequences = 20
generated_sequences = []

# Build the MLP model architecture with the specified hyperparameters
model = ProteinTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_len=max_len
    )

# Define the path to the trained model file
model_path = "trained_models/trained_transformer_models/stage1/d384_h12_l4_lr0.001_drop0.2_e60_s0.pth"

# Load the trained model weights into the MLP model
model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

# Set the model to evaluation mode
model.eval()

# Use the ProteinDataset function to load the decode function
dataset = ProteinDataset("data/train.csv")

# Function to generate a protein sequence using the trained transformer model
def generate_sequence(model, max_length=max_len-2):
    BOS, EOS = 1, 2 # special tokens 
    context = torch.tensor([[BOS]], dtype=torch.long)   # start with just <BOS> (1, 1)
    generated_tokens = [] #initialize list for amino acids generated so far

    # no gradients - only generate
    with torch.no_grad():

        # iterate over each amino acid (one added at a time)
        for _ in range(max_length):
            logits = model(context)[:, -1, :] # scores for the next token (1, vocab)
            logits[:, :2] = -float("inf") # block <PAD> and <BOS> (-inf gives prob 0 after activation)
            probabilities = torch.softmax(logits, dim=-1) # scores to probabilities
            next_token = torch.multinomial(probabilities, num_samples=1) # draw the next token from the probabilities

            # if that token is end of sequence, we stop
            if next_token.item() == EOS:
                break

            # append the token id and add the context for the next step
            generated_tokens.append(next_token.item())
            context = torch.cat([context, next_token], dim=1)   # grow, don't slide

    # return lit of ids, without <BOS> and <EOS>
    return generated_tokens

# Generate and decode one sequence at a time using the generate_sequence func:
for i in range(num_sequences):
    generated_tokens = generate_sequence(model)
    protein_sequence = dataset.decode(generated_tokens)
    print("Generated token IDs:", generated_tokens)
    print("Generated protein:", protein_sequence)
    print("Length of generated protein:", len(protein_sequence))

    generated_sequences.append(protein_sequence)

# Save sequences to CSV
save_path = Path("results/sequences/transformer_sequences.csv")
df = pd.DataFrame({"sequence": generated_sequences})
df.to_csv(save_path, index=False)
print(f"Saved {num_sequences} sequences to {save_path}")
