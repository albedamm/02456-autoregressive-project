import torch
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import math
import argparse

from protgen.data import ProteinDataset
from protgen.models.conditional_transformer import ConditionalProteinTransformer, null_label

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 6.1 cell 2
print(f"Device: {DEVICE}")


# Hyperparameter settings
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--num-sequences", type=int, default=100)
parser.add_argument("--w", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 5.0])
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--out", type=str, default="results/sequences/cfg_sequences.csv")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--family", type=str, default="HU")
args = parser.parse_args()


# d_model = 384
# num_heads = 12
# num_layers = 4
# d_ff = 4 * d_model
# dropout = 0.2
# max_len = 129
# vocab_size = 23

# # Define the path to the trained model file
# model_path = "trained_models/trained_transformer_models/stage1/d384_h12_l4_lr0.001_drop0.2_e60_s0.pth"

checkpoint = torch.load(args.model_path, map_location=DEVICE, weights_only=True)

# Build the MLP model architecture with the specified hyperparameters
model = ConditionalProteinTransformer(**checkpoint["config"])

model.load_state_dict(checkpoint["state_dict"])

# # Load the trained model weights into the MLP model
# model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

# Move the model to the appropriate device (GPU or CPU)
model = model.to(DEVICE)

# Set the model to evaluation mode
model.eval()

# Load the dataset for decoding generated sequences
dataset = ProteinDataset("data/train.csv")

# Family to index mapping
fam_to_idx = dataset.fam_to_idx

# Define special token IDs
PAD = dataset.vocab["<PAD>"]
BOS = dataset.vocab["<BOS>"]
EOS = dataset.vocab["<EOS>"]

# Define max length of generated sequence
max_len = checkpoint["config"]["max_seq_len"]
w_values = args.w
temperature = args.temperature
num_sequences = args.num_sequences
family = args.family

rows = []

# Function to generate a protein sequence using the trained transformer model
def cfg_sequence(model, target_label, w, temperature=1.0, max_length=max_len - 1):
    
    # Initialize the context with the beginning-of-sequence token 
    context = torch.tensor([[BOS]], dtype=torch.long, device=DEVICE)  

    
    # Create label tensors for the conditional and null labels 
    cond_labels = torch.tensor([target_label], dtype=torch.long, device=DEVICE)
    null_labels = torch.tensor([null_label], dtype=torch.long, device=DEVICE)

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            
            # Forward pass through the model to get logits for both conditional and null labels
            cond_logits = model(context, labels=cond_labels)[:, -1, :]
            null_logits = model(context, labels=null_labels)[:, -1, :]
        
            logits = (1 + w) * cond_logits - w * null_logits
            
            # Avoid sampling <PAD> and <BOS> tokens by setting their logits to -inf
            logits[:, PAD] = -float("inf") 
            logits[:, BOS] = -float("inf")
            
            # Convert logits to probabilities using softmax 
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            if next_token.item() == EOS:
                break

            generated_tokens.append(next_token.item())
            context = torch.cat([context, next_token], dim=1)   # grow, don't slide

    return generated_tokens

# Generate and decode a sequence
for i in range(num_sequences):
    for w in w_values:
        for label, label_idx in family.items():
            generated_tokens = cfg_sequence(model, target_label=label_idx, w=w, temperature=temperature)
            protein_sequence = dataset.decode(generated_tokens)
            print(f"Generated token IDs (w={w}, label={label}):", generated_tokens)
            print(f"Generated protein (w={w}, label={label}):", protein_sequence)
            print(f"Length of generated protein (w={w}, label={label}):", len(protein_sequence))

            rows.append({"sequence": protein_sequence, "label": label, "w": w})
            
# Save sequences to CSV
save_path = Path("results/sequences/cfg_sequences.csv")

df = pd.DataFrame(rows)
df.to_csv(save_path, index=False)

print(f"Saved {num_sequences} sequences to {save_path}")
