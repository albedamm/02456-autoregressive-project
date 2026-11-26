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
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--out", type=str, default="results/sequences/cfg_sequences.csv")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


# d_model = 384
# num_heads = 12
# num_layers = 4
# d_ff = 4 * d_model
# dropout = 0.2
# max_len = 129
# vocab_size = 23

rows = []


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
batch_size = args.batch_size
temperature = args.temperature


# Function to generate a protein sequence using the trained transformer model
def cfg_sequence(model, target_label, w, batch_size=64, temperature=1.0, max_length=max_len - 1):
    BOS, EOS = 1, 2
    context = torch.tensor([[BOS]], dtype=torch.long)   # just <BOS>
    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(context)[:, -1, :]
            logits[:, :2] = -float("inf")               # block <PAD> and <BOS>

            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            if next_token.item() == EOS:
                break

            generated_tokens.append(next_token.item())
            context = torch.cat([context, next_token], dim=1)   # grow, don't slide

    return generated_tokens

# Generate and decode a sequence
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
