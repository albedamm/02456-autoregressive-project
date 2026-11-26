import math
import random
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from protgen.data import IGNORE_INDEX, ProteinDataset
from protgen.models.conditional_transformer import ConditionalProteinTransformer, null_label

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 6.1 cell 2
print(f"Device: {DEVICE}")


###########################
### 0. HYPERPARAMETERS ####
###########################
parser = argparse.ArgumentParser()
parser.add_argument("--learning-rate", type=float, default=1e-3)        # Default is set from first round of tuning
parser.add_argument("--dropout", type=float, default=0.1)               # Default is set from first round of tuning
parser.add_argument("--label-dropout", type=float, default=0.1)         # Default is set from first round of tuning
parser.add_argument("--num-epochs", type=int, default=45)               # Default is set from first round of tuning
parser.add_argument("--d-model", type=int, default=128)
parser.add_argument("--num-heads", type=int, default=4)
parser.add_argument("--num-layers", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)                      # Random seed, so runs can be reproduced and repeated
parser.add_argument("--save-dir", type=str, default="trained_models/trained_cond_transformer_models")
args = parser.parse_args()

# Fix the random seed (weight init, dropout and data shuffling)
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)    # also seeds all CUDA devices

# Model size, change if needed
d_model = args.d_model
num_heads = args.num_heads
num_layers = args.num_layers
d_ff = 4 * d_model    # Feed-forward hidden size, 4x d_model as in Vaswani et al.
dropout = args.dropout
label_dropout = args.label_dropout

# Training
batch_size = 64       # From exercise 6.1
learning_rate = args.learning_rate
num_epochs = args.num_epochs
max_len = 129


#####################
### 1. LOAD DATA ####
#####################
train_data = ProteinDataset("data/train.csv")
val_data = ProteinDataset("data/val.csv")
#print(train_data)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

vocab_size = train_data.vocab_size        # 23 = 3 special tokens + 20 amino acids
PAD = train_data.vocab["<PAD>"]
BOS = train_data.vocab["<BOS>"]
EOS = train_data.vocab["<EOS>"]


###############
### 2. LOSS ###
###############

criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)


###################
### 3. SAMPLING ###
###################

@torch.no_grad()  # same as `with torch.no_grad():` in exercise 6.1
def sample(model, batch_size=10, temperature=1.0, max_len=max_len, label=0):
    model.eval()
    token_ids = torch.full((batch_size, 1), BOS, dtype=torch.long, device=DEVICE)
    labels = torch.full((batch_size,), label, dtype=torch.long, device=DEVICE)
    for t in range(max_len - 1):
        logits = model(token_ids, labels)
        logits_t = logits[:, -1:] / temperature
        forbidden = torch.tensor([PAD, BOS], device=DEVICE).view(1, 1, -1).expand(batch_size, 1, -1)
        logits_t.scatter_(index=forbidden, dim=2, value=-math.inf)
        p_wt = torch.distributions.Categorical(logits=logits_t)
        tokens_t = p_wt.sample()                                      # (B, 1)
        token_ids = torch.cat([token_ids, tokens_t], dim=1)
        if (token_ids == EOS).any(dim=1).all():
            break

    # cut each sequence at its first <EOS> and turn ids into amino-acid letters
    sequences = []
    for seq in token_ids.cpu():
        eos_positions = (seq == EOS).nonzero()
        if len(eos_positions) > 0:
            seq = seq[: eos_positions[0, 0]]
        sequences.append(train_data.decode(seq))
    return sequences


###################
### 4. TRAINING ###
###################

if __name__ == "__main__":
    model = ConditionalProteinTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_len=max_len,
    )
    model = model.to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)     # From exercise 6.1

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"d{d_model}_h{num_heads}_l{num_layers}_lr{learning_rate}_drop{dropout}_labeldrop{label_dropout}_e{num_epochs}_s{seed}"
    checkpoint_file = save_dir / f"{model_name}.pth"

    train_losses, val_losses, val_losses_uncond = [], [], []
    best_val_loss = float("inf")

    # Epoch loop with a train pass and a validation pass from exercise Week 3
    # The inside of the training from execise 6.1
    for epoch in range(num_epochs):
        ## Train
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{num_epochs}")
        for step, (x, x_shifted, y, length) in enumerate(pbar):
            x, x_shifted, y = x.to(DEVICE), x_shifted.to(DEVICE), y.to(DEVICE)

            # Replace some labels with the null label, so the network also learns unconditional generation
            drop = torch.rand(y.size(0), device=DEVICE) < label_dropout
            labels = y.clone()
            labels[drop] = null_label


            # Forward through the model
            optimiser.zero_grad()
            logits = model(x, labels)

            # CrossEntropyLoss expects logits as (batch, vocab, sequence).
            loss = criterion(logits.transpose(1, 2), x_shifted)

            # Backward and optimize
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            if step % 20 == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}")
        train_losses.append(epoch_loss / len(train_loader))

        ## Validate (no gradients, dropout off)
        model.eval()
        val_loss = 0.0
        val_loss_uncond = 0.0
        with torch.no_grad():
            for x, x_shifted, y, length in val_loader:
                x, x_shifted, y = x.to(DEVICE), x_shifted.to(DEVICE), y.to(DEVICE)
                # Compute the conditional loss (for training)
                val_loss += criterion(model(x, y).transpose(1, 2), x_shifted).item()

                # Also compute the unconditional loss (for monitoring, not for training)
                null_labels = torch.full_like(y, null_label)
                val_loss_uncond += criterion(model(x, null_labels).transpose(1, 2), x_shifted).item()
        val_losses.append(val_loss / len(val_loader))
        val_losses_uncond.append(val_loss_uncond / len(val_loader))
        

        print(f"Epoch {epoch+1}: train loss {train_losses[-1]:.4f}, "
              f"val loss {val_losses[-1]:.4f} (uncond {val_losses_uncond[-1]:.4f}), "
              f"val perplexity {math.exp(val_losses[-1]):.2f}")

        # Save checkpoint (from exercise 6.1, saves every 50 steps; we keep the best val model)
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save({"state_dict": model.state_dict(),
                        "config": {
                            "vocab_size": vocab_size,
                            "d_model": d_model,
                            "n_heads": num_heads,
                            "d_ff": d_ff,
                            "num_layers": num_layers,
                            "dropout": dropout,
                            "max_seq_len": max_len,
                            },
                        "label_dropout": label_dropout,
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                    }, checkpoint_file)


        # Print what the model generates right now
        print("  sample:", sample(model, batch_size=1)[0][:60], "...")

    # Loss curve, as in src/train/train_mlp.py
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.plot(val_losses_uncond, label="Validation loss (unconditional)")
    
    plt.xlabel("Epoch")
    plt.ylabel("Negative log-likelihood per token")
    plt.title(
        "Classifier-free Transformer training and validation loss\n"
        f"learning rate={learning_rate}, dropout={dropout}, label_dropout={label_dropout}\n"
        f"epochs={num_epochs}, d_model={d_model}, heads={num_heads}, layers={num_layers}, seed={seed}"
    )
    
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f"{model_name}_loss_curve.png")

    pd.DataFrame({
        "epoch": range(1, num_epochs + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_loss_uncond": val_losses_uncond,
    }).to_csv(save_dir / f"{model_name}_losses.csv", index=False)