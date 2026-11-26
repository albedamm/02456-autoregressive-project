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
from protgen.models.transformer import ProteinTransformer

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 6.1 cell 2
print(f"Device: {DEVICE}")


###########################
### 0. HYPERPARAMETERS ####
###########################
parser = argparse.ArgumentParser()
parser.add_argument("--learning-rate", type=float, default=1e-3)        # Default is set from first round of tuning
parser.add_argument("--dropout", type=float, default=0.1)               # Default is set from first round of tuning
parser.add_argument("--num-epochs", type=int, default=45)               # Default is set from first round of tuning
parser.add_argument("--d-model", type=int, default=128)
parser.add_argument("--num-heads", type=int, default=4)
parser.add_argument("--num-layers", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)                      # Random seed, so runs can be reproduced and repeated
parser.add_argument("--save-dir", type=str, default="src/train/trained_transformer_models")
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

# cross entropy + skipping padding
criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)


###################
### 3. SAMPLING ###
###################

# no gradients, we only generate
@torch.no_grad()  # same as `with torch.no_grad():` in exercise 6.1


def sample(model, batch_size=10, temperature=1.0, max_len=max_len):
    model.eval() # turn of dropout
    token_ids = torch.full((batch_size, 1), BOS, dtype=torch.long, device=DEVICE) # every sequence starts as <BOS>, (B, 1)
    
    # add one token per step
    for t in range(max_len - 1):
        logits = model(token_ids) # scores for every position (B, t+1, vocab)
        logits_t = logits[:, -1:] / temperature # define temperature
        forbidden = torch.tensor([PAD, BOS], device=DEVICE).view(1, 1, -1).expand(batch_size, 1, -1) # ids that should not be generated (BOS and PAD)
        logits_t.scatter_(index=forbidden, dim=2, value=-math.inf) # set the forbidden scores to -inf to make their probs 0 after activation
        p_wt = torch.distributions.Categorical(logits=logits_t) # Categorical passes softmax, which makes scores to probabilities
        tokens_t = p_wt.sample() # grabs a random token from the sample (but higher probability ones will ofc have higher prob of being picked)
        token_ids = torch.cat([token_ids, tokens_t], dim=1) # append the token to the sequence
        if (token_ids == EOS).any(dim=1).all(): # stop when reaching end of sequence
            break

    # cut each sequence at its first <EOS> and turn ids into amino-acid letters
    sequences = []
    # iterating over one sequence at a time
    for seq in token_ids.cpu():
        eos_positions = (seq == EOS).nonzero() #getting the positions of EOS
        if len(eos_positions) > 0: # keep everything before the EOS
            seq = seq[: eos_positions[0, 0]]
        sequences.append(train_data.decode(seq)) # and decode from ids to letters
    return sequences


###################
### 4. TRAINING ###
###################

if __name__ == "__main__": # only runs if the file is executed directly, not if imported
    
    # build the architecture of the model (with random weights)
    model = ProteinTransformer(
        vocab_size=vocab_size, #23 tokens (20 amino acids + BOS, EOS and PAD)
        d_model=d_model, # size of each position's vector
        n_heads=num_heads, # attention heads per block (d_model / n_heads dimensions each)
        d_ff=d_ff, # size of hidden feed-forward in each block
        num_layers=num_layers, # number of decoder blocks
        dropout=dropout, #fraction of values that are dropped out - set to 0 during training
        max_seq_len=max_len, #longest sequence
    )

    # move the weights to DEVICE
    model = model.to(DEVICE)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Define optimizer (we use Adam + weight decay)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate) # From exercise 6.1

    # directory and names to save the model in 
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"d{d_model}_h{num_heads}_l{num_layers}_lr{learning_rate}_drop{dropout}_e{num_epochs}_s{seed}"
    checkpoint_file = save_dir / f"{model_name}.pth"

    # initialize
    train_losses, val_losses = [], []
    best_val_loss = float("inf") # to track best validation loss so far

    ### TRAINING LOOP ###

    # Epoch loop with a train pass and a validation pass from exercise Week 3
    # The inside of the training from execise 6.1
    
    for epoch in range(num_epochs): # one epoch = one pass over all training data
        
        ## Train

        model.train() # dropout on
        epoch_loss = 0.0 # initialize for sum of batch losses in this epoch

        # progress bar to track training epochs
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{num_epochs}")

        # and iterate through each epoch
        for step, (x, x_shifted, y, length) in enumerate(pbar):
            
            # define input tokens, x and next token, x_shifted
            x, x_shifted = x.to(DEVICE), x_shifted.to(DEVICE)

            # Forward through the model
            optimiser.zero_grad() # clear gradients from previous step
            logits = model(x) # and get scores for next tokens (batch, sequence, vocab)

            # define the loss for the next token
            loss = criterion(logits.transpose(1, 2), x_shifted) # CrossEntropyLoss expects logits as (batch, vocab, sequence) so we transpose

            # Backward and optimize
            loss.backward() # compute gradients of the loss with respect to every weight
            optimiser.step() # nudge the weights to improve loss

            epoch_loss += loss.item() # turns the tensor into normal python number and adds it to loss
            # for every 20th step - show the loss in progress bar
            if step % 20 == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}")

        # average training loss and append this epoch
        train_losses.append(epoch_loss / len(train_loader))


        ## Validate
        # (no gradients)
        
        model.eval() # dropout off
        val_loss = 0.0

        # using the model on validation data 
        with torch.no_grad():
            for x, x_shifted, y, length in val_loader: # val data
                x, x_shifted = x.to(DEVICE), x_shifted.to(DEVICE) # move to device 
                val_loss += criterion(model(x).transpose(1, 2), x_shifted).item() # calculate loss as in train
        val_losses.append(val_loss / len(val_loader)) # average and append loss for this epoch

        print(f"Epoch {epoch+1}: train loss {train_losses[-1]:.4f}, "
              f"val loss {val_losses[-1]:.4f}, val perplexity {math.exp(val_losses[-1]):.2f}") # perplexity = e^loss (because loss is log transformed and hard to interpret)

        # Save checkpoint (from exercise 6.1, saves every 50 steps; we keep the best val model)
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save(model.state_dict(), checkpoint_file) # overwrite with new best weights

        # Print what the model generates right now
        print("  sample:", sample(model, batch_size=1)[0][:60], "...")

    # Loss curve, as in src/train/train_mlp.py
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    
    plt.xlabel("Epoch")
    plt.ylabel("Negative log-likelihood per token")
    plt.title(
        "Transformer training and validation loss\n"
        f"learning rate={learning_rate}, dropout={dropout}, epochs={num_epochs}\n"
        f"d_model={d_model}, heads={num_heads}, layers={num_layers}, seed={seed}"
    )
    
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f"{model_name}_loss_curve.png")

    pd.DataFrame({
        "epoch": range(1, num_epochs + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
    }).to_csv(save_dir / f"{model_name}_losses.csv", index=False)