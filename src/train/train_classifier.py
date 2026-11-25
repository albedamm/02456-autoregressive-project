import math
import argparse
import random
import numpy as np

from pathlib import Path

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from protgen.data import IGNORE_INDEX, ProteinDataset
from protgen.models.classifier import ProteinClassifier, initialize_classifier

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
# vocab_size = 23 # 20 amino acids and <PAD>, <BOS> and <EOS>
d_model = 128 # For every position there's a 128-dimensional vector to carry information, more dimensions means more capacity to represent each position. 
num_heads = 4 # splits d_model dimensions into chunks (d_k = d_model / n_heads)
d_ff = 512
num_layers = 3 # depth
max_len = 512 
max_seq_len = 129 # the maxlength of the dataset sequences (here 127 amino acids (or <PAD>) + <BOS> and <EOS>)
dropout = 0.1
number_output_classes = 3 # number of protein classes

# Training
batch_size = 64       # From exercise 6.1
learning_rate = args.learning_rate
num_epochs = args.num_epochs
# max_len = 129



#####################
### 1. LOAD DATA ####
#####################
train_data = ProteinDataset("data/train.csv")
val_data = ProteinDataset("data/val.csv")

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

vocab_size = train_data.vocab_size # 23 = 3 special tokens + 20 amino acids
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

# turn off gradient tracking
@torch.no_grad()  # same as `with torch.no_grad():` in exercise 6.1
def sample(model, batch_size=10, temperature=1.0, max_len=max_len):
    """
    Produces protein sequences one amino acid at a time
    """

    # turn off dropout
    model.eval()

    # creates Tensors filled with BOS
    token_ids = torch.full(size=(batch_size, 1), fill_value=BOS, dtype=torch.long, device=DEVICE)

    # add one amino acid at a time until max-length is reached:
    for t in range(max_len - 1):
        logits = model(token_ids) # runs everything through the transformer to get scores for each position 
        logits_t = logits[:, -1:] / temperature # picks the last one and normalizes according to temp
        forbidden = torch.tensor([PAD, BOS], device=DEVICE).view(1, 1, -1).expand(batch_size, 1, -1)
        logits_t.scatter_(index=forbidden, dim=2, value=-math.inf)
        p_wt = torch.distributions.Categorical(logits=logits_t)
        tokens_t = p_wt.sample() # (B, 1)
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

    # define the model from the pretrained transformer
    model, lm_head, class_head, data = initialize_classifier(map_location = "cpu", 
                                                            #  number_classes = 3,
                                                            finetune_entire_model = False)

    # and define the classifier
    classifier = ProteinClassifier(model, lm_head, class_head)

    # 
    classifier = classifier.to(DEVICE)

    # print model parameters 
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # only hand the trainable parameters to the optimizer
    optimiser = torch.optim.AdamW(filter(lambda p: p.requires_grad, classifier.parameters()), lr = 1e-3)    # From exercise 6.1

    # make output directory and name for the model
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"d{d_model}_h{num_heads}_l{num_layers}_lr{learning_rate}_drop{dropout}_e{num_epochs}_s{seed}"
    checkpoint_file = save_dir / f"{model_name}.pth"

    # initialize lists for training and validation scores
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    # Epoch loop with a train pass and a validation pass from exercise Week 3
    # The inside of the training from execise 6.1
    for epoch in range(num_epochs):
        
        ## Train
        classifier.train()
        epoch_loss = 0.0

        # define progress bar and since it has the train_loader - the x, x_shifted, y and length will be inside.
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{num_epochs}")
        
        # iterating over the training data (with progress bar)
        for step, (x, x_shifted, y, length) in enumerate(pbar): 

            # define training set (y since classification now)
            x, y = x.to(DEVICE), y.to(DEVICE)
            # define one prediction per position (for target) - so returns list with 129 predicted classes for that protein
            target = y.view(-1, 1).expand_as(x).clone()
            # Ignore the <PAD>, <BOS> and <EOS> tokens in the loss calculation, since they are not part of the protein sequence
            target[x == PAD] = IGNORE_INDEX
            target[x == BOS] = IGNORE_INDEX
            target[x == EOS] = IGNORE_INDEX

            ##### NÅET HERTIL ####

            # Forward through the model
            optimiser.zero_grad()
            logits = classifier(x) # the prediction?

            # CrossEntropyLoss expects logits as (batch, vocab, sequence).
            # Use target instead of y, since y is just a single class per protein, while target is a class per position (with IGNORE_INDEX for <PAD>, <BOS> and <EOS>)
            loss = criterion(logits.transpose(1, 2), target)

            # Backward and optimize
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            if step % 20 == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}")
        train_losses.append(epoch_loss / len(train_loader))

        ## Validate (no gradients, dropout off)
        classifier.eval()
        val_loss = 0.0
        with torch.no_grad():
            # iterating over the validation data:
            for x, x_shifted, y, length in val_loader:

                # define validation datasets (using y since classification)
                x, y = x.to(DEVICE), y.to(DEVICE)
                # Same as above, define one prediction per position (for target) - so returns list with 129 predicted classes for that protein
                target = y.view(-1, 1).expand_as(x).clone()
                # Ignore the <PAD>, <BOS> and <EOS> tokens in the loss calculation, since they are not part of the protein sequence
                target[x == PAD] = IGNORE_INDEX
                target[x == BOS] = IGNORE_INDEX
                target[x == EOS] = IGNORE_INDEX
                val_loss += criterion(classifier(x).transpose(1, 2), target).item()

        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}: train loss {train_losses[-1]:.4f}, "
              f"val loss {val_losses[-1]:.4f}, val perplexity {math.exp(val_losses[-1]):.2f}")

        # Save checkpoint (from exercise 6.1, saves every 50 steps; we keep the best val model)
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save(classifier.state_dict(), checkpoint_file)

        # Print what the model generates right now
        print("  sample:", sample(classifier, batch_size=1)[0][:60], "...")

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