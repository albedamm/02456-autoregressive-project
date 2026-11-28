import math
import argparse
import random
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from protgen.data import IGNORE_INDEX, ProteinDataset
from protgen.models.classifier import ProteinClassifier, initialize_classifier, position_targets

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 6.1 cell 2
print(f"Device: {DEVICE}")


###########################
### 0. HYPERPARAMETERS ####
###########################
parser = argparse.ArgumentParser()
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--num-epochs", type=int, default=45)
parser.add_argument("--class-head-features", type=int, default=64)
# Default hyperparameters for the tuned transformer model
parser.add_argument("--d-model", type=int, default=384)
parser.add_argument("--num-heads", type=int, default=12)
parser.add_argument("--num-layers", type=int, default=4)

parser.add_argument("--seed", type=int, default=0)                      # Random seed, so runs can be reproduced and repeated
parser.add_argument("--save-dir", type=str, default="trained_models/trained_classifier_models")
args = parser.parse_args()

# Fix the random seed (weight init, dropout and data shuffling)
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)    # also seeds all CUDA devices

# Model size, change if needed
# vocab_size = 23 # 20 amino acids and <PAD>, <BOS> and <EOS>
d_model = args.d_model # For every position there's a d_model-dimensional vector to carry information, more dimensions means more capacity to represent each position. 
num_heads = args.num_heads # splits d_model dimensions into chunks (d_k = d_model / n_heads)
d_ff = 4 * d_model    # Feed-forward hidden size, 4x d_model as in Vaswani et al.
num_layers = args.num_layers # depth
max_len = 512 
max_seq_len = 129 # the maxlength of the dataset sequences (here 127 amino acids (or <PAD>) + <BOS> and <EOS>)
dropout = args.dropout
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
### 3. TRAINING ###
###################
if __name__ == "__main__":
    # define the model from the pretrained transformer
    body, lm_head, class_head, data = initialize_classifier(map_location = DEVICE,
                                                            #  number_classes = 3,
                                                            finetune_entire_model = False,
                                                            class_head_features=args.class_head_features,
                                                            class_head_dropout=dropout)

    # and define the classifier
    classifier = ProteinClassifier(body, lm_head, class_head)

    classifier = classifier.to(DEVICE)

    # print model parameters 
    print(f"Parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    # only hand the trainable parameters to the optimizer
    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=learning_rate,
    )

    # make output directory and name for the model
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = (
        f"d{d_model}_h{num_heads}_l{num_layers}"
        f"_cf{args.class_head_features}_lr{learning_rate}"
        f"_drop{dropout}_e{num_epochs}_s{seed}"
    )
    checkpoint_file = save_dir / f"{model_name}.pth"

    # initialize lists for training and validation scores
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float("inf")

    # Epoch loop with a train pass and a validation pass from exercise Week 3
    # The inside of the training from execise 6.1
    for epoch in range(num_epochs):
        
        ## Train
        classifier.train()
        epoch_loss = 0.0
        train_correct = 0
        train_token_count = 0

        # define progress bar and since it has the train_loader - the x, x_shifted, y and length will be inside.
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{num_epochs}")
        
        # iterating over the training data (with progress bar)
        for step, (x, x_shifted, y, length) in enumerate(pbar): 

            # define training set (y since classification now)
            x, y = x.to(DEVICE), y.to(DEVICE)
            # one target class per position (129 per protein), IGNORE_INDEX on <PAD>, <BOS> and <EOS>,
            # since they are not part of the protein sequence
            target = position_targets(x, y, train_data.vocab)

            # Forward through the model
            optimiser.zero_grad()
            # The predicted logits for each position in the protein sequence (for each protein in the batch)
            logits = classifier(x)
            '''
            x: batch of real, padded sequences
            ↓
            classifier(x)
            ↓
            frozen Transformer builds a representation for each position (defines what the model can see for each position)
            ↓
            class_head turns each representation into 3 family scores (for each position it predicts the probability of being in each of the 3 classes)
            ↓
            logits: (batch size, sequence length, 3) (for each sequence in the batch, for each position in the sequence, 3 scores for the 3 classes)
            '''

            #print(f"logits shape: {logits.shape}, target shape: {target.shape}")
            #print(f"logits: {logits}, target: {target}")

            # CrossEntropyLoss expects logits as (batch, vocab, sequence).
            # Use target instead of y, since y is just a single class per protein, while target is a class per position (with IGNORE_INDEX for <PAD>, <BOS> and <EOS>)
            loss = criterion(logits.transpose(1, 2), target)

            # Backward and optimize
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            # Positions should match the target, but we ignore the <PAD>, <BOS> and <EOS> tokens in the accuracy calculation
            valid_positions = target != IGNORE_INDEX
            # Get the predicted class for each position (the one with the highest score)
            predictions = logits.argmax(dim=2)
            # Count the number of correct predictions for the valid positions
            correct = predictions[valid_positions] == target[valid_positions]
            # Count the number of correct predictions and the total number of valid positions
            train_correct += correct.sum().item()
            # Count the total number of valid positions (not <PAD>, <BOS> or <EOS>)
            train_token_count += correct.numel()
            if step % 20 == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}")
        # Save train loss and accuracy for this epoch
        train_losses.append(epoch_loss / len(train_loader))
        train_accuracies.append(train_correct / train_token_count)

        ## Validate (no gradients, dropout off)
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_token_count = 0
        with torch.no_grad():
            # iterating over the validation data:
            for x, x_shifted, y, length in val_loader:

                # define validation datasets (using y since classification)
                x, y = x.to(DEVICE), y.to(DEVICE)
                # Same targets as in the train loop
                target = position_targets(x, y, val_data.vocab)
                # Same procedure as in the train loop, but now for the validation data
                logits = classifier(x)
                # validation loss and accuracy are calculated in the same way as in the training loop
                val_loss += criterion(logits.transpose(1, 2), target).item()
                valid_positions = target != IGNORE_INDEX
                predictions = logits.argmax(dim=-1)
                val_correct += (
                    (predictions == target) & valid_positions
                ).sum().item()
                val_token_count += valid_positions.sum().item()
        # Save validation loss and accuracy for this epoch
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_correct / val_token_count)

        print(f"Epoch {epoch+1}: train loss {train_losses[-1]:.4f}, "
              f"val loss {val_losses[-1]:.4f}, "
              f"train amino-acid accuracy {train_accuracies[-1]:.4f}, "
              f"val amino-acid accuracy {val_accuracies[-1]:.4f}")

        # Save checkpoint (from exercise 6.1, saves every 50 steps; we keep the best val model)
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save(classifier.state_dict(), checkpoint_file)

        # Print what the model generates right now
        #print("  sample:", sample(classifier, batch_size=1)[0][:60], "...")

    epochs = range(1, num_epochs + 1)
    best_epoch = val_losses.index(min(val_losses)) + 1    # the epoch the saved checkpoint is from
    settings = (
        f"head features={args.class_head_features}, learning rate={learning_rate}, "
        f"dropout={dropout}, epochs={num_epochs}, seed={seed}\n"
        f"frozen transformer: d_model={d_model}, heads={num_heads}, layers={num_layers}"
    )

    # Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Training loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.axvline(best_epoch, color="grey", linestyle="--", label=f"Saved checkpoint (epoch {best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy per amino-acid position (nats)")
    plt.title(f"Classifier training and validation loss\n{settings}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_loss_curve.png")

    # Accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label="Training accuracy")
    plt.plot(epochs, val_accuracies, label="Validation accuracy")
    plt.axvline(best_epoch, color="grey", linestyle="--", label=f"Saved checkpoint (epoch {best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy per amino-acid position")
    plt.title(f"Classifier training and validation accuracy\n{settings}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_accuracy_curve.png")

    pd.DataFrame({
        "epoch": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_accuracy": train_accuracies,
        "val_accuracy": val_accuracies,
    }).to_csv(save_dir / f"{model_name}_losses.csv", index=False)