import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protgen.data import ProteinDataset
from protgen.models.mlp import ProteinMLP

# Paths to test data and chosen model
data_path = Path("data/test.csv")
model_path = Path("trained_models/trained_mlp_models/mlp_emb16_h64_lr0.005_e250.pth")

# Hyperparameter settings
batch_size = 32
k = 3
embedding_dim = 16
num_hidden = 64
num_output = 23

# Load data
test_data = ProteinDataset(data_path)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()
# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Same context-window function as during training
def create_context_windows(x, x_shifted, length, k=3):
    x_pad = F.pad(x, (k - 1, 0), value=0)

    contexts, targets = [], []

    for i in range(length + 1):
        context = x_pad[i:i+k]
        target = x_shifted[i]

        contexts.append(context)
        targets.append(target)

    return torch.stack(contexts), torch.stack(targets)


# Same batch preparation as during training
def prepare_batch(batch, k=3):
    x_batch, x_shifted_batch, y_batch, length_batch = batch

    all_contexts = []
    all_targets = []

    for x, x_shifted, length in zip(
        x_batch, x_shifted_batch, length_batch
    ):
        contexts, targets = create_context_windows(
            x, x_shifted, length, k
        )

        all_contexts.append(contexts)
        all_targets.append(targets)

    return torch.cat(all_contexts, dim=0), torch.cat(all_targets, dim=0)

# Function to evaluate the model on the test dataset and compute the average loss and perplexity
# no_grad diables gradient computation
@torch.no_grad()
def score(model, loader, device):
    # Initialize the test loss to zero
    test_loss = 0.0

    for batch in loader:
        # Prepare the batch of data for evaluation
        contexts, targets = prepare_batch(batch, k=k)

        # Move the contexts and targets to the specified device (GPU or CPU)
        contexts = contexts.to(device)
        targets = targets.to(device)

        # Compute the loss for the current batch and accumulate it into the total test loss
        test_loss += criterion(model(contexts), targets).item()

    return test_loss / len(loader), len(loader.dataset)

def main():
    # Load the trained model from the specified checkpoint
    checkpoint = Path(model_path)

    # Initialize the ProteinMLP model with the specified hyperparameters
    model = ProteinMLP(
        num_features=k * embedding_dim,
        num_hidden=num_hidden,
        num_output=num_output,
        embedding_dim=embedding_dim
    )

    # Load the model's state dictionary from the checkpoint
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    # Move the model to the specified device (GPU or CPU) and set it to evaluation mode
    model.to(device).eval()

    print(f"checkpoint: {checkpoint}")
    print(f"{'sequences':>9s} {'loss':>18s} {'perplexity':>10s}")

    # Evaluate the model on the test dataset and print the results
    loss, n_seq = score(model, test_loader, device)
    print(f"{n_seq:9d} {loss:18.4f} {math.exp(loss):10.3f}")


if __name__ == "__main__":
    main()