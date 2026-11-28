import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from protgen.data import IGNORE_INDEX, ProteinDataset
from protgen.models.transformer import ProteinTransformer

# Paths to test data and chosen model
data_path = Path("data/test.csv")
model_path = "trained_models/trained_transformer_models/d384_h12_l4_lr0.001_drop0.2_e60_s0.pth"
results_path = Path("results/eval/test_transformer_results.csv")

# Hyperparameter settings
batch_size = 64
d_model = 384
num_heads = 12
num_layers = 4
d_ff = 4 * d_model
dropout = 0.2
max_len = 129


# Load data
test_data = ProteinDataset(data_path)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

vocab_size = test_data.vocab_size

criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def score(model, loader, device):
    test_loss = 0.0
    for x, x_shifted, y, length in loader:
        x, x_shifted = x.to(device), x_shifted.to(device)
        test_loss += criterion(model(x).transpose(1, 2), x_shifted).item()
    return test_loss / len(loader), len(loader.dataset)


def main():
    checkpoint = Path(model_path)

    model = ProteinTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_len=max_len,
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()

    print(f"checkpoint: {checkpoint}")
    loss, n_seq = score(model, test_loader, device)
    perplexity = math.exp(loss)
    print(f"{'sequences':>9s} {'loss':>18s} {'perplexity':>12s}")
    print(f"{n_seq:9d} {loss:18.4f} {perplexity:12.4f}")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="") as results_file:
        writer = csv.writer(results_file)
        writer.writerow(("checkpoint", "sequences", "loss", "perplexity"))
        writer.writerow((checkpoint, n_seq, loss, perplexity))
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
