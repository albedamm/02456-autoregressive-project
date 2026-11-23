import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from protgen.data import IGNORE_INDEX, ProteinDataset
from protgen.models.transformer import ProteinTransformer

# Paths to test data and chosen model
data_path = Path("/work3/s203530/02456-autoregressive-project/data/test.csv")
model_path = "/work3/s203530/02456-autoregressive-project/trained_models/trained_transformer_models/stage1/d128_h4_l3_lr0.001_drop0.1_e60_s0.pth"

# Hyperparameter settings
batch_size = 64
d_model = 128
num_heads = 4
num_layers = 3
d_ff = 4 * d_model
dropout = 0.1
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
    print(f"{'sequences':>9s} {'loss':>18s} {'perplexity':>10s}")
    loss, n_seq = score(model, test_loader, device)
    print(f"{n_seq:9d} {loss:18.4f} {math.exp(loss):10.3f}")


if __name__ == "__main__":
    main()
