import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from protgen.data import ProteinDataset

#####################
### 0. LOAD DATA ####
#####################
train_data = ProteinDataset("data/train.csv")
val_data = ProteinDataset("data/val.csv")
# ProteinDataset(num_samples=16020, vocab_size=23, num_classes=3, max_length=129)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

##### 

def train_mlp(x, x_shifted, length, k=3):

    # add padding two tokens to the left of
    x_pad = F.pad(x, (k - 1, 0), value=0)

    # empty context and target list of tensors
    contexts, targets = [], []

    # loop through the length of the embedded amino acids
    for i in range(length + 1):
        context = x_pad[i:i+k]
        # target index is i, because we added padding
        target = x_shifted[i]

        print("context (input)", context)
        print("shifted target:", target)

        contexts.append(context)
        targets.append(target)

    return torch.stack(contexts), torch.stack(targets)


x, x_shifted, y, length = train_data[0]
contexts, targets = train_mlp(x, x_shifted, length)