import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from protgen.data import ProteinDataset

#####################
### 0. LOAD DATA ####
#####################
train_data = ProteinDataset("data/train.csv")
val_data = ProteinDataset("data/val.csv")
# ProteinDataset(num_samples=16020, vocab_size=23, num_classes=3, max_length=129)

x, x_shifted, y, length = train_data[0]
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


# print("\n X:\n", x, "\n x_shifted:\n", x_shifted, "\n y:\n", y, "\n length:\n", length)
print(x)


##### 

def train_mlp_kristine(k=3, x=x, x_shifted=x_shifted):

    k = k # number of amino acids to take into account
    # max_length=129 # length of protein sequences (should all be the same)

    # loop through the length of the embedded amino acids
    for i in range(len(x) - k + 1):
        context = x[i:i+k]
        target = x_shifted[i + k - 1]

        

        print("context (input)", context)
        print("shifted target:", target)

    return x

    # # # our input (the first three amino acids)
    # input = print(x[i : k])
    # print(input)

    # # our target (the 4th amino acid)
    # target = print(x[k + 1])
    # print(target)

    # # blah blah 
    # # a lot of training

    # # At the end of the loop, define a new x and a new x_shifted
    # x = x_shifted
    # x_shifted = ProteinDataset.shift_tokens(x_shifted)

