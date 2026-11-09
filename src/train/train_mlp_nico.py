import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from protgen.data import ProteinDataset
from utilities.utilities import Var
from sklearn.metrics import accuracy_score
import random
from typing import Sequence
import numpy as np
import protgen.models.mlp as mlp
import torch.optim as optim
import torch.nn.init as init
import torch
from torch import nn
import torch.nn.functional as F

train_data = ProteinDataset("data/train_test.csv")
#val_data = ProteinDataset("data/val.csv")

# Take 300 training samples and 100 validation samples for faster training
#train_data.df = train_data.df.sample(n=300, random_state=42)
#val_data.df = val_data.df.sample(n=100, random_state=42)

#print(val_data)
# ProteinDataset(num_samples=300, vocab_size=23, num_classes=3, max_length=129)

x, x_shifted, y, length = train_data[0]
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


num_features = 2
num_hidden = 4
num_output = 1

model = mlp.ProteinMLP(num_features, num_hidden, num_output)

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# setting hyperparameters and gettings epoch sizes
batch_size = 100
num_epochs = 200
num_samples_train = x.shape[0]  # Fix for correct training data 
num_batches_train = num_samples_train // batch_size
num_samples_valid = x_shifted.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

for epoch in range(num_epochs):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss = 0
    model.train()
    for i in range(num_batches_train):  # 100 sequences per batch
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        for index in slce:
            x_train = x[index].unsqueeze(0)  # Hopefully one sequence
            print("x_train", x_train)
            target_train = x_shifted[index].unsqueeze(0)  # Hopefully one sequence
            print("target_train", target_train)
            #targets_train = x_shifted[index].unsqueeze(0)  # Add batch dimension
            k = 3 # number of amino acids to take into account
            # loop through the length of the embedded amino acids
            for i in range(len(x_train) - k + 1):
                context = x_train[i:i+k]
                target = target_train[i + k - 1]
                print("context (input)", context)
                print("shifted target:", target)

                output = model(context)
        
                # compute gradients given loss
                #target_batch = targets_train[slce]
                batch_loss = criterion(output, target)
                batch_loss.backward()
                optimizer.step()
        
                cur_loss += batch_loss   
            losses.append(cur_loss / batch_size)

    model.eval()
    ### Evaluate training
    train_preds, train_targs = [], []
    for i in range(num_batches_train):
        slce = get_slice(i, batch_size)
        output = model(x_train[slce])
        
        preds = torch.max(output, 1)[1]
        
        train_targs += list(targets_train[slce].numpy())
        train_preds += list(preds.data.numpy())
    
    ### Evaluate validation
    val_preds, val_targs = [], []
    for i in range(num_batches_valid):
        slce = get_slice(i, batch_size)
        
        output = model(x_valid[slce])
        preds = torch.max(output, 1)[1]
        val_targs += list(targets_valid[slce].numpy())
        val_preds += list(preds.data.numpy())
        

    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)
    
    if epoch % 10 == 0:
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch+1, losses[-1], train_acc_cur, valid_acc_cur))

epoch = np.arange(len(train_acc))
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Accucary','Validation Accuracy'])
plt.xlabel('Updates'), plt.ylabel('Acc')


'''
def train(self, x, x_shifted, epochs=1000, lr=0.01):
    losses = []
    for epoch in range(epochs):
        output = ProteinMLP.forward(X)
        #Compute loss using (Mean Squared Error)
        loss = torch.mean((output - y) ** 2)
        losses.append(loss.item())
        #update weights
        self.backward(x, x_shifted, output, lr)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    return losses

num_features = 2
num_hidden = 4
num_output = 1
model = ProteinMLP(num_features, num_hidden, num_output)

#Train  model and store the losses
losses = model.train(x, x_shifted, epochs=1000, lr=0.1)
'''
