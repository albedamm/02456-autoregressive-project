from torch.utils.data import DataLoader
from protgen.data import ProteinDataset
import torch.nn.functional as F
import protgen.models.mlp as mlp
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from pathlib import Path

#####################
### 0. LOAD DATA ####
#####################
train_data = ProteinDataset("data/train.csv")
val_data = ProteinDataset("data/val.csv")
print(train_data)
# ProteinDataset(num_samples=16020, vocab_size=23, num_classes=3, max_length=129)

# 
x, x_shifted, y, length = train_data[0]

# Handling the data with a DataLoader to create batches and shuffle the data (k=32 - each batch will have 32 samples)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Function to translate the sequences into context windows and targets for the MLP model
def create_context_windows(x, x_shifted, length, k=3):

    # add padding two tokens to the left of the beginning of the sequence
    # Allows the model to learn the first amino acids in the sequence
    x_pad = F.pad(x, (k - 1, 0), value=0)

    # empty context and target list of tensors
    contexts, targets = [], []

    # loop through the length of the embedded amino acids
    for i in range(length + 1):
        context = x_pad[i:i+k]
        # target index is i, because we added padding
        target = x_shifted[i]

        #print("context (input)", context)
        #print("shifted target:", target)

        contexts.append(context)
        targets.append(target)

    return torch.stack(contexts), torch.stack(targets)

# Handling of the batch of data to create context windows and targets for the MLP model
def prepare_batch(batch, k=3):
    x_batch, x_shifted_batch, y_batch, length_batch = batch
    
    # Initialize empty lists to store all contexts and targets 
    all_contexts = []
    all_targets = []
    
    # Loop through each sequence in the batch and create context windows and targets
    for x, x_shifted, length in zip(x_batch, x_shifted_batch, length_batch):
        contexts, targets = create_context_windows(x, x_shifted, length, k)
        
        all_contexts.append(contexts)
        all_targets.append(targets)
    
    # Concatenate all contexts and targets into single tensors
    all_contexts = torch.cat(all_contexts, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return all_contexts, all_targets


# Define the parameters for the MLP model
k = 3
# Selected embedding dimension for the amino acids
embedding_dim = 16
# Number of hidden units in the MLP (possible hyperparameter to tune)
num_hidden = 64
num_output = 23
learning_rate = 0.01

# Initialize the MLP model with the specified parameters
model = mlp.ProteinMLP( num_features=k * embedding_dim, num_hidden=num_hidden, num_output=num_output, embedding_dim=embedding_dim)


# Define the optimizer and loss function for training
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()


# Define the training loop parameters
num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()
    # Resetting the cumulative loss for the epoch
    epoch_loss = 0.0
    
    # Loop through each batch in the training data
    for batch in train_loader:
        # Prepare the batch by creating context windows and targets
        contexts, targets = prepare_batch(batch, k=k)
        
        # Reset the gradients for the optimizer
        optimizer.zero_grad()
        
        # Forward pass through the model to get predictions
        output = model(contexts)
        
        # Compute the loss between the model's predictions and the true targets
        loss = criterion(output, targets)
        
        # Backward pass to compute gradients
        loss.backward()
        # Update the model parameters
        optimizer.step()
        
        # Add the loss for this batch to the cumulative epoch loss
        epoch_loss += loss.item() 
    
    # Append the average loss for this epoch to the losses list
    train_losses.append(epoch_loss / len(train_loader))
    
    # Set the model to evaluation mode 
    model.eval()  
    
    # Set a variable to accumulate the validation loss for this epoch
    val_loss = 0.0    
    
    # Disable gradient computation
    with torch.no_grad():
        
        # Iterate through each batch in the validation data
        for batch in val_loader:
            # Prepare the batch by creating context windows and targets
            contexts, targets = prepare_batch(batch, k=3)
            
            # Forward pass through the model to get predictions
            output = model(contexts)
            
            # Compute the loss for this batch and add it to the cumulative validation loss
            loss = criterion(output, targets)
            val_loss += loss.item()
            
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}: Train loss = {train_losses[-1]:.4f}, Val loss = {val_losses[-1]:.4f}")

# Directory for saved models
save_dir = Path("src/train/trained_mlp_models")
save_dir.mkdir(parents=True, exist_ok=True)

# Defining the model name for saving the trained model
model_name = f"emb{embedding_dim}_h{num_hidden}_lr{learning_rate}_e{num_epochs}"
# Save the trained model to a file
torch.save(model.state_dict(), save_dir / f"{model_name}.pth")
    
# Plotting the training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("MLP Training and Validation Loss")

plt.legend()
plt.grid(True)
plt.savefig(save_dir / f"{model_name}_loss_curve.png")

