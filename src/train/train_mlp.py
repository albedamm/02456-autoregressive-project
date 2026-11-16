from torch.utils.data import DataLoader
from protgen.data import ProteinDataset
import torch.nn.functional as F
import protgen.models.mlp as mlp
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
import pandas as pd


## Load the training and validation datasets
train_data = ProteinDataset("data/train.csv")
val_data = ProteinDataset("data/val.csv")
# print(train_data)

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

 # Defining the fixed paramters for the MLP model
k = 3
embedding_dim = 16
num_output = 23
num_epochs = 250

# Hyperparameter tuning for the MLP model
learning_rates = [0.005]
hidden_neurons = [32, 64, 128]

# Generate all hyperparameter combinations
configurations = list(product(learning_rates, hidden_neurons))

# Directory to save the model and results in
save_dir = Path("src/train/trained_mlp_models")
save_dir.mkdir(parents=True, exist_ok=True)

# Store results from each model
results = []

for learning_rate, num_hidden in configurations:
    print(f"\nTraining model: lr={learning_rate}, hidden={num_hidden}")

    # Initialize the MLP model with the specified parameters for each configuration
    model = mlp.ProteinMLP(
        num_features=k * embedding_dim, 
        num_hidden=num_hidden,
        num_output=num_output, 
        embedding_dim=embedding_dim)


    # Define the optimizer and loss function for training
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()


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
                contexts, targets = prepare_batch(batch, k=k)
                
                # Forward pass through the model to get predictions
                output = model(contexts)
                
                # Compute the loss for this batch and add it to the cumulative validation loss
                loss = criterion(output, targets)
                val_loss += loss.item()
                
        val_losses.append(val_loss / len(val_loader))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Train loss = {train_losses[-1]:.4f}, Val loss = {val_losses[-1]:.4f}")

    # Defining the model name for saving the trained model
    model_name = f"mlp_emb{embedding_dim}_h{num_hidden}_lr{learning_rate}_e{num_epochs}"
    # Save the trained model to a file
    torch.save(model.state_dict(), save_dir / f"{model_name}.pth")
    
    # Plotting the training and validation loss curves
    plt.figure(figsize=(10, 5))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses, label="Training loss")
    plt.plot(epochs, val_losses, label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"MLP: lr={learning_rate}, hidden={num_hidden}")

    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f"{model_name}_loss_curve.png")
    plt.close()
    
    # Store the results for this model configuration
    results.append({
        "learning_rate": learning_rate,
        "hidden_neurons": num_hidden,
        "epochs": num_epochs,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": min(val_losses),
        "best_epoch": val_losses.index(min(val_losses)) + 1})


results_df = pd.DataFrame(results)

# Sort models by final validation loss
results_df = results_df.sort_values("final_val_loss")

# Save results to CSV
results_df.to_csv(save_dir / "hyperparameter_results.csv", index=False)

print("\nHyperparameter tuning results:")
print(results_df.to_string(index=False))

print("\nBest configuration based on final validation loss:")
print(results_df.iloc[0])