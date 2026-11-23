import torch
import protgen.models.transformer as transformer
from protgen.data import ProteinDataset

# Define the hyperparameters for the model
k = 3
embedding_dim = 16
num_hidden = 64
num_output = 23

# Build the MLP model architecture with the specified hyperparameters
model = mlp.ProteinMLP(
    num_features=k * embedding_dim,
    num_hidden=num_hidden,
    num_output=num_output,
    embedding_dim=embedding_dim)

# Define the path to the trained model file
model_path = "/work3/s203530/02456-autoregressive-project/trained_models/trained_transformer_models/stage1/d128_h4_l3_lr0.001_drop0.1_e60_s0.pth"

# Load the trained model weights into the MLP model
model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

# Set the model to evaluation mode
model.eval()

# Load the dataset for decoding generated sequences
dataset = ProteinDataset("data/train.csv")


# Function to generate a protein sequence using the trained MLP model
def generate_sequence(model, max_length=1279-2):
    # Initialize the context with <PAD>, <PAD>, <BOS>
    context = torch.tensor([[0, 0, 1]], dtype=torch.long)
    generated_tokens = []

    # Generate sequences without computing gradients
    with torch.no_grad():
        for i in range(max_length):

            # Predict next token
            logits = model(context)
            # print(logits.shape)  

            # Prevent PAD and BOS from being generated
            logits[:, :2] = -float("inf")

            # Convert logits to probabilities
            probabilities = torch.softmax(logits, dim=-1)
            # print(probabilities)

            # Sample next token
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_id = next_token.item()

            # Stop at EOS
            if token_id == 2:
                break

            # Save the predicted token
            generated_tokens.append(token_id)

            # Update the context window
            context = torch.cat([context[:, 1:], next_token], dim=1)

    return generated_tokens


# Generate and decode a sequence
generated_tokens = generate_sequence(model)
protein_sequence = dataset.decode(generated_tokens)

print("Generated token IDs:", generated_tokens)
print("Generated protein:", protein_sequence)
print("Length of generated protein:", len(protein_sequence))
