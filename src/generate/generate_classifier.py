import torch
from protgen.models.classifier import ProteinClassifier, initialize_classifier
from protgen.data import ProteinDataset
from pathlib import Path
import pandas as pd

# Hyperparameter settings
batch_size = 64
d_model = 384
num_heads = 12
num_layers = 4
d_ff = 4 * d_model
dropout = 0.2
max_len = 129
vocab_size = 23

# The number of sequences to generate
num_sequences = 20
generated_sequences = []

# Build the classifier model architecture with the specified hyperparameters

# define the model from the pretrained transformer
model, lm_head, class_head, data = initialize_classifier(map_location = "cpu", 
                                                        #  number_classes = 3,
                                                        finetune_entire_model = False)

# and define the classifier
classifier = ProteinClassifier(model, lm_head, class_head)

# Define the path to the trained model file [CHANGE WHEN TRAINED MODEL]
model_path = "trained_models/trained_transformer_models/stage1/d384_h12_l4_lr0.001_drop0.2_e60_s0.pth"

# Load the trained model weights into the classifier model
classifier.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

# Set the model to evaluation mode
classifier.eval()

# Use the ProteinDataset function to load the decode function
dataset = ProteinDataset("data/train.csv")

# Function to generate a protein sequence using the trained transformer model
def generate_sequence(classifier, max_length=max_len-2):
    BOS, EOS = 1, 2 # special tokens 
    context = torch.tensor([[BOS]], dtype=torch.long)   # start with just <BOS> (1, 1)
    generated_tokens = [] #initialize list for amino acids generated so far

    # no gradients - only generate
    with torch.no_grad():

        # iterate over each amino acid (one added at a time)
        for _ in range(max_length):

            ### PLAN (PSEUDO CODE ETC)

            # new score(a) = p_LM(a) × p_family(X | prefix + a)
            
            # λ: how much you trust the family detector
            # score(a) = log p_LM(a) + λ · log p_family(X | prefix + a)

            # The theory behind it is Bayes' rule:
            # p(a | prefix, family) is proportional to p(a | prefix) · p(family | prefix + a)
            # (λ = 1 is that formula exactly.)

            # FIRST PART which amino acid would the real proteins have next? p(a | prefix)
            logits = classifier(context)[:, -1, :] # scores for the next token (1, vocab)
            # missing this part: logits[:, :2] = -float("inf") # block <PAD> and <BOS> (-inf gives prob 0 after activation)
            torch.log_softmax(logits) #use log cause adding a lot of small numbers

            # add the amino acids to the context (same as transformer but for all 23 options)
            context = torch.cat([context, next_token], dim=1)   # ???

            # SECOND PART given the sequence with the new amino acid, what is the probability of each class?
            class_logits = classifier(candidates)

            # # multiplying the probabilities according to bayes theorem to get probabilities for the next amino acid:
            # probabilities_combined = logits * class_logits

            # # taking into account both the FIRST (realistic amino acid) and SECOND (protein family) part, draw the best next amino acid
            # next_token = torch.multinomial(probabilities_combined, num_samples=1) # draw the next token from the probabilities

            # # if that token is end of sequence, we stop
            # if next_token.item() == EOS:
            #     break

            # # append the token id and add the context for the next step
            # generated_tokens.append(next_token.item())
            # context = torch.cat([context, next_token], dim=1)   # grow, don't slide

    # return lit of ids, without <BOS> and <EOS>
    return generated_tokens

# Generate and decode one sequence at a time using the generate_sequence func:
for i in range(num_sequences):
    generated_tokens = generate_sequence(classifier)
    protein_sequence = dataset.decode(generated_tokens)
    print("Generated token IDs:", generated_tokens)
    print("Generated protein:", protein_sequence)
    print("Length of generated protein:", len(protein_sequence))

    generated_sequences.append(protein_sequence)

# Save sequences to CSV
save_path = Path("results/sequences/classifier_sequences.csv")
df = pd.DataFrame({"sequence": generated_sequences})
df.to_csv(save_path, index=False)
print(f"Saved {num_sequences} sequences to {save_path}")
