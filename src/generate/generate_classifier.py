import torch
import argparse
from protgen.models.classifier import ProteinClassifier, initialize_classifier
from protgen.data import ProteinDataset
from pathlib import Path
import pandas as pd



###########################
### 0. HYPERPARAMETERS ####
###########################

parser = argparse.ArgumentParser()
parser.add_argument("--num_sequences", type = int, default = 1)
parser.add_argument("--trained_model", type = str, default = "trained_models/trained_classifier_models/d384_h12_l4_cf128_lr0.001_drop0.2_e60_s0.pth")
parser.add_argument("--save_dir", type = str, default = "results/sequences/classifier_sequences")
parser.add_argument("--target_family", type = int, default = 0)
# 0 = AcyP, 1 = HPr, 2 = HU
parser.add_argument("--lam", type = float, default = 0.5)
args = parser.parse_args() # Maybe try for lambda values like: 0, 0.5, 1, 2, 5, 10

# Hyperparameter settings
class_head_features = 128
class_head_dropout = 0.2
batch_size = 64 # not used
d_model = 384 # not used
d_ff = 4 * d_model # not used
num_heads = 12 # num_heads
num_layers = 4 # num_layers

dropout = 0.2 # not used
max_len = 129 # used
vocab_size = 23 # used

max_length = 1 # should be 129


# Initialize list for generated sequences
generated_sequences = []





# Build the classifier model

# define the model from the pretrained transformer
body, lm_head, class_head, data = initialize_classifier(map_location = "cpu", 
                                                        #  number_classes = 3,
                                                        finetune_entire_model = False,
                                                        class_head_features = class_head_features,
                                                        class_head_dropout = class_head_dropout,)
classifier = ProteinClassifier(body, lm_head, class_head) # and build the classifier
# classifier has lm_logits from ProteinClassifier class which gives the aa scores from the frozen transformer layer.

# Define the path to the trained classifier model
# model_path = "trained_models/trained_classifier_models/d128_h4_l3_cf64_lr0.001_drop0.1_e1_s0.pth"

# Load the trained model weights into the classifier model
classifier.load_state_dict(torch.load(f = args.trained_model, map_location="cpu", weights_only=True))

# Set the model to evaluation mode
classifier.eval()

# Use the ProteinDataset function to load the decode function
dataset = ProteinDataset("data/train.csv")


###### SEQUENCE GENERATION #####

# Function to generate a protein sequence using the trained transformer model
def generate_sequence(classifier, max_length=max_len-2):
    
    # define special tokens 
    PAD = dataset.vocab["<PAD>"]
    BOS = dataset.vocab["<BOS>"]
    EOS = dataset.vocab["<EOS>"]

    # define context
    context = torch.tensor([[BOS]], dtype=torch.long)   # start with just <BOS> (1, 1)
    
    #initialize list for amino acids generated so far
    generated_tokens = [] 

    # no gradients - only generate
    with torch.no_grad():

        # iterate over each amino acid (one added at a time)
        for i in range(max_length):

            print(i)
            print("initial context: ", context)

            ### PLAN (PSEUDO CODE ETC)

            # The theory behind it is Bayes' rule:
            # p(a | prefix, family) is proportional to p(a | prefix) · p(family | prefix + a)

            # So we have: 
            # score(a) = log p_LM(a) + λ · log p_family(X | prefix + a)
            # where λ is how much you trust the family detector


            # FIRST PART which amino acid would the real proteins have next? p(a | prefix)
            scores = classifier.lm_logits(context) # get the scores from the frozen generative transformer + lm_head.
            next_scores = scores[:, -1, :] # takes all sequences in batch, the last positionm and all 23 amino acids. 
            next_scores[:, [PAD, BOS]] = -float("inf") # set PAD and BOS to -inf so they become 0 after activation. Used log cause we add a lot of small numbers
            lm_logp = next_scores.log_softmax(dim=-1) # log-probabilities from scores (log so we can add them later).

            # Need to write out sequences with every new possible token as classifier can only predict for full sequence
            repeated = context.repeat(vocab_size,  1) #makes 23 copies of the context
            next_ids = torch.arange(vocab_size).view(-1, 1) # arranges the possible amino acids ids into a single column
            candidates = torch.cat([repeated, next_ids], dim = 1) #adds the amino acids to the contexts (so we have one for each possible amino acid) # dim 1 means it is joined on the side of the context
        
            # SECOND PART given the sequence with the new amino acid, what is the probability of each class?
            class_scores = classifier(candidates)[:, -1, :] # scores for each protein family per candidate
            class_logp = class_scores.log_softmax(dim=-1) # activation function (again with log)
            class_logp = class_logp[:, args.target_family] # choosing the probability obtained for desired protein family
            class_logp[EOS] = classifier(context)[0, -1].log_softmax(-1)[args.target_family] # give EOS the family score of the sequence so far


            # According to Bayes rule, we can combine the class prob and the generation prob in log space
            combined = lm_logp + args.lam * class_logp
            # again do the activation function to turn into probabilities
            combined_logp = combined.softmax(dim=-1)

            # and then taking into account both the FIRST (realistic amino acid) and SECOND (protein family) part, draw the best next amino acid
            next_token = torch.multinomial(combined_logp, num_samples=1) # draw the next token from the probabilities


            # And FINALLY we can add that token to the context, but stop if we reach EOS:
            if next_token.item() == EOS:
                break
            else:
                generated_tokens.append(next_token.item()) # pyyhon list - with the extended sequence, no bos and eos
                context = torch.cat([context, next_token], dim=1) # context as input to the model in the next loop

    # return lit of ids, without <BOS> and <EOS>
    return generated_tokens


# Generate and decode one sequence at a time using the generate_sequence func:
for i in range(args.num_sequences):
    generated_tokens = generate_sequence(classifier)
    protein_sequence = dataset.decode(generated_tokens)
    print("Generated token IDs:", generated_tokens)
    print("Generated protein:", protein_sequence)
    print("Length of generated protein:", len(protein_sequence))

    generated_sequences.append(protein_sequence)

print(generated_sequences)

# Save sequences to CSV
save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / f"family{args.target_family}_lam{args.lam}.csv" # save sequences with name based on protein family and lambda value

df = pd.DataFrame({"sequence": generated_sequences,
                   "target_family": args.target_family,
                   "lam": args.lam})

df.to_csv(save_path, index=False)
print(f"Saved {args.num_sequences} sequences to {save_path}")
