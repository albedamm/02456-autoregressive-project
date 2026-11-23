import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from protgen.models.transformer import ProteinTransformer

    
# MAYBE CHANGE? makes sure that the model cant look ahead when predicting the next amino acid, but for the classifier, we should see the whole sequence right.
def causal_mask(T, device):
    " Create a causal mask for self-attention to prevent attending to future positions "
    mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0) # (1, T, T)
    
    return mask
    

class ProteinClassifier(nn.Module):
    """ Causal (decoder-only) Transformer model for protein sequences """

    # use the ProteinTransformer that you imported above, and modify it to be a classifier instead of a language model. You can do this by changing the output layer to output the number of classes you have, and by changing the forward method to return the logits for the classes instead of the logits for the next amino acid.


    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=256, num_layers=3, dropout=0.1, max_seq_len=129):
        super(ProteinClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)


def initialize_classifier(
      map_location="cpu",
      model_path = str,
      finetune_entire_model=False,
      number_of_output_classes = int, 
      weights_only=True # restricts loading to plain tensors
      ):
    
    print(
    f"Loading model '{model_path}', with "
    f"finetune_entire_model={finetune_entire_model}, changing the "
    f"last layer to output {number_of_output_classes} logits."
    )

    model = ProteinTransformer(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    dropout=torch.dropout,
    max_seq_len=max_seq_len,
    )

    # checkpoint_file = Path("src/train/trained_transformer_models/d128_h4_l3_lr0.001_drop0.1_e15.pth")
    checkpoint_file = Path("/Users/kristinetoftjohansen/Desktop/DeepLearning/02456-autoregressive-project/src/train/trained_mlp_models/mlp_emb16_h64_lr0.01_e250.pth")
    model.load_state_dict(torch.load(checkpoint_file))
    num_model_parameters = 0

    # 
    for p in model.parameters():
        if p.requires_grad:
            num_model_parameters += p.numel()

    # if finetine_entire_model is set to False, the parameters will be frozen
    if not finetune_entire_model:
        for name, param in model.named_parameters():
            param.requires_grad = False

    model.linear1 = nn.Linear(in_features=784, out_features=number_of_output_classes)
            
    # Make the last layer trainable.
    layer.weight.requires_grad_()
    layer.bias.requires_grad_()

    num_trainable_model_parameters = 0
    for p in model.parameters():
        if p.requires_grad:
            num_trainable_model_parameters += p.numel()

    return model, {
        "num_model_parameters": num_model_parameters,
        "num_trainable_model_parameters": num_trainable_model_parameters,
    }

model, data = initialize_model('resnet34d', num_classes=len(np.unique(labels)), finetune_entire_model=False)

print(model)
print("Number of model parameters:", data["num_model_parameters"])
print("Number of trainable parameters:", data["num_trainable_model_parameters"])