import math
import argparse
import numpy as np
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from protgen.data import IGNORE_INDEX, ProteinDataset
from protgen.models.transformer import ProteinTransformer


### INITIALIZE STUFF ###
vocab_size = 23
d_model = 128
n_heads = 4
d_ff = 512
num_layers = 3
dropout = 0.1
max_seq_len = 129

class ProteinClassifier(nn.Module):
    """
    The classifier with frozen transformer + a two heads: generation (use for guided sequence generation) and classification.
    """

    def __init__(self, transf_frozen, lm_head, class_head):
        super().__init__() #initialize the nn.Module class (parent)

        self.transf_frozen = transf_frozen #ProteinTransformer with output layer = nn.Identity
        self.lm_head = lm_head # the predicted next amino acid (dim 23)
        self.class_head = class_head # the predicted class (dim 3)

    def forward(self, x):
        h = self.transf_frozen(x) # running the frozen transformer
        return self.class_head(h) # and then run class_head on that to get protein class scores per position
    
    def lm_logits(self, x):
        return self.lm_head(self.transf_frozen(x)) # next token scores - for guidance

    # make sure the frozen parts' dropout is turned off
    def train(self, mode=True):
        super().train(mode) # switch to the nn.Module train mode
        self.transf_frozen.eval() #but keep the other parts frozen (eval mode)
        self.lm_head.eval() #but keep the other parts frozen (eval mode)
        return self



def initialize_classifier(
      map_location="cpu",
      model_path = str,
      finetune_entire_model=False,
    #   number_classes = 3, 
    #   weights_only=True # restricts loading to plain tensors
      number_output_classes = 3, # number of protein classes
      class_head_features = 64 # also try 32 and 128
      ):

    model = ProteinTransformer(
    vocab_size = vocab_size,
    d_model = d_model,
    n_heads = n_heads,
    d_ff = d_ff,
    num_layers = num_layers,
    dropout = dropout,
    max_seq_len = max_seq_len,
    )

    checkpoint_file = Path("/Users/kristinetoftjohansen/Desktop/DeepLearning/02456-autoregressive-project/trained_models/trained_transformer_models/stage1/d128_h4_l3_lr0.001_drop0.1_e60_s0.pth")
    model.load_state_dict(torch.load(checkpoint_file, map_location=map_location))
    
    num_model_parameters = 0

    # counts number of parameters
    for p in model.parameters():
        if p.requires_grad:
            num_model_parameters += p.numel()

    # if finetine_entire_model is set to False, the parameters will be frozen
    if not finetune_entire_model:
        for name, param in model.named_parameters():
            param.requires_grad = False

    # save the original output layer, which is a linear layer that outputs the logits for the next amino acid (128 to 23 aas)
    lm_head = model.output_layer

    # make model(x) return the hidden states
    model.output_layer = nn.Identity()

    # adding the classification head
    class_head = nn.Sequential(
        nn.Linear(in_features = d_model, out_features=class_head_features), 
        nn.ReLU(), #activation function
        nn.Dropout(0.1), #applying dropout again for more robust learning
        nn.Linear(in_features = class_head_features, out_features = number_output_classes) #128 input features to this layer (d_model)
    ) # 128 numbers → 64 numbers → 3 family scores

    return model, lm_head, class_head, {
        "num_model_parameters": num_model_parameters,
    }