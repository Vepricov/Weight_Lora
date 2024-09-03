import json
import random
import torch
import numpy as np
import torch.nn as nn

def proj_0(x, mask):
  return x.mul(mask)

class AdapterLayer(nn.Module):
    """Wraps a linear layer with LoRA-like adapter. Wraps an existing OPT linear layer"""
    def __init__(self, module: nn.Linear):
        super().__init__()
        self.module = module  # pre-trained (frozen) linear layer
        self.adapter = nn.Linear(in_features=module.in_features,
                                 out_features=module.out_features,
                                 bias=False)
        # nn.init.kaiming_uniform_(self.adapter_A, a=5 ** 0.5)
        #self.adapter_B = nn.Parameter(torch.zeros(rank, module.out_features, device=module.weight.device))

    def forward(self, x):
        # Apply self.module and LoRA adapter, return the sum (self.module outputs + adapter outputs)
        frwd_module = self.module(x)
        frwd_adapter = self.adapter(x)
        return frwd_module + frwd_adapter
    
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def load_and_transform_jsonl(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    transformed_data = [{"chat": item["chat"]} for item in data]
    return transformed_data

def set_seed(seed): # ставит сид
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)