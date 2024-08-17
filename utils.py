import torch.optim as optim
from typing import Callable
import torch
import torch.nn as nn
import numpy as np

class StoIHT(optim.Optimizer):
  def __init__(self, params, k, approx, proj, prob, lr=0.01):
    if lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    defaults = dict(lr=lr, approx=approx, proj=proj, k=k, prob=prob)
    super(StoIHT, self).__init__(params, defaults)

  def step(self, closure: Callable = None):
    loss = None
    if closure is not None:
      loss = closure()
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        d_p = p.grad.data
        b_t = p.data - group['lr'] * d_p
        if np.random.random() < group['prob']:
          Gamma_t = group['approx'](b_t, group['k'])
          p.data = group['proj'](b_t, Gamma_t)
        else:
           p.data = b_t
    return loss

def approx_0(x, k):
  if len(x.shape) == 2:
    x_long = x.reshape(-1)
    idxs = torch.sort(x_long, descending=True).indices[:k]
    mask = torch.zeros_like(x_long, dtype=torch.float32)
    for i in idxs:
      mask[i] = 1.
    return mask.reshape(x.shape)

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

