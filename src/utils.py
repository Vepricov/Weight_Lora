import random
import torch
import torch.nn as nn
import numpy as np

def print_trainable_parameters(model, verbose=True):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    for param in model.buffers():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if verbose:
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    return all_param, trainable_params

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_device(device_no: int):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_no}")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(device_no))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    return device

def count_atapters(model, peft_type):
    if peft_type in ["LoRA", "ADALoRA", "DoRA", "rsLoRA"]:
        adapter_name = "lora_A"
    elif peft_type == "LoKR":
        adapter_name = "lokr_w1"
    elif peft_type == "LoHA":
        adapter_name = "hada_w1_a" 
    elif peft_type == "VERA":
        adapter_name = "vera_lambda_b"
    elif peft_type in ["WeightLoRA", "RandLoRA"]:
        adapter_name = "weight_lora_A"
    elif peft_type == "Full":
        adapter_name = None
    else:
        raise ValueError(f"Wrong peft_type: {peft_type}")
    
    num_adapters = None
    if adapter_name is not None:
        num_adapters = 0
        for name, _ in model.named_parameters():
            if adapter_name in name:
                num_adapters += 1
    
    return num_adapters

def apply_rand_weight_lora(model, n, k):
    idxs = torch.randperm(n)[:k] 
    i = 0
    for name, param in model.named_parameters():
        if "weight_lora_w" in name:
            param.requires_grad = False
            if i not in idxs:
                param.data = torch.tensor([0.])
            i += 1

################################# Memory staff #################################
class AdapterLayer(nn.Module):
    """Wraps a linear layer with LoRA-like adapter. Wraps an existing OPT linear layer"""
    def __init__(self, module: nn.Linear, r: int = 8):
        super().__init__()
        self.module = module  # pre-trained (frozen) linear layer
        self.lora_A = nn.Linear(in_features=module.in_features,
                                out_features=r, bias=False,
                                dtype=module.weight.dtype,
                                device=module.weight.device)
        self.lora_B = nn.Linear(in_features=r,
                                out_features=module.out_features, bias=False,
                                dtype=module.weight.dtype,
                                device=module.weight.device)
        self.w = torch.tensor(1., requires_grad=True)
        # nn.init.kaiming_uniform_(self.adapter_A, a=5 ** 0.5)
        #self.adapter_B = nn.Parameter(torch.zeros(rank, module.out_features, device=module.weight.device))

    def forward(self, x):
        # Apply self.module and LoRA adapter, return the sum (self.module outputs + adapter outputs)
        frwd_module = self.module(x)
        frwd_adapter = self.w * self.lora_B(self.lora_A(x))
        return frwd_module + frwd_adapter
    
# num_peft_adapters = 0
# for name, param in model.named_parameters():
# # if name not in chosen_layers:
# #     param.requires_grad = False
#     param.requires_grad = False
# for i in range(12):
#     if num_peft_adapters < training_args.k:
#         model.deberta.encoder.layer[i].attention.self.query_proj = AdapterLayer(model.deberta.encoder.layer[i].attention.self.query_proj, r = training_args.lora_r)
#         num_peft_adapters += 1
#     if num_peft_adapters < training_args.k:
#         model.deberta.encoder.layer[i].attention.self.key_proj = AdapterLayer(model.deberta.encoder.layer[i].attention.self.key_proj, r = training_args.lora_r)
#         num_peft_adapters += 1
#     if num_peft_adapters < training_args.k:
#         model.deberta.encoder.layer[i].attention.self.value_proj = AdapterLayer(model.deberta.encoder.layer[i].attention.self.value_proj, r = training_args.lora_r)
#         num_peft_adapters += 1