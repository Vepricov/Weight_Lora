import torch
import peft
import torch.nn as nn

class SingleNumberLayer(nn.Module):
    def __init__(self, init_value=1., device=None):
        super(SingleNumberLayer, self).__init__()
        self.value = nn.Parameter(torch.tensor(init_value, device=device))

    def forward(self, x):
        return self.value * x

class WeightLoraLayer(nn.Module):
    def __init__(self, module: nn.Linear, rank=8, w_init_value = 1., 
                 dtype=torch.float16):
        super().__init__()
        self.base_module = module  # base model
        # self.w = SingleNumberLayer(init_value=w_init_value, 
        #                            device=module.weight.device)
        self.w = nn.Linear(in_features=1, out_features=1, bias=False, dtype=dtype,
                           device=module.weight.device)
        self.w.weight.requires_grad = False
        #self.adapter_A = nn.Parameter(torch.empty(module.in_features, rank, 
        #                                          device=module.weight.device))
        self.adapter_A = nn.Linear(in_features=module.in_features,
                                   out_features=rank, bias=False, dtype=dtype,
                                   device=module.weight.device)
        # nn.init.kaiming_uniform_(self.adapter_A, a=5 ** 0.5)
        #self.adapter_B = nn.Parameter(torch.zeros(rank, module.out_features, 
        #                                          device=module.weight.device))
        self.adapter_B = nn.Linear(in_features=rank, bias=False, dtype=dtype,
                                   out_features=module.out_features,
                                   device=module.weight.device)
        # nn.init.zeros_(self.adapter_B)

    def forward(self, input):
        # Apply self.module and LoRA adapter, return the sum (self.module outputs + adapter outputs)
        module_outs = self.base_module(input)
        # adapter_outs = torch.matmul(input, self.adapter_A)
        # adapter_outs = torch.matmul(adapter_outs, self.adapter_B)
        adapter_outs = self.adapter_A(input)
        adapter_outs = self.adapter_B(adapter_outs)
        # if self.w is not None:
        #     # adapter_outs = torch.matmul(self.w, adapter_outs)
        # adapter_outs = self.w(adapter_outs)
        adapter_outs = self.w.weight * adapter_outs
        # return module_outs + adapter_outs
        return module_outs + adapter_outs

class AdapterLayer(nn.Module):
    """Wraps a linear layer with LoRA-like adapter. Wraps an existing OPT linear layer"""
    def __init__(self, module: nn.Linear):
        super().__init__()
        self.base_module = module  # pre-trained (frozen) linear layer
        self.adapter = nn.Linear(in_features=module.in_features,
                                 out_features=module.out_features,
                                 bias=False)
        # nn.init.kaiming_uniform_(self.adapter_A, a=5 ** 0.5)
        #self.adapter_B = nn.Parameter(torch.zeros(rank, module.out_features, device=module.weight.device))

    def forward(self, x):
        # Apply self.module and LoRA adapter, return the sum (self.module outputs + adapter outputs)
        frwd_module = self.base_module(x)
        frwd_adapter = self.adapter(x)
        return frwd_module + frwd_adapter
    
def get_peft_model(model, adapter, adapter_args={},
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]):
    if model.__class__.__name__ == 'LlamaForCausalLM':
        self_attn_num, mlp_num = 0, 0
        for name, _ in model.named_parameters():
            if 'self_attn.q_proj' in name:
                self_attn_num += 1
            if 'mlp.gate_proj' in name:
                mlp_num += 1
        if "q_proj" in target_modules:
            for num in range(self_attn_num):
                model.model.layers[num].self_attn.q_proj = \
                    adapter(model.model.layers[num].self_attn.q_proj, **adapter_args)
        if "k_proj" in target_modules:
            for num in range(self_attn_num):
                model.model.layers[num].self_attn.k_proj = \
                    adapter(model.model.layers[num].self_attn.k_proj, **adapter_args)
        if "v_proj" in target_modules:
            for num in range(self_attn_num):
                model.model.layers[num].self_attn.v_proj = \
                    adapter(model.model.layers[num].self_attn.v_proj, **adapter_args)
        if "o_proj" in target_modules:
            for num in range(self_attn_num):
                model.model.layers[num].self_attn.o_proj = \
                    adapter(model.model.layers[num].self_attn.o_proj, **adapter_args)
        if "gate_proj" in target_modules:
            for num in range(mlp_num):
                model.model.layers[num].mlp.gate_proj = \
                    adapter(model.model.layers[num].mlp.gate_proj, **adapter_args)
        if "up_proj" in target_modules:
            for num in range(mlp_num):
                model.model.layers[num].mlp.up_proj = \
                    adapter(model.model.layers[num].mlp.up_proj, **adapter_args)
        if "down_proj" in target_modules:
            for num in range(mlp_num):
                model.model.layers[num].mlp.down_proj = \
                    adapter(model.model.layers[num].mlp.down_proj, **adapter_args)
        # if "rank" in adapter_args.keys():
        #     r = adapter_args["rank"]
        # else:
        #     r = 8
        # config = {
        #     "peft_type": "LORA",
        #     # "task_type": "CAUSAL_LM",
        #     # "inference_mode": False,
        #     # "num_virtual_tokens": 20,
        #     # "token_dim": 1280,
        #     # "num_transformer_submodules": 1,
        #     # "num_attention_heads": 20,
        #     # "num_layers": 36,
        #     # "encoder_hidden_size": 1280,
        #     # "prefix_projection": False,
        #     # "postprocess_past_key_value_function": None,
        # }
        # peft_config = peft.get_peft_config(config)
        # model = peft.PeftModelForCausalLM(model=model, peft_config=peft_config)
        #model.is_quantized = False
    elif model.__class__.__name__ == 'RobertaForCausalLM':
        self_attn_num = 0
        for name, _ in model.named_parameters():
            if 'attention.self.query' in name and 'bias' not in name:
                self_attn_num += 1

        if "query" in target_modules:
            for num in range(self_attn_num):
                model.roberta.encoder.layer[num].attention.self.query = \
                    adapter(model.roberta.encoder.layer[num].attention.self.query, **adapter_args)
        if "key" in target_modules:
            for num in range(self_attn_num):
                model.roberta.encoder.layer[num].attention.self.key = \
                    adapter(model.roberta.encoder.layer[num].attention.self.key, **adapter_args)
        if "value" in target_modules:
            for num in range(self_attn_num):
                model.roberta.encoder.layer[num].attention.self.value = \
                    adapter(model.roberta.encoder.layer[num].attention.self.value, **adapter_args)
    else:
        raise NotImplementedError(f"{model.__class__.__name__}")

    return model
        
