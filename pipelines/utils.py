import json
import random
import torch
import datasets
import numpy as np

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

def set_device(device_no: int): # выбирает GPU-шку и выводит название
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_no}")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(device_no))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    return device

def make_mlm_dataset_form_mmlu(dataset):
    dataset_list = []
    for a in dataset:
        q = a['question']
        q = q.replace('_', '')
        q += ' ' + a['choices'][a['answer']]
        q = q.replace('.', '')
        q = q.replace('  ', ' ')
        q += '.'
        dataset_list.append({"text" : q})

    return_dataset = datasets.Dataset.from_list(dataset_list)
    return return_dataset