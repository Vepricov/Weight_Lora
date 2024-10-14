import json
import random
import torch
import datasets
import os
import numpy as np
import transformers
import evaluate
from datasets import load_dataset
# from unsloth.chat_templates import get_chat_template

# Loading MMLU categories
if not os.path.exists('./categories.py'):
    os.system("wget https://raw.githubusercontent.com/hendrycks/test/master/categories.py")
from src.categories import subcategories, categories as categories_inv

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

def mmlu_preporcess(config):
    for subcat_name, cat_names in subcategories.items():
        subcategories[subcat_name] = cat_names[0] if isinstance(cat_names, list) else cat_names
    categories = {}
    for cat_name, subcats in categories_inv.items():
        for subcat in subcats:
            categories[subcat] = cat_name

    def subcat_to_cat(subcat):
        cat_name = subcategories[subcat]
        cat_name = categories[cat_name]
    
        return cat_name

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, 
        padding_side=config.tokenizer_config.padding_side,
        model_max_length=config.max_length,
    )
    # tokenizer_mistral = transformers.AutoTokenizer.from_pretrained(
    #     "mistralai/Mistral-7B-Instruct-v0.3", 
    #     padding_side=config.tokenizer_config.padding_side,
    # #     model_max_length=512,
    # )
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer = get_chat_template(
    #     tokenizer,
    #     chat_template="llama-3"
    # )
    # print(tokenizer.chat_template) # = "<prompt_template>"
    # tokenizer.chat_template = tokenizer_mistral.chat_template
    EOS_TOKEN = tokenizer.eos_token
    mmlu_dataset =  load_dataset("cais/mmlu", config.task_name)

    few_shot_datasets = {
        subject: mmlu_dataset['dev'].filter(lambda row: row['subject'] == subject)
        for subject in set(mmlu_dataset['dev']['subject'])
    }
    def prepare_question(examples):
        prompt = f"{examples['question']}\n"
        for letter, choice in zip(('A', 'B', 'C', 'D'), examples['choices']):
            prompt += f"{letter}. {choice}\n"

        answer = chr(65 + examples['answer'])
        
        return prompt, answer

    def prepare_prompt(examples, dev_dataset = None):
        if dev_dataset:
            yield from map(prepare_question, dev_dataset)
        
        yield prepare_question(examples)
    def prepare_instruction_text(example):
        instructions = [
            {"role": "system", "content": f"The following are multiple choice questions (with answers) about {example['subject']}. Output 'A', 'B', 'C', or 'D'. Full answer not needed."},
        ]

        if config['n_shots'] and example['subject']:
            few_shot_dataset = few_shot_datasets[example['subject']]
            few_shot_dataset = few_shot_dataset.select(range(config['n_shots']))
        else:
            few_shot_dataset = None
        
        for prompt, ans in prepare_prompt(example, dev_dataset=few_shot_dataset):
            instructions.append({"role": "user", "content": prompt})
            instructions.append({"role": "assistant", "content": ans})
        
        text = tokenizer.apply_chat_template(
            instructions,
            tokenize=False
        )
        
        return {'text': text}
    
    def r_replace(line, old, new):
        return line[::-1].replace(old[::-1], new[::-1], 1)[::-1]
    def remove_answer(example):
        text_wa_answer = example['text']
        text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0][:-1]
        
        return {'text_wa_answer': text_wa_answer}
    
    instructions_datasets = mmlu_dataset.map(prepare_instruction_text, batched=False, num_proc=2)
    instructions_datasets['validation'] = instructions_datasets['validation'].map(remove_answer, batched=False)
    instructions_datasets['test'] = instructions_datasets['test'].map(remove_answer, batched=False)

    instructions_datasets.set_format("torch")

    # Accessing the train, validation, and test splits
    validation_dataset = instructions_datasets["validation"]
    test_dataset = instructions_datasets["test"]
    dev_dataset = instructions_datasets["dev"]  # dataset for few shot
    auxiliary_train_dataset  = instructions_datasets['auxiliary_train']

    # Check the size of each split
    print(f"Validation dataset size: {len(validation_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Dev dataset size: {len(dev_dataset)}")
    print(f"Auxiliary train dataset size: {len(auxiliary_train_dataset)}")

    return auxiliary_train_dataset, test_dataset, validation_dataset, dev_dataset

def process_prediction(pred):
    pred = pred['generated_text']
    
    pred = pred.strip().upper()
    
    pred = pred[0] if pred else 'I'
    pred = pred if pred in {'A', 'B', 'C', 'D'} else 'I'
    
    return pred

def compute_accuracy(pred):   
    model_preds, labels = pred.predictions, pred.label_ids
    print("$$$\n", print(pred.inputs), "\n$$$")
    print("$$$\n", print(pred.label_ids), "\n$$$")
    print("$$$\n", print(pred.predictions), "\n$$$")
    raise ValueError("ZALUPA")
    accuracy_metric = evaluate.load("accuracy")
    model_preds = list(map(process_prediction, model_preds))
    
    model_preds  = torch.LongTensor(list(map(ord, model_preds)))
    actual_labels = ord('A') + labels
    incorrect_labels = actual_labels.new_full(actual_labels.shape, ord('I'))
    
    acc_res = accuracy_metric.compute(predictions=model_preds, references=actual_labels)['accuracy']
    corr_res = 1.0 - accuracy_metric.compute(predictions=model_preds, references=incorrect_labels)['accuracy']
    
    return {'accuracy': acc_res, 'correctness': corr_res}
