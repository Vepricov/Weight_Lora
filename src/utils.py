import json
import random
import torch
import datasets
import os
import numpy as np
import transformers
import logging
from datasets import load_dataset, load_metric
from transformers import (
    PretrainedConfig,
    EvalPrediction,
    DataCollatorWithPadding,
    default_data_collator,
)
logger = logging.getLogger(__name__)
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

def mmlu_process_prediction(pred):
    pred = pred['generated_text']
    
    pred = pred.strip().upper()
    
    pred = pred[0] if pred else 'I'
    pred = pred if pred in {'A', 'B', 'C', 'D'} else 'I'
    
    return pred

def glue_preprocess(data_args, training_args, tokenizer, model):
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        raise RuntimeError("Pass the data_args.task_name !!!")
    
    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    
    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(
        preprocess_function, 
        batched=True, 
        load_from_cache_file=not data_args.overwrite_cache
    )
    train_dataset = None
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    eval_dataset = None
    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
    
    # Get the metric function
    metric = load_metric("glue", data_args.task_name)
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif data_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    return train_dataset, eval_dataset, data_collator, compute_metrics
