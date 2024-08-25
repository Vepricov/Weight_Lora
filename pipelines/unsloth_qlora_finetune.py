import numpy as np
import json
import random
import os
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, HfArgumentParser
from dataclasses import dataclass 
from datasets import Dataset

@dataclass
class ModelArguments:
    #model_name: str = "/media/ssd-3t/akazakov/llama31instr/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693"
    model_name: str = "unsloth/Meta-Llama-3.1-8B"
    max_seq_length: int = 2048
    dtype: str = None
    load_in_4bit: bool = True

@dataclass
class TrainingArguments(TrainingArguments):
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    num_train_epochs: int = 5
    learning_rate: float = 5e-4
    fp16: bool = not is_bfloat16_supported()
    bf16: bool = is_bfloat16_supported()
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 18
    output_dir: str = "train_outputs"
    # output_dir: str = None
    device_no = 0
    max_steps: int = 1
    report_to: str = "none" # "none" or "wandb"

 
@dataclass
class DataArguments:
    train_file: str = 'data/train_ft_short_system.jsonl'

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
    os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_no}")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(device_no))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    return device 
 
def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    device = set_device(training_args.device_no)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name,
        max_seq_length=model_args.max_seq_length,
        dtype=model_args.dtype,
        load_in_4bit=model_args.load_in_4bit,
        device_map=device
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=training_args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3"
    )

    transformed_data = load_and_transform_jsonl(data_args.train_file)
    train_dataset = Dataset.from_list(transformed_data)

    def formatting_prompts_func(examples):
        convos = examples["chat"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}
    
    dataset = train_dataset.map(formatting_prompts_func, batched=True)

    os.environ["WANDB_PROJECT"] = "SBER_LORA"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            warmup_steps=training_args.warmup_steps,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            logging_steps=training_args.logging_steps,
            optim=training_args.optim,
            weight_decay=training_args.weight_decay,
            lr_scheduler_type=training_args.lr_scheduler_type,
            seed=training_args.seed,
            output_dir=training_args.output_dir,
            max_steps=training_args.max_steps,
            report_to=training_args.report_to,
            run_name=training_args.optim
        ),
    )

    trainer_stats = trainer.train()

if __name__ == "__main__":
    main()
