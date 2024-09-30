import gc
import os
import torch
import wandb
from dataclasses import dataclass 
from datasets import load_dataset

from peft import (
    LoraConfig, 
    WeightLoraConfig, 
    prepare_model_for_kbit_training, 
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    get_scheduler,
    DataCollatorForLanguageModeling
)
from trl import SFTConfig, SFTTrainer

try:
    import adapters
    import optimizers
    import utils
except ModuleNotFoundError:
    import pipelines.adapters as adapters
    import pipelines.optimizers as optimizers
    import pipelines.utils as utils

################################## Arguments ###################################
@dataclass
class ModelArguments:
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_seq_length: int = 1000
    dtype: str = None
    load_in_4bit: bool = True
    bnb_4bit_quant_type="nf4"
    bnb_4bit_use_double_quant=True
    device_map="auto"
    new_model_name = "Mistral-7B-Instruct_new"

@dataclass
class TrainingArguments(TrainingArguments):
    lr_scheduler_type="linear",
    max_seq_length=1024,
    learning_rate = 5e-4,
    weight_decay = 0.01,
    # max_prompt_length=512,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    optim="adamw_8bit",
    num_train_epochs=50,
    max_steps = 25, # overrides num_train_epochs
    eval_strategy="epoch",
    logging_steps=1,
    warmup_steps=2,
    report_to="none", # "none" or "wandb"
    output_dir="./train_outputs/",
    k=20 # number of non-null loras (for WeightLora)

@dataclass
class DataArguments:
    dataset_name: str = "mlabonne/FineTome-Alpaca-100k"
    num_samples: int = 1000
    test_size: float = 0.02
    num_proc: int = 4
################################################################################

def main():
    ### Start ###
    model_args = ModelArguments
    training_args = TrainingArguments
    data_args = DataArguments
    utils.set_seed(training_args.seed)
    os.environ["WANDB_PROJECT"] = "SBER_LORA"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'
    if torch.cuda.get_device_capability()[0] >= 8:
        attn_implementation = "flash_attention_2"
        torch_dtype = torch.bfloat16
    else:
        attn_implementation = "eager"
        torch_dtype = torch.float16
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("We will use the GPU:", torch.cuda.get_device_name(i))

    ### BNB config ###
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_args.load_in_4bit,
        bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
    )
    ### Load model ###
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        quantization_config=bnb_config,
        device_map=model_args.device_map,
        attn_implementation=attn_implementation
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    ### Load tokenizer ###
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

    ############################### PEFT config ################################
    target_modules=['up_proj', 'down_proj', 'gate_proj', 
                    'k_proj', 'q_proj', 'v_proj', 'o_proj']
    peft_config = WeightLoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )
    ############################################################################

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    num_loras = 0
    for name, _ in model.named_parameters():
        if "lora_A" in name:
            num_loras += 1
    print(f"There are {num_loras} Lora Adapters")

    ######################### Optimizer and Scheduler ##########################
    optimizer, scheduler = None, None
    optimizer = optimizers.WeightAdamW(
        model.parameters(), 
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        k=training_args.k,
    )
    scheduler = get_scheduler(
        name=training_args.lr_scheduler_type, 
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps
    )
    ############################################################################

    ### Dataset ###
    dataset = load_dataset(data_args.dataset_name, split="all")
    dataset = dataset.shuffle(seed=training_args.seed).select(range(data_args.num_samples))

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["output"]])
    tokenized_dataset = dataset.map(preprocess_function,
                                    batched=True, num_proc=data_args.num_proc,
                                    remove_columns=dataset.column_names)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.02)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ################################# Training #################################
    run_name = None
    if training_args.report_to == "wandb":
        # if optimizer is not None:
        #     run_name=optimizer.__class__.__name__
        # else:
        #     run_name=training_args.optim
        if peft_config.__class__.__name__ == "LoraConfig":
            run_name = "Lora"
        elif peft_config.__class__.__name__ == "WeightLoraConfig":
            run_name = f"WeightLora, k={training_args.k}"
        else:
            run_name = "ZALUPA"

    sft_config = SFTConfig(
        lr_scheduler_type=training_args.lr_scheduler_type,
        max_seq_length=training_args.max_seq_length,
        learning_rate = training_args.learning_rate,
        weight_decay = training_args.weight_decay,
        # max_prompt_length=training_args.max_prompt_length,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        optim=training_args.optim,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=training_args.max_steps,
        eval_strategy=training_args.eval_strategy,
        logging_steps=training_args.logging_steps,
        warmup_steps=training_args.warmup_steps,
        report_to=training_args.report_to,
        output_dir=training_args.output_dir,
        run_name=run_name,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=[optimizer, scheduler]
    )
    trainer.train()
    # trainer.save_model(model_args.new_model_name)
    ############################################################################

    ### Finishing ###
    for name, param in model.named_parameters():
        if "weight_lora_w" in name:
            print(name, "; sum = ", param.sum().item())

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
