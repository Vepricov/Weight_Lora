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
)

from trl import ORPOConfig, ORPOTrainer, setup_chat_format

try:
    import adapters
    import optimizers
    import src.utils as utils
except ModuleNotFoundError:
    import pipelines.adapters as adapters
    import pipelines.optimizers as optimizers
    import pipelines.utils as utils

################################## Arguments ###################################
@dataclass
class ModelArguments:
    # model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    model_name: str = "meta-llama/Meta-Llama-3.1-8B"
    max_seq_length: int = 1000
    dtype: str = None
    load_in_4bit: bool = True
    bnb_4bit_quant_type="nf4"
    bnb_4bit_use_double_quant=True
    device_map="auto"
    new_model_name = "Mistral-7B-Instruct_new"

@dataclass
class TrainingArguments(TrainingArguments):
    learning_rate=8e-6
    beta=0.1
    lr_scheduler_type="linear"
    max_length=1024
    max_prompt_length=512
    per_device_train_batch_size=20
    per_device_eval_batch_size=20
    gradient_accumulation_steps=4
    optim="paged_adamw_8bit"
    num_train_epochs=50
    evaluation_strategy="steps"
    eval_steps=0.2
    logging_steps=1
    warmup_steps=10
    report_to="wandb" # "none" or "wandb"
    output_dir="./train_outputs/"
    max_steps=20
    k=18 # number of non-null loras (for WeightLora)

@dataclass
class DataArguments:
    dataset_name: str = "mlabonne/orpo-dpo-mix-40k"
    num_samples: int = 50
    test_size: float = 0.02
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

    model, tokenizer = setup_chat_format(model, tokenizer)
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

    def format_chat_template(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    dataset = dataset.map(
        format_chat_template,
        num_proc= os.cpu_count(),
    )
    dataset = dataset.train_test_split(test_size=data_args.test_size)

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

    orpo_args = ORPOConfig(
        learning_rate=training_args.learning_rate,
        beta=training_args.beta,
        lr_scheduler_type=training_args.lr_scheduler_type,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        optim=training_args.optim,
        num_train_epochs=training_args.num_train_epochs,
        evaluation_strategy=training_args.evaluation_strategy,
        eval_steps=training_args.eval_steps,
        logging_steps=training_args.logging_steps,
        warmup_steps=training_args.warmup_steps,
        report_to=training_args.report_to,
        output_dir=training_args.output_dir,
        max_steps=training_args.max_steps,
        run_name = run_name,
    )
    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        # peft_config=peft_config,
        tokenizer=tokenizer,
        optimizers=[optimizer, scheduler]
    )
    trainer.train()
    # trainer.save_model(model_args.new_model_name)
    ############################################################################

    ### Finishing ###
    num_non_zero = 0
    for name, param in model.named_parameters():
        if "weight_lora_w" in name:
            if param.sum().item() > 0:
                print(name, "; sum = ", param.sum().item())
                num_non_zero += 1
    print(f"Overall non-null: {num_non_zero} and k = {training_args.k}")

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
