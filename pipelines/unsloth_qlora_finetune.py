import os
# import torch_optimizer
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, HfArgumentParser, get_scheduler
from dataclasses import dataclass 
from datasets import Dataset

try:
    import optimizers
    import utils
except ModuleNotFoundError:
    import pipelines.optimizers as optimizers
    import pipelines.utils as utils

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
    learning_rate: float = 1e-1
    fp16: bool = not is_bfloat16_supported()
    bf16: bool = is_bfloat16_supported()
    logging_steps: int = 1
    optim: str = "adamw_hf"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 18
    output_dir: str = "train_outputs"
    # output_dir: str = None
    sign_step: int = 5000
    max_grad_norm: float = 1.0
    max_steps: int = 2 # overrides num_train_epochs
    report_to: str = "none" # "none" or "wandb"
 
@dataclass
class DataArguments:
    train_file: str = 'data/train_ft_short_system.jsonl'
 
def main():
    # parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    # model_args, training_args, data_args = parser.parse_args_into_dataclasses()
    model_args = ModelArguments
    training_args = TrainingArguments
    data_args = DataArguments

    utils.set_seed(training_args.seed)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name,
        max_seq_length=model_args.max_seq_length,
        dtype=model_args.dtype,
        load_in_4bit=model_args.load_in_4bit
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

    transformed_data = utils.load_and_transform_jsonl(data_args.train_file)
    train_dataset = Dataset.from_list(transformed_data)

    def formatting_prompts_func(examples):
        convos = examples["chat"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}
    
    dataset = train_dataset.map(formatting_prompts_func, batched=True)

    optimizer, scheduler = None, None
    ######################### Optimizer and Scheduler ##########################
    # optimizer = optimizers.signAdamW(model.parameters(), 
    #                                  lr=training_args.learning_rate,
    #                                  weight_decay=training_args.weight_decay)
    # scheduler = get_scheduler(name=training_args.lr_scheduler_type, 
    #                           optimizer=optimizer, 
    #                           num_warmup_steps=training_args.warmup_steps,
    #                           num_training_steps=training_args.max_steps)
    import dadaptation
    optimizer = dadaptation.DAdaptAdam(model.parameters(), 
                                       lr=training_args.learning_rate,
                                       weight_decay=training_args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=training_args.max_steps)
    ############################################################################

    run_name = None
    if training_args.report_to == "wandb":
        if optimizer is not None:
            run_name=optimizer.__class__.__name__
        else:
            run_name=training_args.optim
    
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
            run_name=run_name,
            max_grad_norm=training_args.max_grad_norm
        ),
        optimizers=[optimizer, scheduler]
    )

    trainer_stats = trainer.train()

    if training_args.report_to == "wandb":
        import wandb
        if optimizer is not None:
            add_wandb_config = {"optimizer" : optimizer.__class__.__name__,
                                "scheduler" : scheduler.__class__.__name__,
                                }
        else:
            add_wandb_config = {"optimizer" : training_args.optim,
                                "scheduler" : training_args.lr_scheduler_type,
                                }

        wandb.config.update(add_wandb_config)

if __name__ == "__main__":
    main()
