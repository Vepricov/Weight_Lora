import torch
import gc
import os
import yaml
import wandb
from omegaconf import OmegaConf

from transformers import (
    AutoModelForCausalLM,
    get_scheduler
)
from peft import (
    BOFTConfig, 
    get_peft_model, 
    LoraConfig, 
    TaskType
)
from trl import (
    SFTConfig, 
    SFTTrainer, 
    DataCollatorForCompletionOnlyLM
)

from src import (
    optimizers,
    utils
)

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.create(config)
    for i in range(torch.cuda.device_count()):
        print("We will use the GPU:", torch.cuda.get_device_name(i))
    utils.set_seed(config.trainer_config.seed)

    auxiliary_train_dataset, _, validation_dataset, _ =\
        utils.mmlu_preporcess(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
    )

    print(f"Using eval strategy {config.adapter_config.ft_strategy}")
    if config.adapter_config.ft_strategy == "LoRA":
        adapter_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, 
            **OmegaConf.to_object(config.adapter_config.LoRA_config),
        )
    elif config.adapter_config.ft_strategy == "BOFT":
        adapter_config = BOFTConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            **OmegaConf.to_object(config.adapter_config.BOFT_config)
        )
    else:
        raise ValueError("Incorrect FT type")

    model_adapter = get_peft_model(model, adapter_config)    
    model_adapter.print_trainable_parameters()
    
    ######################### Optimizer and Scheduler ##########################
    optimizer, scheduler = None, None
    # optimizer = optimizers.AdamW(
    #     model.parameters(), 
    #     lr=config.trainer_config.learning_rate,
    #     weight_decay=config.trainer_config.weight_decay,
    # )
    # scheduler = get_scheduler(
    #     name=config.trainer_config.lr_scheduler_type, 
    #     optimizer=optimizer,
    #     num_warmup_steps=config.trainer_config.warmup_steps,
    #     num_training_steps=config.trainer_config.max_steps
    # )
    import prodigyopt
    optimizer = prodigyopt.Prodigy(
        model.parameters(), 
        lr=1.
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.trainer_config.max_steps
    )
    ############################# Training #####################################
    os.environ["WANDB_PROJECT"] = "SBER_LORA"
    run_name = f"{config.model_name} + {optimizer.__class__.__name__}"
    config.trainer_config["run_name"] = run_name

    training_args = SFTConfig(
        **OmegaConf.to_object(config.trainer_config),
    )
    trainer = SFTTrainer(
        model=model_adapter,
        args=training_args,
        # args=SFTConfig(
        #     output_dir="/tmp",
        #     per_device_train_batch_size=1,
        #     per_device_eval_batch_size=2,
        #     fp16=True,
        # ),
        train_dataset=auxiliary_train_dataset,
        eval_dataset=validation_dataset.shuffle(42).select(range(64)),
        # formatting_func=formatting_prompts_func,
        # data_collator=collator,
        # compute_metrics=utils.compute_accuracy,
        optimizers=[optimizer, scheduler]
    )

    trainer.train()
    ############################################################################

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()