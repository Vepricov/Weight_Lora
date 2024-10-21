import torch
import gc
import os
import yaml
import wandb
import peft
from omegaconf import OmegaConf

from transformers import (
    AutoModelForCausalLM,
    get_scheduler
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
        adapter_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False, 
            **OmegaConf.to_object(config.adapter_config.LoRA_config),
        )
    elif config.adapter_config.ft_strategy == "LoKR":
        adapter_config = peft.LoKrConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            **OmegaConf.to_object(config.adapter_config.LoKR_config)
        )
    elif config.adapter_config.ft_strategy == "LoHA":
        adapter_config = peft.LoHaConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            **OmegaConf.to_object(config.adapter_config.LoHA_config)
        )
    elif config.adapter_config.ft_strategy == "VERA":
        adapter_config = peft.VeraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            **OmegaConf.to_object(config.adapter_config.VERA_config)
        )
    elif config.adapter_config.ft_strategy == "ADALoRA":
        adapter_config = peft.AdaLoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            **OmegaConf.to_object(config.adapter_config.ADALoRA_config)
        )
    elif config.adapter_config.ft_strategy == "BOFT":
        adapter_config = peft.BOFTConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            **OmegaConf.to_object(config.adapter_config.BOFT_config)
        )
    else:
        raise ValueError(f"Incorrect FT type {config.adapter_config.ft_strategy}!")

    model_adapter = peft.get_peft_model(model, adapter_config)    
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
    #run_name = f"{config.model_name} + {optimizer.__class__.__name__}"
    run_name = f"{config.adapter_config.ft_strategy}"
    config.trainer_config["run_name"] = run_name

    training_args = SFTConfig(
        **OmegaConf.to_object(config.trainer_config),
    )
    eval_dataset=validation_dataset.shuffle(config.trainer_config.seed)
    eval_dataset = eval_dataset.select(range(64))
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
        eval_dataset=eval_dataset,
        # formatting_func=formatting_prompts_func,
        # data_collator=collator,
        # compute_metrics=utils.compute_accuracy,
        optimizers=[optimizer, scheduler]
    )
    # wandb.init(project = "SBER_LORA", config = config)
    # print(wandb.config.keys())
    # return 0
    trainer.train()
    if config.trainer_config.report_to == "wandb":
        if optimizer is not None:
            add_wandb_config = {"optimizer" : optimizer.__class__.__name__,
                                "scheduler" : scheduler.__class__.__name__,
                                }
        else:
            add_wandb_config = {"optimizer" : config.trainer_config.optim,
                                "scheduler" : config.trainer_config.lr_scheduler_type,
                                }

        wandb.config.update(add_wandb_config)
    ############################################################################

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()