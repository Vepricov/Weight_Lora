import torch, gc, os, wandb, peft
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    HfArgumentParser,
    get_scheduler
)
from src import (
    config,
    optimizers,
    utils
)

def main():
    for i in range(torch.cuda.device_count()):
        print("We will use the GPU:", torch.cuda.get_device_name(i))
    parser = HfArgumentParser((
        config.ModelArguments, 
        config.DataTrainingArguments, 
        config.TrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    utils.set_seed(training_args.seed)
    ################# Model, tokenizer and dataset downloading #################
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    if data_args.dataset_name == "glue":
        train_dataset, eval_dataset, data_collator, compute_metrics = utils.glue_preprocess(
            data_args, training_args, tokenizer, model
        )
    else:
        raise ValueError(f"Wrong dataset name: {data_args.dataset_name}!")
    ############################## PEFT Adapters ###############################
    print(f"Using eval strategy {training_args.ft_strategy}")
    if training_args.ft_strategy != "Full":
        if training_args.ft_strategy == "LoRA":
            peft_args = config.LoraArguments
        elif training_args.ft_strategy == "LoKR":
            peft_args = config.LokrArguments
        elif training_args.ft_strategy == "LoHA":
            peft_args = config.LohaArguments
        elif training_args.ft_strategy == "VERA":
            peft_args = config.VeraArguments
        elif training_args.ft_strategy == "ADALoRA":
            peft_args = config.Adalorarguments
        elif training_args.ft_strategy == "BOFT":
            peft_args = config.BoftArguments
        else:
            raise ValueError(f"Incorrect FT type {training_args.ft_strategy}!")
        
        if "deberta" in model_args.model_name_or_path:
            peft_args.target_modules = ["query_proj", "key_proj", "value_proj",
                                        "intermediate.dence", "output.dence"]
        elif ("llama", "mistralai") in model_args.model_name_or_path:
            peft_args.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"]
    
        model = peft.get_peft_model(model, peft_args)    
    utils.print_trainable_parameters(model)
    ######################### Optimizer and Scheduler ##########################
    optimizer, scheduler = None, None
    # optimizer = optimizers.AdamW(
    #     model.parameters(), 
    #     lr=training_args.learning_rate,
    #     weight_decay=training_args.weight_decay,
    # )
    # scheduler = get_scheduler(
    #     name=training_args.lr_scheduler_type, 
    #     optimizer=optimizer,
    #     num_warmup_steps=training_args.warmup_steps,
    #     num_training_steps=training_args.max_steps
    # )
    # import prodigyopt
    # optimizer = prodigyopt.Prodigy(
    #     model.parameters(), 
    #     lr=1.
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max=training_args.max_steps
    # )
    ############################# Training #####################################
    os.environ["WANDB_PROJECT"] = "SBER_LORA"
    #run_name = f"{config.model_name} + {optimizer.__class__.__name__}"
    run_name = f"[{training_args.ft_strategy}], {data_args.task_name}"
    training_args.run_name = run_name
    training_args.label_names = ["labels"]

    print(f"Len of train / eval datasets = {len(train_dataset)} / {len(eval_dataset)}")
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    if training_args.report_to == "wandb":
        if optimizer is not None:
            add_wandb_config = {"optimizer" : optimizer.__class__.__name__,
                                "scheduler" : scheduler.__class__.__name__,
                                }
        else:
            add_wandb_config = {"optimizer" : training_args.optim,
                                "scheduler" : training_args.lr_scheduler_type,
                                }

        wandb.config.update(add_wandb_config)
    ############################################################################

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()