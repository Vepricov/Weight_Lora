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
    training_args.model_name = model_args.model_name_or_path # for wandb
    peft_args = config.get_peft_arguments(training_args)
    if peft_args is not None:
        model = peft.get_peft_model(model, peft_args)    
    training_args.all_params, training_args.trainable_params, \
        training_args.proportion=utils.print_trainable_parameters(model) # wandb
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
    training_args.label_names = ["labels"] # peft and compute_metrics() problem
    #run_name = f"{config.model_name} + {optimizer.__class__.__name__}"
    run_name = f"[{training_args.ft_strategy}] {data_args.task_name}"
    training_args.run_name = run_name                                # for wandb
    if optimizer is not None:
        training_args.optimizer = optimizer.__class__.__name__       # for wandb
        training_args.scheduler = scheduler.__class__.__name__       # for wandb
    else:
        training_args.optimizer = training_args.optim                # for wandb
        training_args.scheduler = training_args.lr_scheduler_type    # for wandb
    training_args.benchmark_name = data_args.dataset_name            # for wandb
    training_args.tsk_name = data_args.task_name                     # for wandb

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

    # if training_args.report_to == "wandb":
    #     if optimizer is not None:
    #         add_wandb_config = {"optimizer" : optimizer.__class__.__name__,
    #                             "scheduler" : scheduler.__class__.__name__,
    #                             }
    #     else:
    #         add_wandb_config = {"optimizer" : training_args.optim,
    #                             "scheduler" : training_args.lr_scheduler_type,
    #                             }

    #     wandb.config.update(add_wandb_config)
    ############################################################################

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()