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
    ################# Model, Tokenizer and Dataset Downloading #################
    if data_args.dataset_name == "glue":
        train_dataset, eval_dataset, datasets, data_collator, compute_metrics, model, tokenizer =\
            utils.glue_preprocess(
                data_args, training_args, model_args
            )
    else:
        raise ValueError(f"Wrong dataset name: {data_args.dataset_name}!")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    ############################### PEFT Adapters ##############################
    print(f"Using eval strategy {training_args.ft_strategy}")
    all_params_before_peft, _ = utils.print_trainable_parameters(model, verbose=False)
    training_args.model_name = model_args.model_name_or_path         # for wandb
    peft_args = config.get_peft_arguments(training_args)
    if peft_args is not None:
        model = peft.get_peft_model(model, peft_args)
    ############################### Wandb Saves ################################
    training_args.all_params, training_args.trainable_params = \
        utils.print_trainable_parameters(model)
    training_args.num_peft_adapters = utils.count_atapters(model, training_args.ft_strategy)
    training_args.peft_params = training_args.all_params - all_params_before_peft
    training_args.train_proportion = training_args.trainable_params / training_args.all_params * 100 
    training_args.peft_proportion = training_args.peft_params / training_args.all_params * 100 
    ######################### Optimizer and Scheduler ##########################
    optimizer, scheduler = None, None
    # optimizer = optimizers.WeightAdamW(
    #     model.parameters(), 
    #     lr=training_args.learning_rate,
    #     weight_decay=training_args.weight_decay,
    #     k=training_args.k,
    # )
    compressor_params = {"compression_rate": training_args.compression_rate,
                         "K" : training_args.K_compress, "b" : training_args.b, 
                         "proj" : None}
    compression_name = training_args.compression_name
    # compression_name = None
    optimizer = optimizers.QSGD(
        model.parameters(), 
        lr=training_args.learning_rate,
        compression_name=compression_name,
        compressor_params=compressor_params
    )
    scheduler = get_scheduler(
        name=training_args.lr_scheduler_type, 
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps
    )
    ############################### Wandb Saves ################################
    os.environ["WANDB_PROJECT"] = "ICLR KAWASAKI"
    training_args.label_names = ["labels"] # peft and compute_metrics() problem
    # run_name = f"{config.model_name} + {optimizer.__class__.__name__}"
    # run_name = f"[{training_args.ft_strategy}, k={training_args.k}] {data_args.task_name}"
    # run_name = "[TEST]"
    run_name = f"{compression_name}, compr_rate = {compressor_params['compression_rate']}"
    training_args.run_name = run_name
    training_args.output_dir = f"{training_args.output_dir}/{run_name}"
    if optimizer is not None:
        training_args.optimizer = optimizer.__class__.__name__
        training_args.scheduler = scheduler.__class__.__name__
    else:
        training_args.optimizer = training_args.optim
        training_args.scheduler = training_args.lr_scheduler_type
    training_args.benchmark_name = data_args.dataset_name
    training_args.tsk_name = data_args.task_name
    ############################# Training #####################################
    print(f"Len of train / eval datasets = {len(train_dataset)} / {len(eval_dataset)}")
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_evaluate else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=[optimizer, scheduler]
    )
    if training_args.do_train:
        train_result = trainer.train()
        train_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        train_metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        train_metrics["train_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
        train_metrics["train_runtime"] /= 60
        if training_args.ft_strategy == "WeightLoRA":
            i = 0
            for name, param in model.named_parameters():
                if "weight_lora_w" in name:
                    if param.sum().item() > 0:
                        i += 1
                        train_metrics[f"active_adapters_{i}"] = name

        trainer.save_model()

        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()
        if "wandb" in training_args.report_to:
            wandb.config.update(train_metrics, allow_val_change=True)
    ################################ Evaluation ################################
    if training_args.do_evaluate:
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            eval_metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
            trainer.log_metrics("Eval_%s"%task, eval_metrics)
            trainer.save_metrics("Eval_%s"%task, eval_metrics)
        eval_metrics["eval_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
        eval_metrics["eval_runtime"] /= 60
        if "wandb" in training_args.report_to:
            wandb.config.update(eval_metrics, allow_val_change=True)
    ############################################################################

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()