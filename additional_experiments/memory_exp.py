with open(f"mem_{training_args.lora_r}_{training_args.per_device_train_batch_size}.txt", "a") as f:
    f.write(f"k={training_args.k}: {train_metrics['train_memory_gb']}\n")