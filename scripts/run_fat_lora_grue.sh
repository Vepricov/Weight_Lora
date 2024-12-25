for task_name in cola mnli mrpc qnli qqp rte sst2 stsb
do
    clear
    CUDA_VISIBLE_DEVICES=2 python run_experiment.py \
        --dataset_name glue \
        --task_name $task_name \
        --model_name_or_path microsoft/deberta-v3-base \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 6 \
        --learning_rate 8e-4 \
        --learning_rate_w 8e-0 \
        --lr_scheduler_type cosine \
        --warmup_steps 50 \
        --max_steps 512 \
        --eval_steps 64 \
        --save_steps 256 \
        --ft_strategy WeightLoRA \
        --lora_r 4 \
        --lora_dropout 0.05 \
        --lora_alpha 32 \
        --use_fat true \
        --do_evaluate true \
        --fat_step 5 \
        --max_fat_steps 2 \
        --lora_extention smart \
        --report_to wandb # none or wandb
done