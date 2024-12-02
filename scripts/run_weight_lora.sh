#for task_name in cola mnli mrpc qnli qqp rte sst2 stsb
for task_name in rte
do
    for k in 1 5 10
    do
        CUDA_VISIBLE_DEVICES=1 python run_experiment.py \
            --dataset_name glue \
            --task_name $task_name \
            --model_name_or_path microsoft/deberta-v3-base \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 6 \
            --learning_rate 8e-4 \
            --learning_rate_w 1e1 \
            --lr_scheduler_type linear \
            --warmup_steps 100 \
            --max_steps 512 \
            --eval_steps 64 \
            --save_steps 256 \
            --ft_strategy WeightLoRA \
            --lora_r 8 \
            --lora_alpha 32 \
            --lora_dropout 0.05 \
            --k $k \
            --report_to wandb # none or wandb --learning_rate_w 1e10 \
    done
done