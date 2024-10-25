for ft_strategy in LoRA LoKR LoHA VERA ADALoRA Full
do
    CUDA_VISIBLE_DEVICES=2 python run_experiment.py \
        --dataset_name glue \
        --task_name cola \
        --model_name_or_path "microsoft/deberta-v3-base" \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 6 \
        --learning_rate 8e-4 \
        --lr_scheduler_type linear \
        --warmup_steps 100 \
        --max_steps 512 \
        --eval_steps 16 \
        --save_steps 256 \
        --ft_strategy $ft_strategy \
        --report_to wandb # none or wandb
done

