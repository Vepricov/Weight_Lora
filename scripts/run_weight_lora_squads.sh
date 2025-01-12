for dataset_name in squad squad_v2
do
    for lr in 8e-5 5e-5 1e-4 3e-4 5e-4 8e-4 1e-3
    do
        for seed in 18 52 1917
        do
            clear
            CUDA_VISIBLE_DEVICES=4 python run_experiment.py \
            --dataset_name squad \
            --model_name_or_path microsoft/deberta-v3-base \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 6 \
            --learning_rate $lr \
            --lr_scheduler_type linear \
            --warmup_steps 100 \
            --max_steps 512 \
            --eval_steps 64 \
            --save_steps 256 \
            --ft_strategy WeightLoRA \
            --lora_r 8 \
            --lora_alpha 32 \
            --lora_dropout 0.05 \
            --k 10 \
            --seed $seed \
            --report_to wandb # none or wandb
        done
    done
done