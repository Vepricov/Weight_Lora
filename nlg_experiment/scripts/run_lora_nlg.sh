for dataset_name in cnn_dailymail
do
    for lr in 8e-5 5e-5 1e-4 3e-4 5e-4 8e-4 1e-3
    do
        for seed in 18 52 1917
        do
            clear
            CUDA_VISIBLE_DEVICES=0 python ./nlg_experiment/run_nlg.py \
            --dataset_name $dataset_name \
            --model_name_or_path facebook/bart-large \
            --dataset_config "3.0.0" \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16  \
            --gradient_accumulation_steps 6 \
            --max_val_samples 1000 \
            --learning_rate $lr \
            --lr_scheduler_type linear \
            --warmup_steps 100 \
            --max_steps 512 \
            --eval_steps 128 \
            --save_steps 256 \
            --ft_strategy LoRA \
            --lora_r 8 \
            --lora_alpha 32 \
            --lora_dropout 0.05 \
            --seed $seed \
            --predict_with_generate true \
            --report_to wandb # none or wandb
        done
    done
done