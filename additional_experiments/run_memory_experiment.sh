for ((k=1;k<=36;k++));
do
    CUDA_VISIBLE_DEVICES=7 python run_experiment.py \
        --dataset_name glue \
        --task_name mnli \
        --model_name_or_path microsoft/deberta-v3-base \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 6 \
        --learning_rate tuned \
        --learning_rate_w 1 \
        --lr_scheduler_type cosine \
        --warmup_steps 50 \
        --max_steps 5 \
        --eval_steps 64 \
        --save_steps 256 \
        --ft_strategy LoRA \
        --lora_r 16 \
        --lora_dropout 0.05 \
        --lora_alpha 32 \
        --use_fat true \
        --do_evaluate false \
        --fat_step 5 \
        --max_fat_steps 2 \
        --lora_extention smart \
        --seed 52 \
        --k $k \
        --report_to none
done