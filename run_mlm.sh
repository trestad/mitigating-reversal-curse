python run_mlm.py --do_train \
    --save_strategy "steps" \
    --save_steps 9000 \
    --evaluation_strategy "no" \
    --warmup_steps 100 \
    --learning_rate 2e-5 \
    --max_length 256 \
    --gradient_accumulation_steps 1 \
    --report_to 'none' \
    --per_device_train_batch_size 4 \
    --dataloader_num_workers 1 \
    --logging_steps 1 --num_train_epochs 20 \
    --data $1 \
    --output_dir $2 \
    --model_name_or_path $3 \
    --fp16 \
   