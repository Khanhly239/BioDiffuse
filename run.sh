python main.py \
    --data_path /media/mountHDD3/data_storage/z2h/ISIC/ISIC2017/all_data \
    --image_size 128 --batch_size 32 --learning_rate 8e-5 --num_steps 500 \
    --gradient_accumulation 2 --ema_decay 0.995 --mixed_precision --calculate_fid --save_dir ./models \
    --checkpoint_interval 10000 --epochs 1 --timesteps 1000 --sampling_timesteps 250 \
    --model_dim 64 --model_dim_mults 1 2 4 8 --flash_attn
