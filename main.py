
from trainer import run
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a denoising diffusion model.")

    # Các tham số model và train
    parser.add_argument('--data_path', type=str, required=True, help='Path to the image dataset.')
    parser.add_argument('--image_size', type=int, default=128, help='Size of the images.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--learning_rate', type=float, default=8e-5, help='Learning rate.')
    parser.add_argument('--num_steps', type=int, default=700000, help='Number of training steps.')
    parser.add_argument('--gradient_accumulation', type=int, default=2, help='Number of gradient accumulation steps.')
    parser.add_argument('--ema_decay', type=float, default=0.995, help='EMA decay rate.')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training.')
    parser.add_argument('--calculate_fid', action='store_true', help='Calculate FID during training.')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models.')
    parser.add_argument('--checkpoint_interval', type=int, default=10000, help='Interval steps for saving checkpoints.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training.')
    
    # Tham số Gaussian Diffusion
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion steps.')
    parser.add_argument('--sampling_timesteps', type=int, default=250, help='Number of sampling timesteps.')


    # Tham số mô hình Unet
    parser.add_argument('--model_dim', type=int, default=64, help='Dimension of the model.')
    parser.add_argument('--model_dim_mults', type=int, nargs='+', default=[1, 2, 4, 8], help='Dimension multipliers for the model.')
    parser.add_argument('--flash_attn', action='store_true', help='Use flash attention.')

    args = parser.parse_args()
    run(args)