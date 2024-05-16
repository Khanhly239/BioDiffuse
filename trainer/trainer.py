import logging
import argparse
import torch
import os
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_model(model, path):
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Model
    model = Unet(
        dim = args.model_dim,
        dim_mults = args.model_dim_mults,
        flash_attn = args.flash_attn
    ).to(device)

    # Gaussian Diffusion
    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps
    )

    #Trainer
    trainer = Trainer(
        diffusion,
        folder=args.data_path,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        train_num_steps=args.num_steps,
        gradient_accumulate_every=args.gradient_accumulation,
        ema_decay=args.ema_decay,
        amp=args.mixed_precision,
        calculate_fid=args.calculate_fid,
        args = args
    )
    trainer.train()