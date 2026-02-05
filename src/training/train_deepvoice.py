"""
Training script for DEEP-VOICE dataset.
Usage: python src/training/train_deepvoice.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.utils.logger import setup_logger
from src.data.deepvoice_dataset import DeepVoiceDataset
from src.models.resnet_bilstm import ResNetBiLSTM
from src.training.trainer import Trainer

def main():
    config = load_config("configs/deepvoice.yaml")
    
    # Add model name for logger
    config['model']['name'] = 'DeepVoice_ResNetBiLSTM'
    
    seed_everything(42)
    
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    logger = setup_logger("DeepVoice_Training", log_dir=config['training']['output_dir'])
    
    logger.info("Loading DEEP-VOICE dataset...")
    
    train_dataset = DeepVoiceDataset(config, subset="train")
    dev_dataset = DeepVoiceDataset(config, subset="dev")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=0
    )
    
    logger.info(f"Train: {len(train_dataset)} samples, Dev: {len(dev_dataset)} samples")
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = ResNetBiLSTM(config)
    
    # Create trainer and run
    trainer = Trainer(model=model, config=config, device=device)
    trainer.run(train_loader, dev_loader, epochs=config['training']['epochs'])
    
    logger.info("Training complete!")
    logger.info(f"Best model saved to: {config['training']['output_dir']}/best_model.pth")

if __name__ == "__main__":
    main()
