import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.models.resnet_bilstm import ResNetBiLSTM
from src.data.asvspoof_dataset import ASVSpoofDataset
from src.training.trainer import Trainer

def main():
    # 1. Load Config (Reuse fuzzy config for model params)
    config = load_config('configs/deepvoice_fuzzy.yaml')
    
    # Update for Phase 2
    config['training']['batch_size'] = 32 # Reduce batch size for larger images/stability? Or keep 32.
    config['training']['learning_rate'] = 1e-4
    config['training']['num_epochs'] = 10 
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Datasets (Large Scale)
    # User path: /Users/agraw/Desktop/sem 6/trustcall/data/ASVspoof 2019 Dataset 2/LA/LA
    base_dir = '/Users/agraw/Desktop/sem 6/trustcall/data/ASVspoof 2019 Dataset 2/LA/LA'
    
    print("Initializing Phase 2 Datasets...")
    train_dataset = ASVSpoofDataset(base_dir, partition='train')
    dev_dataset = ASVSpoofDataset(base_dir, partition='dev')
    
    print(f"Training Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(dev_dataset)}")
    
    # 3. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 4. Model
    model = ResNetBiLSTM(config).to(device)
    
    # 6. Trainer
    trainer = Trainer(model, config, device)
    
    # 7. Run
    print("Starting Phase 2 Training...")
    trainer.run(train_loader, dev_loader)

if __name__ == "__main__":
    main()
