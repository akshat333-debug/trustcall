import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from src.utils.config import load_config
from src.data.asvspoof_dataset import ASVSpoofDataset
from src.models.resnet_bilstm import ResNetBiLSTM
from src.training.trainer import Trainer
from src.utils.seed import seed_everything

def train_phase3():
    # Load Robust Config
    config = load_config('configs/robust_training.yaml')
    seed_everything(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset with Augmentation=True
    # path: data/ASVspoof 2019 Dataset 2/LA/LA
    base_dir = '/Users/agraw/Desktop/sem 6/trustcall/data/ASVspoof 2019 Dataset 2/LA/LA' 
    print("Loading ASVspoof 2019 Train (With Augmentation)...")
    train_dataset = ASVSpoofDataset(base_dir, partition='train', augment=True)
    
    print("Loading ASVspoof 2019 Dev (No Augmentation)...")
    # We validate on CLEAN dev set to ensure we don't regress on standard metric
    dev_dataset = ASVSpoofDataset(base_dir, partition='dev', augment=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=0 # MPS stability
    )
    
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=0 # MPS stability
    )

    # Initialize Model
    model = ResNetBiLSTM(config).to(device)
    
    # Load Phase 2 Best Weights
    # The logic: We start from the super-accurate "Clean" model and fine-tune it
    phase2_checkpoint = "outputs/deepvoice_fuzzy/best_model.pth"
    
    if os.path.exists(phase2_checkpoint):
        print(f"Loading Phase 2 weights from {phase2_checkpoint} for fine-tuning...")
        model.load_state_dict(torch.load(phase2_checkpoint, map_location=device))
    else:
        print(f"Warning: Phase 2 checkpoint not found at {phase2_checkpoint}. Starting from scratch (not recommended).")

    # Trainer
    trainer = Trainer(model, config, device)
    
    # Start Fine-Tuning
    print("Starting Phase 3: Robustness Fine-Tuning...")
    trainer.run(train_loader, dev_loader, epochs=config['training']['epochs'])

if __name__ == "__main__":
    train_phase3()
