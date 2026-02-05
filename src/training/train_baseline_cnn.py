import torch
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.data.asvspoof_dataset import ASVspoofDataset
from src.models.baseline_cnn import BaselineCNN
from src.training.trainer import Trainer

def main():
    config = load_config("configs/baseline_cnn.yaml")
    seed_everything(42)
    
    # Dataset
    train_ds = ASVspoofDataset(config, subset="train")
    dev_ds = ASVspoofDataset(config, subset="dev")
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batck_size'], shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_ds, batch_size=config['training']['batck_size'], shuffle=False, num_workers=2)
    
    # Model
    model = BaselineCNN(config)
    
    # Trainer
    trainer = Trainer(model, config)
    trainer.run(train_loader, dev_loader)

if __name__ == "__main__":
    main()
