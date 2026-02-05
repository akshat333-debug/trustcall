import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.data.asvspoof_dataset import ASVspoofDataset
from src.models.resnet_bilstm import ResNetBiLSTM
from src.training.trainer import Trainer

# Optional: Custom Trainer if we want to train Fuzzy layer end-to-end
# For proposed model (Deep Learning only part), standard Trainer is fine.
# The Fuzzy layer is designed to be an interpretation head or post-processor.
# If we want to train fuzzy parameters, we need a joint model.

class ProposedTrainer(Trainer):
    def __init__(self, model, config, device=None):
        super().__init__(model, config, device)
        # Any custom logic for proposed model e.g. schedulers
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)

    def run(self, train_loader, dev_loader, epochs=None):
        # Override run to include scheduler step
        epochs = epochs or self.config['training']['epochs']
        output_dir = self.config['training']['output_dir']
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(dev_loader)
            
            self.logger.info(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val EER: {val_metrics['eer']:.4f}")
            
            self.scheduler.step(val_loss)
            
            if val_metrics['eer'] < self.best_eer:
                self.best_eer = val_metrics['eer']
                self.patience_counter = 0
                torch.save(self.model.state_dict(), f"{output_dir}/best_model.pth")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                break

import argparse
from src.data.deepvoice_dataset import DeepVoiceDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/proposed_resnet_bilstm.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(42)
    
    # Select dataset based on config
    dataset_type = config.get('data', {}).get('dataset_type', 'asvspoof')
    
    if dataset_type == 'deepvoice':
        print("Using DeepVoiceDataset (Kaggle)...")
        train_ds = DeepVoiceDataset(config, subset="train")
        dev_ds = DeepVoiceDataset(config, subset="dev")
    else:
        print("Using ASVspoofDataset...")
        train_ds = ASVspoofDataset(config, subset="train")
        dev_ds = ASVspoofDataset(config, subset="dev")
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0) # reduced workers for safety
    dev_loader = DataLoader(dev_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)
    
    model = ResNetBiLSTM(config)
    
    trainer = ProposedTrainer(model, config)
    trainer.run(train_loader, dev_loader)

if __name__ == "__main__":
    main()
