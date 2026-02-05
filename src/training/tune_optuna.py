import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.data.asvspoof_dataset import ASVspoofDataset
from src.models.resnet_bilstm import ResNetBiLSTM
from src.training.trainer import Trainer

def objective(trial):
    # Load base config
    config = load_config("configs/proposed_resnet_bilstm.yaml")
    
    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    lstm_hidden = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    # Update config
    config['training']['learning_rate'] = lr
    config['model']['lstm_hidden_size'] = lstm_hidden
    config['model']['dropout'] = dropout
    config['training']['epochs'] = 5 # Short runs for tuning
    config['training']['output_dir'] = f"outputs/optuna/trial_{trial.number}"
    
    # Data - use smaller subset/batch for speed
    train_ds = ASVspoofDataset(config, subset="train") 
    dev_ds = ASVspoofDataset(config, subset="dev")
    
    # Subsample for speed if needed (not implemented here but good practice)
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=config['training']['batch_size'], shuffle=False)
    
    model = ResNetBiLSTM(config)
    trainer = Trainer(model, config) # Base trainer enough for metric check
    
    # Run 1 epoch or full short training
    # For speed in demo, maybe just 1-2 epochs
    trainer.run(train_loader, dev_loader, epochs=2)
    
    return trainer.best_eer

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5) # 5 trials for demo
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
