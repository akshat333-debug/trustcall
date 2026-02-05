import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

from src.utils.logger import setup_logger
from src.eval.metrics import compute_metrics

class Trainer:
    def __init__(self, model, config, device=None):
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.logger = setup_logger(f"Trainer_{config['model']['name']}", log_dir=config['training']['output_dir'])
        
        self.criterion = nn.CrossEntropyLoss() # or BCEWithLogitsLoss if output is 1 dim
        # Our models output (B, 2), so CrossEntropy is good.
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay'])
        )
        
        self.best_eer = 1.0
        self.patience_counter = 0

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in tqdm(loader, desc="Training"):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        return total_loss / len(loader), correct / total

    def evaluate(self, loader):
        self.model.eval()
        all_labels = []
        all_probs = [] # Prob of bonafide (class 1)
        total_loss = 0
        
        with torch.no_grad():
            for features, labels in tqdm(loader, desc="Evaluating"):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Softmax
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
        return total_loss / len(loader), metrics

    def run(self, train_loader, dev_loader, epochs=None):
        epochs = epochs or self.config['training']['epochs']
        output_dir = self.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(dev_loader)
            
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            self.logger.info(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f} | EER: {val_metrics['eer']:.4f} | AUC: {val_metrics['auc']:.4f}")
            
            # Checkpoint
            if val_metrics['eer'] < self.best_eer:
                self.best_eer = val_metrics['eer']
                self.patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                self.logger.info("Saved best model.")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config['training'].get('early_stopping_patience', 10):
                self.logger.info("Early stopping triggered.")
                break
                
        self.logger.info(f"Best EER: {self.best_eer}")
