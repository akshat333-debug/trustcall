import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, accuracy_score
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.utils.config import load_config
from src.models.resnet_bilstm import ResNetBiLSTM
from src.data.asvspoof_dataset import ASVSpoofDataset

def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    
    # Find EER
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[idx]
    return eer, thresholds[idx]

def evaluate():
    # Load Config
    config_path = 'configs/deepvoice_fuzzy.yaml'
    config = load_config(config_path)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Dataset (EVAL Partition)
    # User path: /Users/agraw/Desktop/sem 6/trustcall/data/ASVspoof 2019 Dataset 2/LA/LA
    base_dir = '/Users/agraw/Desktop/sem 6/trustcall/data/ASVspoof 2019 Dataset 2/LA/LA'
    print("Loading ASVspoof 2019 EVAL partition...")
    
    # NOTE: ASVSpoofDataset needs to be checked if it handles 'eval' partition correctly.
    # The folders are ASVspoof2019_LA_eval.
    eval_dataset = ASVSpoofDataset(base_dir, partition='eval')
    print(f"Eval Samples: {len(eval_dataset)}")
    
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load Model
    model = ResNetBiLSTM(config).to(device)
    checkpoint_path = os.path.join(config['training']['output_dir'], "best_model.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Error: Checkpoint not found!")
        return

    model.eval()
    
    all_probs = []
    all_labels = []
    
    print("Running Inference...")
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(eval_loader):
            features, labels = features.to(device), labels.to(device)
            
            # Use forward_explain to get the exact risk score the app uses
            # or just forward() since we fixed it to return clamped risk in logits?
            # Creating logits from risk: 
            # forward() returns logits. prob_spoof = softmax(logits)[1] should == risk
            
            logits, risk_score, _, _ = model.forward_explain(features)
             
            if risk_score is not None:
                probs = risk_score.squeeze().cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 50 == 0:
                print(f"Processed batch {batch_idx}/{len(eval_loader)}")
                
    # Metrics
    y_true = np.array(all_labels)
    y_score = np.array(all_probs)
    
    # EER
    eer, threshold = compute_eer(y_true, y_score)
    
    # AUC
    roc_auc = auc(*roc_curve(y_true, y_score, pos_label=1)[:2])
    
    # Accuracy at 0.5 threshold
    y_pred = (y_score > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*30)
    print("PHASE 2 COMPREHENSIVE TEST RESULTS")
    print("="*30)
    print(f"Test Set: ASVspoof 2019 LA Eval ({len(y_true)} samples)")
    print(f"EER: {eer:.4%}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"Accuracy: {acc:.4%}")
    print("="*30)

if __name__ == "__main__":
    evaluate()
