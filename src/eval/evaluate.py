import torch
import argparse
import yaml
import os
import numpy as np
from torch.utils.data import DataLoader
from src.data.asvspoof_dataset import ASVspoofDataset
from src.data.deepvoice_dataset import DeepVoiceDataset
from src.models.resnet_bilstm import ResNetBiLSTM
from src.models.baseline_cnn import BaselineCNN
from src.eval.metrics import compute_metrics
from src.utils.seed import seed_everything
from src.eval.plots import plot_roc_curve, plot_confusion_matrix, plot_det_curve

def load_model(model_name, config, checkpoint_path):
    if model_name == 'baseline_cnn':
        model = BaselineCNN(config)
    elif model_name == 'resnet_bilstm':
        model = ResNetBiLSTM(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--subset", type=str, default="eval")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    seed_everything(42)
    
    # Dataset selection
    dataset_type = config.get('data', {}).get('dataset_type', 'asvspoof')
    if dataset_type == 'deepvoice':
        print(f"Using DeepVoiceDataset ({args.subset})...")
        ds = DeepVoiceDataset(config, subset=args.subset)
    else:
        ds = ASVspoofDataset(config, subset=args.subset)
        
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    model = load_model(config['model']['name'], config, args.model_path)
    
    y_true = []
    y_scores = []
    
    print("Running inference...")
    with torch.no_grad():
        for features, label in loader:
            output = model(features)
            prob = torch.softmax(output, dim=1)[:, 1].item()
            y_true.append(label.item())
            y_scores.append(prob)
            
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    metrics = compute_metrics(y_true, y_scores)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
        
    # Plots
    os.makedirs("outputs/plots", exist_ok=True)
    plot_roc_curve(y_true, y_scores, save_path="outputs/plots/roc.png")
    plot_det_curve(y_true, y_scores, save_path="outputs/plots/det.png")
    # For CM, we need binary preds
    y_pred = (y_scores >= 0.5).astype(int)
    plot_confusion_matrix(y_true, y_pred, save_path="outputs/plots/cm.png")

if __name__ == "__main__":
    main()
