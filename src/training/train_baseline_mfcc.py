import os
import pickle
import numpy as np
import yaml
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.protocols import parse_protocol, get_file_list
from src.eval.metrics import compute_metrics, compute_eer

# Simple feature extractor for sklearn models using librosa
def extract_mfcc_stats(path, n_mfcc=40):
    try:
        wav, sr = librosa.load(path, sr=None, mono=True)
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)
        
        # Stats: Mean + Std
        mean = np.mean(mfcc, axis=1)
        std = np.std(mfcc, axis=1)
        return np.concatenate([mean, std])
    except Exception as e:
        return np.zeros(n_mfcc * 2)

def main():
    config = load_config("configs/baseline_mfcc.yaml")
    logger = setup_logger("Baseline_MFCC", log_dir=config['training']['output_dir'])
    
    root_dir = config['data']['root_dir']
    track = config['data']['track']
    
    # 1. Load Data Lists
    logger.info("Loading protocols...")
    train_df = parse_protocol(os.path.join(root_dir, track, f"ASVspoof2019_{track}_cm_protocols/ASVspoof2019.{track}.cm.train.trn.txt"))
    train_paths, train_labels, _ = get_file_list(train_df, root_dir, track, "train")
    
    dev_df = parse_protocol(os.path.join(root_dir, track, f"ASVspoof2019_{track}_cm_protocols/ASVspoof2019.{track}.cm.dev.trl.txt"))
    dev_paths, dev_labels, _ = get_file_list(dev_df, root_dir, track, "dev")
    
    # 2. Extract Features
    logger.info(f"Extracting features for {len(train_paths)} training files...")
    X_train = []
    for p in tqdm(train_paths[:1000]): # Limit for demo/speed -> Remove slice for full training!
        X_train.append(extract_mfcc_stats(p))
    X_train = np.stack(X_train)
    y_train = np.array(train_labels[:1000])
    
    logger.info(f"Extracting features for {len(dev_paths)} dev files...")
    X_dev = []
    for p in tqdm(dev_paths[:200]): # Limit for demo
        X_dev.append(extract_mfcc_stats(p))
    X_dev = np.stack(X_dev)
    y_dev = np.array(dev_labels[:200])
    
    # 3. Train
    logger.info("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    y_pred = clf.predict_proba(X_dev)[:, 1] # Prob of class 1 (Bonafide)
    
    metrics = compute_metrics(y_dev, y_pred)
    logger.info(f"Dev Metrics: {metrics}")
    
    # Save model
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    with open(os.path.join(config['training']['output_dir'], "rf_model.pkl"), "wb") as f:
        pickle.dump(clf, f)
        
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
