import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_fscore_support
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def compute_eer(y_true, y_score):
    """
    Compute Equal Error Rate (EER).
    y_true: 0 (spoof) or 1 (bonafide)
    y_score: probability of being bonafide (target) or score
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    
    # EER: where FPR = FNR = 1 - TPR
    # fnr = 1 - tpr
    # We look for fpr - fnr = 0
    
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # thresh = interp1d(fpr, thresholds)(eer)
    return eer

def compute_metrics(y_true, y_score, threshold=0.5):
    """
    Compute full suite of metrics.
    """
    y_pred = (np.array(y_score) >= threshold).astype(int)
    
    # Precision, Recall, F1
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
    
    # AUC
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    # EER
    try:
        eer = compute_eer(y_true, y_score)
    except:
        eer = 0.0 # Fallback
        
    return {
        "accuracy": np.mean(y_true == y_pred),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": roc_auc,
        "eer": eer
    }
