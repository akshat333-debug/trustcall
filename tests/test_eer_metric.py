import pytest
import numpy as np
from src.eval.metrics import compute_metrics, compute_eer

def test_perfect_prediction():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.9, 0.8])
    
    metrics = compute_metrics(y_true, y_score)
    assert metrics['accuracy'] == 1.0
    assert metrics['eer'] == 0.0

def test_eer_calculation():
    # Simple case
    y_true = np.array([0, 1])
    y_score = np.array([0.4, 0.6])
    # Threshold 0.5 works perfectly
    eer = compute_eer(y_true, y_score)
    assert eer == 0.0
    
    # Error case
    y_true = np.array([0, 1])
    y_score = np.array([0.6, 0.4]) # Swapped
    # EER should be high (likely 1.0 or 0.5 depending on calculation around x=y)
    # roc_curve might return specific points
    eer = compute_eer(y_true, y_score)
    assert eer > 0.0
