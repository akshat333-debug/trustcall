import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, det_curve

def plot_roc_curve(y_true, y_scores, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Spoof', 'Bonafide'], yticklabels=['Spoof', 'Bonafide'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_det_curve(y_true, y_scores, save_path=None):
    # DET curve: FPR vs FNR
    # sklearn det_curve returns fpr, fnr, thresholds
    fpr, fnr, thresholds = det_curve(y_true, y_scores, pos_label=1)
    plt.figure()
    plt.plot(fpr, fnr)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('DET Curve')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()
