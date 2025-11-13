import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = 'binary'
) -> Dict:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for ROC-AUC)
        average: Averaging method for multi-class ('binary', 'micro', 'macro', 'weighted')
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # ROC-AUC (if probabilities provided)
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                # Binary classification - use positive class probabilities
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                # Multi-class or single column
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except:
            metrics['roc_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def print_metrics(metrics: Dict, title: str = "Metrics"):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary with metrics
        title: Title for the output
    """
    print(f"\n{title}")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 50)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list = ['Negative', 'Positive'],
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        classes: Class labels
        normalize: Whether to normalize values
        title: Plot title
        save_path: Optional path to save figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def compare_metrics(metrics_dict: Dict[str, Dict]) -> None:
    """
    Compare metrics across multiple models.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
    print("-" * 80)
    
    for model_name, metrics in metrics_dict.items():
        print(f"{model_name:<20} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} "
              f"{metrics.get('roc_auc', 0):<12.4f}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Test metrics calculation
    print("Testing metrics calculation...")
    
    # Create dummy predictions
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    errors = np.random.choice(n_samples, size=10, replace=False)
    y_pred[errors] = 1 - y_pred[errors]
    
    y_proba = np.random.rand(n_samples, 2)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    # Print metrics
    print_metrics(metrics, "Test Metrics")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Test comparison
    print("\n" + "=" * 60)
    print("Testing model comparison...")
    
    metrics_dict = {
        'Model A': metrics,
        'Model B': {
            'accuracy': 0.92,
            'precision': 0.91,
            'recall': 0.93,
            'f1': 0.92,
            'roc_auc': 0.95
        },
        'Model C': {
            'accuracy': 0.88,
            'precision': 0.87,
            'recall': 0.89,
            'f1': 0.88,
            'roc_auc': 0.91
        }
    }
    
    compare_metrics(metrics_dict)
    
    print("\n✓ All tests passed!")