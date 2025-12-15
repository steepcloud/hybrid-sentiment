"""
Training Metrics Visualization
Plot training curves, loss, accuracy, F1 scores over epochs.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional


class MetricsVisualizer:
    """Visualize training metrics and performance."""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        sns.set_style("whitegrid")
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """
        Plot training and validation curves.
        
        Args:
            history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc', etc.
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = range(1, len(history.get('train_loss', [])) + 1)
        
        # loss
        if 'train_loss' in history:
            axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss Over Epochs', fontsize=12)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # accuracy
        if 'train_acc' in history:
            axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        if 'val_acc' in history:
            axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].set_title('Accuracy Over Epochs', fontsize=12)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        if 'train_f1' in history:
            axes[1, 0].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
        if 'val_f1' in history:
            axes[1, 0].plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
        axes[1, 0].set_title('F1 Score Over Epochs', fontsize=12)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # learning Rate (if available)
        if 'learning_rate' in history:
            axes[1, 1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
            axes[1, 1].set_title('Learning Rate Schedule', fontsize=12)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()