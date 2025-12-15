"""
Confusion Matrix and Classification Report Visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from typing import Optional, List
import pandas as pd


class ConfusionVisualizer:
    """Visualize confusion matrices and classification metrics."""
    
    def __init__(self, figsize=(10, 8)):
        self.figsize = figsize
        sns.set_style("whitegrid")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str] = ['Negative', 'Positive'],
        normalize: bool = False,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class names
            normalize: If True, normalize values
            save_path: Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    def plot_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str] = ['Negative', 'Positive'],
        save_path: Optional[str] = None
    ):
        """
        Visualize classification report as heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class names
            save_path: Path to save figure
        """
        # get classification report as dict
        report = classification_report(
            y_true,
            y_pred,
            target_names=labels,
            output_dict=True
        )
        
        # convert to DataFrame
        df = pd.DataFrame(report).transpose()
        
        # remove support column for cleaner visualization
        df_viz = df.drop('support', axis=1).iloc[:-3]  # remove avg rows
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            df_viz,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            cbar_kws={'label': 'Score'},
            ax=ax
        )
        
        ax.set_title('Classification Report', fontsize=14)
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Classes', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()

        print("\nDetailed Classification Report:")
        print("=" * 50)
        print(classification_report(y_true, y_pred, target_names=labels))


if __name__ == "__main__":
    # test with dummy data
    np.random.seed(42)
    
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.randint(0, 2, 1000)
    
    viz = ConfusionVisualizer()
    
    print("Testing Confusion Matrix...")
    viz.plot_confusion_matrix(y_true, y_pred)
    
    print("\nTesting Classification Report...")
    viz.plot_classification_report(y_true, y_pred)