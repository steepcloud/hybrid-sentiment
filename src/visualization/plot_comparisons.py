"""
Model Comparison Visualization
Compare performance across different models and datasets.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path


class ComparisonVisualizer:
    """Visualize model comparisons."""
    
    def __init__(self, figsize=(14, 8)):
        self.figsize = figsize
        sns.set_style("whitegrid")
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = ['accuracy', 'f1', 'precision', 'recall'],
        save_path: Optional[str] = None
    ):
        """
        Compare multiple models across metrics.
        
        Args:
            results: Dict of {model_name: {metric: score}}
            metrics: Metrics to compare
            save_path: Path to save figure
            
        Example:
            results = {
                'LSTM': {'accuracy': 0.89, 'f1': 0.88, 'precision': 0.90, 'recall': 0.87},
                'GRU': {'accuracy': 0.87, 'f1': 0.86, 'precision': 0.88, 'recall': 0.85},
                'Transformer': {'accuracy': 0.91, 'f1': 0.90, 'precision': 0.92, 'recall': 0.89}
            }
        """
        # Convert to DataFrame
        df = pd.DataFrame(results).T
        df = df[metrics]  # Select only specified metrics
        
        # Plot grouped bar chart
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(df.index))
        width = 0.8 / len(metrics)
        
        for i, metric in enumerate(metrics):
            offset = (i - len(metrics)/2) * width + width/2
            bars = ax.bar(
                x + offset,
                df[metric],
                width,
                label=metric.capitalize(),
                alpha=0.8
            )
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    def plot_training_time_comparison(
        self,
        training_times: Dict[str, float],
        save_path: Optional[str] = None
    ):
        """
        Compare training times across models.
        
        Args:
            training_times: Dict of {model_name: time_in_seconds}
            save_path: Path to save figure
        """
        models = list(training_times.keys())
        times = list(training_times.values())
        
        # Convert to minutes
        times_min = [t / 60 for t in times]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(models, times_min, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
        
        # Add value labels
        for bar, time in zip(bars, times_min):
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height()/2,
                f'{time:.1f} min',
                ha='left',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_xlabel('Training Time (minutes)', fontsize=12)
        ax.set_title('Model Training Time Comparison', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    def plot_model_size_comparison(
        self,
        model_sizes: Dict[str, float],
        save_path: Optional[str] = None
    ):
        """
        Compare model sizes.
        
        Args:
            model_sizes: Dict of {model_name: size_in_mb}
            save_path: Path to save figure
        """
        models = list(model_sizes.keys())
        sizes = list(model_sizes.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(models, sizes, color=plt.cm.plasma(np.linspace(0, 1, len(models))))
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f'{height:.1f} MB',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_ylabel('Model Size (MB)', fontsize=12)
        ax.set_title('Model Size Comparison', fontsize=14)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    def plot_comprehensive_comparison(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive comparison dashboard.
        
        Args:
            results: Dict of model results with metrics, time, size
            save_path: Path to save figure
            
        Example:
            results = {
                'LSTM': {
                    'accuracy': 0.89,
                    'f1': 0.88,
                    'training_time': 1200,  # seconds
                    'model_size': 45.3      # MB
                }
            }
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy comparison
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        
        axes[0, 0].bar(models, accuracies, color='skyblue')
        axes[0, 0].set_title('Accuracy Comparison', fontsize=12)
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        for i, (m, acc) in enumerate(zip(models, accuracies)):
            axes[0, 0].text(i, acc, f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. F1 Score comparison
        f1_scores = [results[m]['f1'] for m in models]
        
        axes[0, 1].bar(models, f1_scores, color='lightgreen')
        axes[0, 1].set_title('F1 Score Comparison', fontsize=12)
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        for i, (m, f1) in enumerate(zip(models, f1_scores)):
            axes[0, 1].text(i, f1, f'{f1:.3f}', ha='center', va='bottom')
        
        # 3. Training time
        times = [results[m]['training_time'] / 60 for m in models]  # Convert to minutes
        
        axes[1, 0].barh(models, times, color='salmon')
        axes[1, 0].set_title('Training Time', fontsize=12)
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        for i, (m, t) in enumerate(zip(models, times)):
            axes[1, 0].text(t, i, f'{t:.1f}m', va='center', ha='left')
        
        # 4. Model size
        sizes = [results[m]['model_size'] for m in models]
        
        axes[1, 1].bar(models, sizes, color='mediumpurple')
        axes[1, 1].set_title('Model Size', fontsize=12)
        axes[1, 1].set_ylabel('Size (MB)')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        for i, (m, s) in enumerate(zip(models, sizes)):
            axes[1, 1].text(i, s, f'{s:.1f}MB', ha='center', va='bottom')
        
        plt.suptitle('Comprehensive Model Comparison', fontsize=16, y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Test with dummy data
    results = {
        'LSTM': {
            'accuracy': 0.89,
            'f1': 0.88,
            'precision': 0.90,
            'recall': 0.87,
            'training_time': 1200,
            'model_size': 45.3
        },
        'GRU': {
            'accuracy': 0.87,
            'f1': 0.86,
            'precision': 0.88,
            'recall': 0.85,
            'training_time': 1100,
            'model_size': 42.1
        },
        'Transformer': {
            'accuracy': 0.91,
            'f1': 0.90,
            'precision': 0.92,
            'recall': 0.89,
            'training_time': 1800,
            'model_size': 78.5
        }
    }
    
    viz = ComparisonVisualizer()
    
    print("Testing model comparison...")
    viz.plot_model_comparison(results)
    
    print("\nTesting comprehensive comparison...")
    viz.plot_comprehensive_comparison(results)