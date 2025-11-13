import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
import pickle
from pathlib import Path


class ModelComparator:
    """Compare performance of multiple models."""
    
    def __init__(self):
        """Initialize model comparator."""
        self.results = {}
    
    def add_result(
        self,
        model_name: str,
        metrics: Dict,
        dataset: str = 'test'
    ):
        """
        Add model results.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary with metrics
            dataset: Dataset name ('train', 'val', 'test')
        """
        if model_name not in self.results:
            self.results[model_name] = {}
        
        self.results[model_name][dataset] = metrics
    
    def load_results(self, results_dir: str):
        """
        Load results from directory.
        
        Args:
            results_dir: Directory containing result pickle files
        """
        print(f"Loading results from {results_dir}...")
        
        for file in Path(results_dir).glob("*.pkl"):
            with open(file, 'rb') as f:
                data = pickle.load(f)
                
                # Extract model name from file
                model_name = file.stem
                
                if isinstance(data, dict):
                    for key, value in data.items():
                        self.add_result(f"{model_name}_{key}", value, 'test')
        
        print(f"✓ Loaded results for {len(self.results)} model(s)")
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create comparison table of all models.
        
        Returns:
            DataFrame with comparison
        """
        rows = []
        
        for model_name, datasets in self.results.items():
            for dataset_name, metrics in datasets.items():
                row = {
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1': metrics.get('f1', 0),
                    'ROC-AUC': metrics.get('roc_auc', 0)
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def print_comparison(self):
        """Print comparison table."""
        df = self.create_comparison_table()
        
        print("\n" + "=" * 100)
        print("MODEL COMPARISON")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)
    
    def plot_metric_comparison(
        self,
        metric: str = 'accuracy',
        dataset: str = 'test',
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of specific metric across models.
        
        Args:
            metric: Metric to compare
            dataset: Dataset to compare on
            save_path: Optional path to save figure
        """
        # Extract data
        model_names = []
        values = []
        
        for model_name, datasets in self.results.items():
            if dataset in datasets:
                model_names.append(model_name)
                values.append(datasets[dataset].get(metric, 0))
        
        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(model_names)), values, color='steelblue')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Comparison ({dataset} set)')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_all_metrics(
        self,
        dataset: str = 'test',
        save_path: Optional[str] = None
    ):
        """
        Plot all metrics for all models.
        
        Args:
            dataset: Dataset to compare on
            save_path: Optional path to save figure
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Extract data
        model_names = []
        data = {metric: [] for metric in metrics}
        
        for model_name, datasets in self.results.items():
            if dataset in datasets:
                model_names.append(model_name)
                for metric in metrics:
                    data[metric].append(datasets[dataset].get(metric, 0))
        
        # Create plot
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            bars = ax.bar(range(len(model_names)), data[metric], color='steelblue')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel('Score')
            ax.set_title(metric.upper().replace('_', '-'))
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, data[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.suptitle(f'All Metrics Comparison ({dataset} set)', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def find_best_model(self, metric: str = 'f1', dataset: str = 'test') -> Tuple[str, float]:
        """
        Find best model based on metric.
        
        Args:
            metric: Metric to compare
            dataset: Dataset to compare on
            
        Returns:
            Tuple of (model_name, metric_value)
        """
        best_model = None
        best_value = -1
        
        for model_name, datasets in self.results.items():
            if dataset in datasets:
                value = datasets[dataset].get(metric, 0)
                if value > best_value:
                    best_value = value
                    best_model = model_name
        
        return best_model, best_value
    
    def save_comparison(self, save_path: str):
        """
        Save comparison results to file.
        
        Args:
            save_path: Path to save comparison
        """
        df = self.create_comparison_table()
        
        # Save as CSV
        csv_path = save_path.replace('.pkl', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Comparison saved to {csv_path}")
        
        # Save full results as pickle
        with open(save_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Full results saved to {save_path}")


if __name__ == "__main__":
    # Test model comparator
    print("Testing Model Comparator...")
    
    # Create dummy results
    comparator = ModelComparator()
    
    # Add some models
    models = {
        'LSTM_LR': {
            'accuracy': 0.85, 'precision': 0.84, 'recall': 0.86, 'f1': 0.85, 'roc_auc': 0.88
        },
        'LSTM_RF': {
            'accuracy': 0.87, 'precision': 0.86, 'recall': 0.88, 'f1': 0.87, 'roc_auc': 0.90
        },
        'LSTM_XGB': {
            'accuracy': 0.89, 'precision': 0.88, 'recall': 0.90, 'f1': 0.89, 'roc_auc': 0.92
        },
        'GRU_LR': {
            'accuracy': 0.84, 'precision': 0.83, 'recall': 0.85, 'f1': 0.84, 'roc_auc': 0.87
        },
        'Transformer_XGB': {
            'accuracy': 0.91, 'precision': 0.90, 'recall': 0.92, 'f1': 0.91, 'roc_auc': 0.94
        }
    }
    
    for model_name, metrics in models.items():
        comparator.add_result(model_name, metrics, 'test')
    
    # Print comparison
    comparator.print_comparison()
    
    # Find best model
    print("\n" + "=" * 60)
    best_model, best_f1 = comparator.find_best_model('f1', 'test')
    print(f"Best model (F1): {best_model} with score {best_f1:.4f}")
    
    best_model, best_acc = comparator.find_best_model('accuracy', 'test')
    print(f"Best model (Accuracy): {best_model} with score {best_acc:.4f}")
    
    # Save comparison
    os.makedirs('results/comparisons', exist_ok=True)
    comparator.save_comparison('results/comparisons/test_comparison.pkl')
    
    print("\n✓ All tests passed!")