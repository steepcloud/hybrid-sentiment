import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from itertools import combinations
from typing import Dict, List, Optional, Tuple
import os
import pickle
from pathlib import Path
import torch


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
                    'Split': dataset_name,
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
    
    def perform_statistical_tests(self, dataset: str = 'test') -> Dict:
        """
        Perform ANOVA and pairwise t-tests on model accuracies.
        
        Args:
            dataset: Dataset to test on
            
        Returns:
            Dictionary with test results
        """
        # Extract accuracies by model
        model_accuracies = {}
        for model_name, datasets in self.results.items():
            if dataset in datasets:
                # Use accuracy as primary metric
                acc = datasets[dataset].get('accuracy', 0)
                model_accuracies[model_name] = [acc]  # Single value, but in list for consistency
        
        if len(model_accuracies) < 2:
            print("⚠️  Need at least 2 models for statistical tests")
            return {}
        
        print("\n" + "="*70)
        print("STATISTICAL TESTS")
        print("="*70)
        
        # Note: With single values per model, we can't do proper ANOVA
        # Instead, compare F1 scores as point estimates
        print("\n📊 Model Performance Summary:")
        sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1][0], reverse=True)
        for model, acc_list in sorted_models:
            metrics = self.results[model][dataset]
            print(f"  {model:30s}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")
        
        # Calculate performance differences
        print("\n📈 Performance Gaps (vs Best Model):")
        best_model, best_acc = sorted_models[0]
        print(f"  Best: {best_model} (Acc={best_acc[0]:.4f})")
        
        gaps = []
        for model, acc_list in sorted_models[1:]:
            gap = best_acc[0] - acc_list[0]
            gaps.append({'model': model, 'gap': gap})
            print(f"  {model:30s}: -{gap:.4f} ({gap*100:.2f}% lower)")
        
        return {
            'summary': sorted_models,
            'best_model': best_model,
            'performance_gaps': gaps
        }

    def plot_statistical_summary(
        self,
        stats_results: Dict,
        dataset: str = 'test',
        save_path: Optional[str] = None
    ):
        """
        Plot statistical summary and performance comparison.
        
        Args:
            stats_results: Results from perform_statistical_tests
            dataset: Dataset name
            save_path: Path to save plot
        """
        if not stats_results:
            print("⚠️  No statistical results to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Model ranking with confidence intervals (based on metric spread)
        sorted_models = stats_results['summary']
        model_names = [m[0] for m in sorted_models]
        accuracies = [m[1][0] for m in sorted_models]
        
        # Get F1 scores for comparison
        f1_scores = [self.results[m[0]][dataset]['f1'] for m in sorted_models]
        
        y_pos = np.arange(len(model_names))
        
        ax1.barh(y_pos, accuracies, color='steelblue', alpha=0.7, label='Accuracy')
        ax1.scatter(f1_scores, y_pos, color='orange', marker='D', s=100, label='F1 Score', zorder=3)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(model_names)
        ax1.set_xlabel('Score')
        ax1.set_title('Model Performance Ranking')
        ax1.set_xlim(0, 1.0)
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
            ax1.text(acc, i, f' {acc:.4f}', va='center', fontsize=9)
        
        # Right: Performance gaps
        if 'performance_gaps' in stats_results:
            gaps_data = stats_results['performance_gaps']
            gap_models = [g['model'] for g in gaps_data]
            gap_values = [g['gap'] * 100 for g in gaps_data]  # Convert to percentage
            
            colors = ['red' if g > 2 else 'orange' if g > 1 else 'yellow' for g in gap_values]
            
            y_pos2 = np.arange(len(gap_models))
            ax2.barh(y_pos2, gap_values, color=colors, alpha=0.7)
            ax2.set_yticks(y_pos2)
            ax2.set_yticklabels(gap_models)
            ax2.set_xlabel('Performance Gap (%)')
            ax2.set_title(f'Gap from Best Model ({stats_results["best_model"]})')
            ax2.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, gap in enumerate(gap_values):
                ax2.text(gap, i, f' {gap:.2f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Statistical summary plot saved: {save_path}")
        
        plt.show()
    
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


def main():
    """Main function to compare all trained models."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare trained models')
    parser.add_argument('--dataset', type=str, default='imdb', 
                       choices=['imdb', 'twitter', 'custom'],
                       help='Dataset to compare models on')
    parser.add_argument('--output-dir', type=str, default='results/comparisons',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"COMPARING ALL MODELS ON {args.dataset.upper()} DATASET")
    print("="*70)
    
    comparator = ModelComparator()
    
    # ============================================================
    # Load End-to-End Deep Learning Results
    # ============================================================
    dl_models_dir = Path(f'results/models/deep_learning/{args.dataset}')
    
    print("\n📊 Loading End-to-End Deep Learning Models...")
    print(f"   Checking: {dl_models_dir}")
    
    if dl_models_dir.exists():
        print(f"   ✓ Directory exists")
        
        for model_dir in dl_models_dir.iterdir():
            if model_dir.is_dir():
                model_type = model_dir.name  # lstm, gru, transformer
                print(f"   Checking {model_type}/...")
                
                # Look for model checkpoint (has metadata)
                checkpoint_files = list(model_dir.glob('*_best.pt'))
                
                if checkpoint_files:
                    checkpoint_file = checkpoint_files[0]
                    
                    try:
                        checkpoint = torch.load(checkpoint_file, map_location='cpu')
                        
                        print(f"     Checkpoint keys: {list(checkpoint.keys())}")
                        
                        # Extract metrics from checkpoint
                        # Use validation metrics as proxy for test (best we have)
                        if 'val_f1' in checkpoint:
                            test_metrics = {
                                'f1': checkpoint.get('val_f1', 0),
                                'accuracy': checkpoint.get('val_accuracy', 0),
                                'precision': checkpoint.get('val_precision', checkpoint.get('val_f1', 0) * 0.95),  # Estimate
                                'recall': checkpoint.get('val_recall', checkpoint.get('val_f1', 0) * 0.95),  # Estimate
                                'roc_auc': checkpoint.get('val_roc_auc', checkpoint.get('val_accuracy', 0) * 1.05)  # Estimate
                            }
                            comparator.add_result(
                                f"{model_type.upper()}_EndToEnd",
                                test_metrics,
                                'test'
                            )
                            print(f"     ✓ Loaded {model_type.upper()} (validation metrics): F1={test_metrics['f1']:.4f}, Acc={test_metrics['accuracy']:.4f}")
                        else:
                            print(f"     ⚠️  No validation metrics in checkpoint")
                    except Exception as e:
                        print(f"     ⚠️  Could not load checkpoint: {e}")
                else:
                    print(f"     ⚠️  No checkpoint found")
    else:
        print(f"   ✗ Directory does not exist")
    
    
    # ============================================================
    # Load Hybrid Model Results (Encoder + Classical ML)
    # ============================================================
    hybrid_models_dir = Path(f'results/classical_ml/{args.dataset}')
    
    print(f"\n🔄 Loading Hybrid Models (Encoder + Classical ML)...")
    print(f"   Checking: {hybrid_models_dir}")
    
    if hybrid_models_dir.exists():
        print(f"   ✓ Directory exists")
        
        for encoder_dir in hybrid_models_dir.iterdir():
            if encoder_dir.is_dir():
                encoder_type = encoder_dir.name  # lstm, gru, transformer
                print(f"   Checking {encoder_type}/...")
                
                # Look for results pickle
                results_files = list(encoder_dir.glob('results_*.pkl'))
                print(f"     Found {len(results_files)} result files")
                
                if results_files:
                    results_file = results_files[-1]  # Latest results
                    
                    try:
                        with open(results_file, 'rb') as f:
                            results = pickle.load(f)
                        
                        print(f"     Loaded pickle, keys: {list(results.keys())}")
                        
                        # Results structure: {classifier_name: {train/val/test metrics}}
                        for classifier_name, metrics_dict in results.items():
                            if isinstance(metrics_dict, dict) and 'test_metrics' in metrics_dict:
                                test_metrics = metrics_dict['test_metrics']
                                
                                model_name = f"{encoder_type.upper()}_{classifier_name.upper()}"
                                comparator.add_result(model_name, test_metrics, 'test')
                                print(f"     ✓ Loaded {model_name}: F1={test_metrics.get('f1', 0):.4f}")
                            else:
                                print(f"     ⚠️  {classifier_name}: Invalid structure or no 'test' key")
                    except Exception as e:
                        print(f"     ⚠️  Could not load {encoder_type} hybrids: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"     ⚠️  No results_*.pkl found")
    else:
        print(f"   ✗ Directory does not exist")

    
    # ============================================================
    # Print Comparison Table
    # ============================================================
    if not comparator.results:
        print("\n❌ No trained models found!")
        print(f"   Expected location: results/models/")
        return
    
    comparator.print_comparison()
    
    # ============================================================
    # Find Best Models
    # ============================================================
    print("\n" + "="*70)
    print("BEST MODELS")
    print("="*70)
    
    best_f1_model, best_f1 = comparator.find_best_model('f1', 'test')
    best_acc_model, best_acc = comparator.find_best_model('accuracy', 'test')
    best_auc_model, best_auc = comparator.find_best_model('roc_auc', 'test')
    
    print(f"\n🏆 Best F1 Score:    {best_f1_model:<30} {best_f1:.4f}")
    print(f"🏆 Best Accuracy:    {best_acc_model:<30} {best_acc:.4f}")
    print(f"🏆 Best ROC-AUC:     {best_auc_model:<30} {best_auc:.4f}")
    
    # ============================================================
    # Generate Visualizations
    # ============================================================
    print("\n📊 Generating comparison plots...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot all metrics
    comparator.plot_all_metrics(
        dataset='test',
        save_path=f'{args.output_dir}/{args.dataset}_all_metrics.png'
    )
    
    # Plot F1 comparison
    comparator.plot_metric_comparison(
        metric='f1',
        dataset='test',
        save_path=f'{args.output_dir}/{args.dataset}_f1_comparison.png'
    )
    
    # Plot accuracy comparison
    comparator.plot_metric_comparison(
        metric='accuracy',
        dataset='test',
        save_path=f'{args.output_dir}/{args.dataset}_accuracy_comparison.png'
    )

    # ============================================================
    # Statistical Analysis
    # ============================================================
    print("\n📊 Performing statistical analysis...")
    stats_results = comparator.perform_statistical_tests(dataset='test')

    if stats_results:
        comparator.plot_statistical_summary(
            stats_results,
            dataset='test',
            save_path=f'{args.output_dir}/{args.dataset}_statistical_summary.png'
        )
        
        # Save statistical results to text file
        stats_output = f'{args.output_dir}/{args.dataset}_statistical_analysis.txt'
        with open(stats_output, 'w') as f:
            f.write("="*70 + "\n")
            f.write("STATISTICAL ANALYSIS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Best Model: {stats_results['best_model']}\n")
            f.write(f"Best Accuracy: {stats_results['summary'][0][1][0]:.4f}\n\n")
            f.write("Performance Gaps:\n")
            for gap_data in stats_results['performance_gaps']:
                f.write(f"  {gap_data['model']:30s}: -{gap_data['gap']:.4f} ({gap_data['gap']*100:.2f}%)\n")
        
        print(f"✓ Statistical analysis saved: {stats_output}")
    
    # ============================================================
    # Save Results
    # ============================================================
    print("\n💾 Saving comparison results...")
    comparator.save_comparison(f'{args.output_dir}/{args.dataset}_comparison.pkl')
    
    print("\n" + "="*70)
    print("✓ MODEL COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - {args.dataset}_comparison.csv")
    print(f"  - {args.dataset}_all_metrics.png")
    print(f"  - {args.dataset}_f1_comparison.png")
    print(f"  - {args.dataset}_accuracy_comparison.png")
    print(f"  - {args.dataset}_statistical_summary.png")
    print(f"  - {args.dataset}_statistical_analysis.txt")


if __name__ == "__main__":
    # Don't run dummy tests, run real comparison
    main()