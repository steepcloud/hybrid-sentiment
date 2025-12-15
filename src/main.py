"""
Hybrid Sentiment Analysis - Main CLI Interface
Unified command-line interface for training, evaluation, and inference.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


def train_embeddings(args):
    """Train Word2Vec embeddings."""
    from src.training.train_embeddings import main as train_emb_main
    
    print("\n" + "="*60)
    print("TRAINING WORD2VEC EMBEDDINGS")
    print("="*60)
    
    # convert args to list for train_embeddings
    emb_args = [
        '--dataset', args.dataset,
        '--embedding', args.embedding_type,
    ]
    
    if args.vector_size:
        emb_args.extend(['--vector_size', str(args.vector_size)])
    if args.window:
        emb_args.extend(['--window', str(args.window)])
    if args.min_count:
        emb_args.extend(['--min_count', str(args.min_count)])
    
    sys.argv = ['train_embeddings.py'] + emb_args
    train_emb_main()


def train_deep_learning(args):
    """Train end-to-end deep learning model."""
    from src.training.train_end_to_end_dl import main as train_dl_main
    
    print("\n" + "="*60)
    print(f"TRAINING {args.model.upper()} MODEL (END-TO-END)")
    print("="*60)
    
    dl_args = [
        '--dataset', args.dataset,
        '--model', args.model,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr)
    ]
    
    sys.argv = ['train_end_to_end_dl.py'] + dl_args
    train_dl_main()


def train_classical_ml(args):
    """Train hybrid model (encoder + classical ML)."""
    from src.training.train_classical_ml import main as train_classical_main
    
    print("\n" + "="*60)
    print(f"TRAINING HYBRID MODEL ({args.encoder.upper()} + Classical ML)")
    print("="*60)
    
    classical_args = [
        '--dataset', args.dataset,
        '--encoder', args.encoder
    ]
    
    sys.argv = ['train_classical_ml.py'] + classical_args
    train_classical_main()


def train_all(args):
    """Train all models for comprehensive comparison."""
    print("\n" + "="*70)
    print("TRAINING ALL MODELS - COMPREHENSIVE PIPELINE")
    print("="*70)
    
    models = ['lstm', 'gru', 'transformer']
    if args.include_bert:
        models.extend(['bert', 'roberta', 'distilbert'])
    
    # 1. Train Word2Vec embeddings
    print("\n[1/4] Training Word2Vec embeddings...")
    args.embedding_type = 'word2vec'
    train_embeddings(args)
    
    # 2. Train all end-to-end deep learning models
    print("\n[2/4] Training end-to-end deep learning models...")
    for model in models:
        args.model = model
        train_deep_learning(args)
    
    # 3. Train all hybrid models
    print("\n[3/4] Training hybrid models...")
    for encoder in models:
        args.encoder = encoder
        train_classical_ml(args)
    
    # 4. Compare results
    print("\n[4/4] Comparing all models...")
    args.output_dir = f'results/comparisons/{args.dataset}'
    evaluate_models(args)
    
    print("\n" + "="*70)
    print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*70)


def evaluate_models(args):
    """Evaluate and compare all trained models."""
    from src.evaluation.compare_models import main as compare_main
    
    print("\n" + "="*60)
    print("EVALUATING AND COMPARING MODELS")
    print("="*60)
    
    compare_args = [
        '--dataset', args.dataset,
        '--output_dir', args.output_dir or f'results/comparisons/{args.dataset}'
    ]
    
    sys.argv = ['compare_models.py'] + compare_args
    compare_main()


def predict(args):
    """Make predictions on new text."""
    print("\n" + "="*60)
    print("SENTIMENT PREDICTION")
    print("="*60)
    
    from src.models.inference import HybridSentimentPredictor
    
    # load predictor
    predictor = HybridSentimentPredictor(
        encoder_path=args.encoder_path,
        classifier_path=args.model_path,
        config_path='configs/config.yaml'
    )
    
    # predict
    result = predictor.predict(args.text)
    
    print(f"\nText: {args.text}")
    print(f"Sentiment: {result['sentiment'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Processing time: {result['processing_time_ms']:.2f}ms")


def visualize(args):
    """Generate visualizations."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    if args.viz_type == 'embeddings':
        from src.visualization.plot_embeddings import visualize_word2vec_embeddings
        
        embedding_path = f'results/embeddings/{args.dataset}/word2vec.model'
        print(f"\nVisualizing embeddings from: {embedding_path}")
        
        visualize_word2vec_embeddings(
            embedding_path=embedding_path,
            n_words=args.n_words,
            method=args.method
        )
        
    elif args.viz_type == 'metrics':
        print("[!]  Metrics visualization requires training history")
        print("   Run training first, then visualizations will be saved automatically")
        
    elif args.viz_type == 'comparison':
        print("Generating model comparison charts...")
        evaluate_models(args)
    
    else:
        print(f"Unknown visualization type: {args.viz_type}")


def demo_predict(args):
    """Demo predictions using pre-trained models."""
    print("\n" + "="*60)
    print("DEMO MODE - SENTIMENT PREDICTION")
    print("="*60)
    
    from src.models.inference import HybridSentimentPredictor
    
    # auto-detect best model
    if not args.model_path:
        # try to find trained models
        model_dir = Path(f'results/models')
        encoder_path = model_dir / f'deep_learning/{args.dataset}/lstm/lstm_best.pt'
        classifier_path = model_dir / f'classical_ml/{args.dataset}/lstm/xgboost.pkl'
        
        if not encoder_path.exists():
            print(f"[X] No trained models found for {args.dataset}")
            print(f"   Expected: {encoder_path}")
            print("\n💡 Download pre-trained models from Google Colab results.zip")
            print("   Or train models first: python src/main.py train-all --dataset imdb")
            return
        
        print(f"✓ Found pre-trained models:")
        print(f"  Encoder: {encoder_path}")
        print(f"  Classifier: {classifier_path}")
    else:
        encoder_path = args.encoder_path
        classifier_path = args.model_path
    
    # load predictor
    print("\nLoading models...")
    predictor = HybridSentimentPredictor(
        encoder_path=str(encoder_path),
        classifier_path=str(classifier_path),
        config_path='configs/config.yaml'
    )
    
    # interactive mode
    if args.interactive:
        print("\n" + "="*60)
        print("INTERACTIVE MODE - Enter 'quit' to exit")
        print("="*60)
        
        while True:
            text = input("\nEnter text to analyze: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                continue
            
            result = predictor.predict(text)
            
            print(f"\n{'='*50}")
            print(f"Sentiment: {result['sentiment'].upper()}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Probabilities: Neg={result['probabilities'][0]:.3f}, Pos={result['probabilities'][1]:.3f}")
            print(f"Time: {result['processing_time_ms']:.2f}ms")
            print(f"{'='*50}")
    
    # single prediction mode
    elif args.text:
        result = predictor.predict(args.text)
        
        print(f"\n{'='*50}")
        print(f"Text: {args.text}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Probabilities:")
        print(f"  Negative: {result['probabilities'][0]:.3f}")
        print(f"  Positive: {result['probabilities'][1]:.3f}")
        print(f"Processing time: {result['processing_time_ms']:.2f}ms")
        print(f"{'='*50}")
    
    # batch demo mode
    else:
        demo_texts = [
            "This movie was absolutely amazing! Best film I've seen all year.",
            "Terrible waste of time and money. Complete garbage.",
            "It was okay, nothing special but not terrible either.",
            "The acting was good but the plot was confusing.",
            "I loved every minute of it! Highly recommend!",
            "Boring and predictable. Don't waste your time."
        ]
        
        print("\n" + "="*60)
        print("DEMO PREDICTIONS")
        print("="*60)
        
        results = predictor.predict_batch(demo_texts)
        
        for i, (text, result) in enumerate(zip(demo_texts, results), 1):
            sentiment_icon = "😊" if result['sentiment'] == 'positive' else "😞"
            print(f"\n{i}. {text}")
            print(f"   {sentiment_icon} {result['sentiment'].upper()} ({result['confidence']:.1%})")


def test_models(args):
    """Test all trained models on test set."""
    print("\n" + "="*60)
    print("TESTING PRE-TRAINED MODELS")
    print("="*60)
    
    from src.data.data_loader import DatasetLoader
    from src.models.inference import HybridSentimentPredictor
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    # load test data
    print(f"\nLoading {args.dataset} test set...")
    loader = DatasetLoader(config_path='configs/config.yaml')
    
    if args.dataset == 'imdb':
        _, _, test_df = loader.load_imdb(use_cache=True)
    elif args.dataset == 'twitter':
        _, _, test_df = loader.load_twitter(use_cache=True)
    else:
        _, _, test_df = loader.load_custom(train_path=args.custom_path, use_cache=True)
    
    # test subset (for speed)
    if args.test_samples:
        test_df = test_df.head(args.test_samples)
    
    print(f"✓ Loaded {len(test_df)} test samples")
    
    # find all trained models
    model_dir = Path(f'results/models')
    encoders = ['lstm', 'gru', 'transformer']
    classifiers = ['xgboost', 'random_forest', 'logistic_regression']
    
    results_summary = []
    
    for encoder in encoders:
        encoder_path = model_dir / f'deep_learning/{args.dataset}/{encoder}/{encoder}_best.pt'
        
        if not encoder_path.exists():
            print(f"[!]  Skipping {encoder} (not trained)")
            continue
        
        for classifier in classifiers:
            classifier_path = model_dir / f'classical_ml/{args.dataset}/{encoder}/{classifier}.pkl'
            
            if not classifier_path.exists():
                continue
            
            model_name = f"{encoder.upper()} + {classifier.upper()}"
            print(f"\n{'='*60}")
            print(f"Testing: {model_name}")
            print(f"{'='*60}")
            
            # load predictor
            predictor = HybridSentimentPredictor(
                encoder_path=str(encoder_path),
                classifier_path=str(classifier_path),
                config_path='configs/config.yaml'
            )
            
            # predict
            print("Making predictions...")
            predictions = predictor.predict_batch(test_df['text'].tolist())
            
            y_pred = [1 if p['sentiment'] == 'positive' else 0 for p in predictions]
            y_true = test_df['label'].values
            
            # calculate metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            print(f"\nResults:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            
            results_summary.append({
                'model': model_name,
                'accuracy': acc,
                'f1': f1
            })
    
    # summary table
    if results_summary:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        import pandas as pd
        df_results = pd.DataFrame(results_summary)
        df_results = df_results.sort_values('f1', ascending=False)
        
        print(df_results.to_string(index=False))
        
        best = df_results.iloc[0]
        print(f"\n🏆 Best Model: {best['model']}")
        print(f"   Accuracy: {best['accuracy']:.4f}")
        print(f"   F1 Score: {best['f1']:.4f}")


def download_pretrained(args):
    """Download pre-trained models from Hugging Face or Google Drive."""
    print("\n" + "="*60)
    print("DOWNLOAD PRE-TRAINED MODELS")
    print("="*60)
    
    print("\n📦 Pre-trained models will be available at:")
    print("   Option 1: Google Drive link (from your Colab training)")
    print("   Option 2: Hugging Face Hub (when you upload)")
    print("   Option 3: GitHub Releases (for small models)")
    
    print("\n💡 For now, download results.zip from Google Colab:")
    print("   1. After training in Colab, download results.zip")
    print("   2. Extract to project root:")
    print("      unzip results.zip")
    print("   3. Models will be in: results/models/")


def list_models(args):
    """List all available trained models."""
    print("\n" + "="*60)
    print("AVAILABLE TRAINED MODELS")
    print("="*60)
    
    model_dir = Path('results/models')
    
    if not model_dir.exists():
        print("\n[X] No models directory found")
        print("   Train models or download pre-trained models first")
        return
    
    # Deep learning models
    dl_dir = model_dir / 'deep_learning'
    if dl_dir.exists():
        print("\n📊 Deep Learning Models (End-to-End):")
        for dataset_dir in dl_dir.iterdir():
            if dataset_dir.is_dir():
                dataset = dataset_dir.name
                print(f"\n  Dataset: {dataset}")
                for model_dir in dataset_dir.iterdir():
                    if model_dir.is_dir():
                        model_files = list(model_dir.glob('*.pt'))
                        if model_files:
                            print(f"    ✓ {model_dir.name}: {len(model_files)} checkpoint(s)")
    
    # Classical ML models
    cm_dir = model_dir / 'classical_ml'
    if cm_dir.exists():
        print("\n🔄 Hybrid Models (Encoder + Classical ML):")
        for dataset_dir in cm_dir.iterdir():
            if dataset_dir.is_dir():
                dataset = dataset_dir.name
                print(f"\n  Dataset: {dataset}")
                for encoder_dir in dataset_dir.iterdir():
                    if encoder_dir.is_dir():
                        encoder = encoder_dir.name
                        classifiers = list(encoder_dir.glob('*.pkl'))
                        if classifiers:
                            print(f"    ✓ {encoder}:")
                            for clf_path in classifiers:
                                clf_name = clf_path.stem
                                size_mb = clf_path.stat().st_size / (1024 * 1024)
                                print(f"      - {clf_name} ({size_mb:.1f} MB)")
    
    print("\n" + "="*60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Hybrid Sentiment Analysis - Unified CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Word2Vec embeddings
  python src/main.py train-embeddings --dataset imdb
  
  # Train LSTM end-to-end
  python src/main.py train-dl --dataset imdb --model lstm --epochs 10
  
  # Train hybrid model (LSTM + XGBoost)
  python src/main.py train-hybrid --dataset imdb --encoder lstm
  
  # Train all models
  python src/main.py train-all --dataset imdb
  
  # Evaluate and compare models
  python src/main.py evaluate --dataset imdb
  
  # Make prediction
  python src/main.py predict --text "This movie is amazing!" --model results/models/lstm_xgb.pkl
  
  # Visualize embeddings
  python src/main.py visualize --type embeddings --dataset imdb
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # ============================================================
    # Train Embeddings
    # ============================================================
    parser_emb = subparsers.add_parser('train-embeddings', help='Train Word2Vec embeddings')
    parser_emb.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'twitter', 'custom'],
                           help='Dataset to use')
    parser_emb.add_argument('--embedding-type', type=str, default='word2vec', choices=['word2vec'],
                           help='Embedding type')
    parser_emb.add_argument('--vector-size', type=int, default=300, help='Embedding dimension')
    parser_emb.add_argument('--window', type=int, default=5, help='Context window size')
    parser_emb.add_argument('--min-count', type=int, default=5, help='Minimum word frequency')
    
    # ============================================================
    # Train Deep Learning
    # ============================================================
    parser_dl = subparsers.add_parser('train-dl', help='Train end-to-end deep learning model')
    parser_dl.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'twitter', 'custom'],
                          help='Dataset to use')
    parser_dl.add_argument('--model', type=str, default='lstm', choices=['lstm', 'gru', 'transformer', 'bert', 'roberta', 'distilbert'],
                          help='Model architecture')
    parser_dl.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser_dl.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser_dl.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    # ============================================================
    # Train Hybrid (Classical ML)
    # ============================================================
    parser_hybrid = subparsers.add_parser('train-hybrid', help='Train hybrid model (encoder + classical ML)')
    parser_hybrid.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'twitter', 'custom'],
                              help='Dataset to use')
    parser_hybrid.add_argument('--encoder', type=str, default='lstm', choices=['lstm', 'gru', 'transformer'],
                              help='Encoder type')
    
    # ============================================================
    # Train All Models
    # ============================================================
    parser_all = subparsers.add_parser('train-all', help='Train all models for comparison')
    parser_all.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'twitter', 'custom'],
                           help='Dataset to use')
    parser_all.add_argument('--epochs', type=int, default=10, help='Number of epochs for DL models')
    parser_all.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser_all.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser_all.add_argument('--include-bert', action='store_true', 
                       help='Include BERT/RoBERTa (slower, requires more memory)')
    
    # ============================================================
    # Evaluate Models
    # ============================================================
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate and compare models')
    parser_eval.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'twitter', 'custom'],
                            help='Dataset to use')
    parser_eval.add_argument('--output-dir', type=str, help='Output directory for results')
    
    # ============================================================
    # Predict
    # ============================================================
    parser_pred = subparsers.add_parser('predict', help='Make predictions on new text')
    parser_pred.add_argument('--text', type=str, required=True, help='Text to analyze')
    parser_pred.add_argument('--model-path', type=str, required=True, help='Path to classifier (.pkl)')
    parser_pred.add_argument('--encoder-path', type=str, required=True, help='Path to encoder (.pt)')
    
    # ============================================================
    # Visualize
    # ============================================================
    parser_viz = subparsers.add_parser('visualize', help='Generate visualizations')
    parser_viz.add_argument('--type', dest='viz_type', type=str, required=True,
                           choices=['embeddings', 'metrics', 'comparison'],
                           help='Visualization type')
    parser_viz.add_argument('--dataset', type=str, default='imdb', help='Dataset')
    parser_viz.add_argument('--n-words', type=int, default=500, help='Number of words for embedding viz')
    parser_viz.add_argument('--method', type=str, default='tsne', choices=['tsne', 'pca'],
                           help='Dimensionality reduction method')
    parser_viz.add_argument('--output-dir', type=str, help='Output directory')
    
    # parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # execute command
    command_map = {
        'train-embeddings': train_embeddings,
        'train-dl': train_deep_learning,
        'train-hybrid': train_classical_ml,
        'train-all': train_all,
        'evaluate': evaluate_models,
        'predict': predict,
        'visualize': visualize
    }
    
    try:
        command_map[args.command](args)
        print("\n✓ Command executed successfully!")
    except KeyboardInterrupt:
        print("\n\n[!]  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()