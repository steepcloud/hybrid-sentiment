import torch
import numpy as np
import os
import sys
import yaml
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_loader import DatasetLoader
from src.data.preprocessor import TextPreprocessor
from src.evaluation.metrics import calculate_metrics
from src.utils.helpers import set_seed


class ClassicalMLTrainer:
    """Trainer for classical ML models using pre-trained embeddings."""
    
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        encoder_type: str = "lstm",
        encoder_path: Optional[str] = None
    ):
        """
        Args:
            config_path: Path to configuration file
            encoder_type: Type of encoder ('lstm', 'gru', 'transformer', 'word2vec')
            encoder_path: Path to pre-trained encoder (if available)
        """
        # load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.encoder_type = encoder_type
        self.encoder_path = encoder_path
        
        # set random seed
        set_seed(self.config['project']['random_seed'])
        
        # initialize components
        self.data_loader = DatasetLoader()
        self.preprocessor = TextPreprocessor()
        
        # set vocab size and max length from config
        self.preprocessor.max_vocab_size = self.config['data']['vocab_size']
        self.preprocessor.max_length = self.config['data']['max_length']

        self.encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Initialized ClassicalMLTrainer")
        print(f"  Encoder type: {encoder_type}")
        print(f"  Device: {self.device}")
    
    def load_encoder(self):
        """Load pre-trained encoder."""
        print(f"\nLoading {self.encoder_type} encoder...")
        
        if self.encoder_path and os.path.exists(self.encoder_path):
            # load pre-trained encoder
            print(f"  Loading from: {self.encoder_path}")
            checkpoint = torch.load(self.encoder_path, map_location=self.device)
            
            vocab_size = checkpoint.get('vocab_size', self.config['data']['vocab_size'])
            
            # create encoder based on type
            if self.encoder_type == 'lstm':
                from src.models.deep_learning.lstm_encoder import create_lstm_encoder_from_config
                self.encoder = create_lstm_encoder_from_config(vocab_size=vocab_size)
            elif self.encoder_type == 'gru':
                from src.models.deep_learning.gru_encoder import create_gru_encoder_from_config
                self.encoder = create_gru_encoder_from_config(vocab_size=vocab_size)
            elif self.encoder_type == 'transformer':
                from src.models.deep_learning.transformer_encoder import create_transformer_encoder_from_config
                self.encoder = create_transformer_encoder_from_config(vocab_size=vocab_size)
            elif self.encoder_type in ['bert', 'roberta', 'distilbert']:
                # load BERT-based encoder
                from src.models.deep_learning.bert_encoder import BERTClassifier
                model_name_map = {
                    'bert': 'bert-base-uncased',
                    'roberta': 'roberta-base',
                    'distilbert': 'distilbert-base-uncased'
                }
                model_name = model_name_map[self.encoder_type]
                self.encoder = BERTClassifier(model_name=model_name)
            
            # load state dict for non-BERT models
            if self.encoder_type not in ['bert', 'roberta', 'distilbert']:
                self.encoder.load_state_dict(checkpoint['model_state_dict'])
            else:
                # for BERT, load the full checkpoint
                self.encoder.load_state_dict(checkpoint['model_state_dict'])
            self.encoder.to(self.device)
            self.encoder.eval()
            print(f"✓ Encoder loaded successfully")
        else:
            print(f"  No pre-trained encoder found, using random initialization")
            # create encoder with random weights
            vocab_size = len(self.preprocessor.vocab) if hasattr(self.preprocessor, 'vocab') else self.config['data']['vocab_size']
            
            if self.encoder_type == 'lstm':
                from src.models.deep_learning.lstm_encoder import create_lstm_encoder_from_config
                self.encoder = create_lstm_encoder_from_config(vocab_size=vocab_size)
            elif self.encoder_type == 'gru':
                from src.models.deep_learning.gru_encoder import create_gru_encoder_from_config
                self.encoder = create_gru_encoder_from_config(vocab_size=vocab_size)
            elif self.encoder_type == 'transformer':
                from src.models.deep_learning.transformer_encoder import create_transformer_encoder_from_config
                self.encoder = create_transformer_encoder_from_config(vocab_size=vocab_size)
            elif self.encoder_type in ['bert', 'roberta', 'distilbert']:
                # create BERT encoder with random weights
                from src.models.deep_learning.bert_encoder import BERTClassifier
                model_name_map = {
                    'bert': 'bert-base-uncased',
                    'roberta': 'roberta-base',
                    'distilbert': 'distilbert-base-uncased'
                }
                model_name = model_name_map[self.encoder_type]
                self.encoder = BERTClassifier(model_name=model_name)
            
            self.encoder.to(self.device)
            self.encoder.eval()
            print(f"✓ Random encoder initialized")
    
    def extract_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract embeddings from texts using the encoder.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Embeddings array [n_samples, embedding_dim]
        """
        print(f"\nExtracting embeddings for {len(texts)} samples...")
        
        embeddings_list = []

        is_bert = self.encoder_type in ['bert', 'roberta', 'distilbert']

        if is_bert:
            # use BERT tokenizer
            tokenizer = self.encoder.encoder.tokenizer
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]

                if is_bert:
                    # BERT tokenization
                    encodings = tokenizer(
                        batch_texts,
                        truncation=True,
                        padding='max_length',
                        max_length=512,
                        return_tensors='pt'
                    )
                    input_ids = encodings['input_ids'].to(self.device)
                    attention_mask = encodings['attention_mask'].to(self.device)
                    
                    # get embeddings
                    batch_embeddings = self.encoder.get_embeddings(input_ids, attention_mask)
                else:
                    # tokenize and encode
                    batch_ids = []
                    batch_lengths = []
                    
                    for text in batch_texts:
                        cleaned = self.preprocessor.clean_text(text)
                        tokens = self.preprocessor.tokenize(cleaned)
                        ids = [self.preprocessor.vocab.get(token, self.preprocessor.UNK_IDX) 
                            for token in tokens]
                        
                        batch_ids.append(ids)
                        batch_lengths.append(len(ids))
                    
                    # pad sequences
                    max_len = min(max(batch_lengths) if batch_lengths else 1, self.config['data']['max_length'])
                    padded_ids = np.zeros((len(batch_ids), max_len), dtype=np.int64)
                    
                    for j, ids in enumerate(batch_ids):
                        length = min(len(ids), max_len)
                        padded_ids[j, :length] = ids[:length]
                    
                    # convert to tensors
                    input_ids = torch.LongTensor(padded_ids).to(self.device)
                    lengths = torch.LongTensor([min(l, max_len) for l in batch_lengths]).to(self.device)
                    
                    # get embeddings
                    if self.encoder_type == 'transformer':
                        batch_embeddings = self.encoder(input_ids, return_sequence=False)
                    else:
                        batch_embeddings = self.encoder(input_ids, lengths, return_sequence=False)
                
                embeddings_list.append(batch_embeddings.cpu().numpy())
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"  Processed {i + len(batch_texts)}/{len(texts)} samples")
        
        embeddings = np.vstack(embeddings_list)
        print(f"✓ Embeddings extracted: {embeddings.shape}")
        
        return embeddings
    
    def train_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[object, Dict]:
        """
        Train a classical ML model.
        
        Args:
            model_type: Type of model ('logistic_regression', 'random_forest', 'xgboost')
            X_train: Training embeddings
            y_train: Training labels
            X_val: Validation embeddings
            y_val: Validation labels
            
        Returns:
            Tuple of (trained model, metrics)
        """
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*60}")
        
        # create model
        if model_type == 'logistic_regression':
            from src.models.classical_ml.logistic_regression import create_logistic_regression_from_config
            model = create_logistic_regression_from_config()
        elif model_type == 'random_forest':
            from src.models.classical_ml.random_forest import create_random_forest_from_config
            model = create_random_forest_from_config()
        elif model_type == 'xgboost':
            from src.models.classical_ml.xgboost_classifier import create_xgboost_from_config
            model = create_xgboost_from_config()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # train
        train_metrics = model.fit(X_train, y_train, X_val, y_val)
        
        return model, train_metrics
    
    def evaluate_model(
        self,
        model: object,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            X_test: Test embeddings
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating on test set...")
        
        # predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        
        print(f"\nTest Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def train_all_models(
        self,
        dataset: str = 'imdb',
        save_models: bool = True
    ) -> Dict:
        """
        Train all classical ML models.
        
        Args:
            dataset: Dataset to use ('imdb' or 'twitter')
            save_models: Whether to save trained models
            
        Returns:
            Dictionary with all results
        """
        print(f"\n{'='*60}")
        print(f"Training Classical ML Models on {dataset.upper()}")
        print(f"{'='*60}")
        
        # load data
        print("\nLoading dataset...")
        train_df, val_df, test_df = self.data_loader.load_dataset(dataset, use_cache=True)
        
        # build vocabulary
        print("\nBuilding vocabulary...")
        all_texts = train_df['text'].tolist()
        self.preprocessor.build_vocab(all_texts)
        
        # load encoder
        self.load_encoder()
        
        # extract embeddings
        X_train = self.extract_embeddings(train_df['text'].tolist())
        y_train = train_df['label'].values
        
        X_val = self.extract_embeddings(val_df['text'].tolist())
        y_val = val_df['label'].values
        
        X_test = self.extract_embeddings(test_df['text'].tolist())
        y_test = test_df['label'].values
        
        # train models
        results = {}
        model_types = ['logistic_regression', 'random_forest', 'xgboost']
        
        for model_type in model_types:
            model, train_metrics = self.train_model(
                model_type, X_train, y_train, X_val, y_val
            )
            
            test_metrics = self.evaluate_model(model, X_test, y_test)
            
            results[model_type] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'model': model
            }
            
            # save model
            if save_models:
                save_dir = f"results/models/classical_ml/{dataset}/{self.encoder_type}"
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, f"{model_type}.pkl")
                model.save_model(model_path)
        
        # save results summary
        self._save_results_summary(results, dataset)
        
        return results
    
    def _save_results_summary(self, results: Dict, dataset: str):
        """Save results summary to file."""
        save_dir = f"results/classical_ml/{dataset}/{self.encoder_type}"
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(save_dir, f"results_{timestamp}.pkl")
        
        # prepare results (remove model objects for pickling)
        results_to_save = {}
        for model_type, data in results.items():
            results_to_save[model_type] = {
                'train_metrics': data['train_metrics'],
                'test_metrics': data['test_metrics']
            }
        
        with open(results_path, 'wb') as f:
            pickle.dump(results_to_save, f)
        
        print(f"\n✓ Results saved to {results_path}")
        
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
        print("-" * 60)
        
        for model_type, data in results.items():
            train_acc = data['train_metrics']['train_accuracy']
            val_acc = data['train_metrics'].get('val_accuracy', 0)
            test_acc = data['test_metrics']['accuracy']
            test_f1 = data['test_metrics']['f1']
            
            print(f"{model_type:<20} {train_acc:<12.4f} {val_acc:<12.4f} {test_acc:<12.4f} {test_f1:<12.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train classical ML models')
    parser.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'twitter'],
                        help='Dataset to use')
    parser.add_argument('--encoder', type=str, default='lstm', choices=['lstm', 'gru', 'transformer', 'bert', 'roberta', 'distilbert'],
                        help='Encoder type')
    parser.add_argument('--encoder_path', type=str, default=None,
                        help='Path to pre-trained encoder')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # create trainer
    trainer = ClassicalMLTrainer(
        config_path=args.config,
        encoder_type=args.encoder,
        encoder_path=args.encoder_path
    )
    
    # train all models
    results = trainer.train_all_models(
        dataset=args.dataset,
        save_models=True
    )
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()