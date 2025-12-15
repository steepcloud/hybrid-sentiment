import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_loader import DatasetLoader
from src.data.preprocessor import TextPreprocessor
from src.models.deep_learning.lstm_encoder import create_lstm_classifier_from_config
from src.models.deep_learning.gru_encoder import create_gru_classifier_from_config
from src.models.deep_learning.transformer_encoder import create_transformer_classifier_from_config
from src.models.deep_learning.bert_encoder import create_bert_classifier_from_config
from src.evaluation.metrics import calculate_metrics, print_metrics
from src.utils.helpers import set_seed, save_checkpoint, format_time, get_device, count_parameters
from src.utils.config import Config


class EndToEndDLTrainer:
    """Trainer for end-to-end deep learning models (LSTM, GRU, Transformer)."""
    
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        model_type: str = "lstm"
    ):
        """
        Args:
            config_path: Path to configuration file
            model_type: Type of model ('lstm', 'gru', 'transformer', 'bert')
        """
        # load config
        self.config = Config(config_path)
        self.model_type = model_type
        
        # set random seed
        set_seed(self.config.get('project.random_seed', 42))
        
        # get device
        self.device = get_device()
        
        # initialize components
        self.data_loader = DatasetLoader()
        self.preprocessor = TextPreprocessor()
        
        # set vocab size from config
        self.preprocessor.max_vocab_size = self.config.get('data.vocab_size', 20000)
        self.preprocessor.max_length = self.config.get('data.max_length', 512)
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"Initialized EndToEndDLTrainer")
        print(f"  Model type: {model_type}")
        print(f"  Device: {self.device}")
    
    def prepare_data(
        self,
        texts: List[str],
        labels: np.ndarray,
        batch_size: int = 32
    ) -> DataLoader:
        """
        Prepare data for training.
        
        Args:
            texts: List of text strings
            labels: Array of labels
            batch_size: Batch size
            
        Returns:
            DataLoader
        """

        if self.model_type in ['bert', 'roberta', 'distilbert']:
            # use BERT tokenizer
            model_name_map = {
                'bert': 'bert-base-uncased',
                'roberta': 'roberta-base',
                'distilbert': 'distilbert-base-uncased'
            }
            model_name = model_name_map[self.model_type]
            
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # tokenize texts
            encodings = tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']
            labels_tensor = torch.LongTensor(labels)
            
            dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            return dataloader
    
        # tokenize and encode texts
        input_ids_list = []
        lengths_list = []
        
        for text in texts:
            cleaned = self.preprocessor.clean_text(text)
            tokens = self.preprocessor.tokenize(cleaned)
            # convert tokens to IDs
            ids = [self.preprocessor.vocab.get(token, self.preprocessor.UNK_IDX) 
                   for token in tokens]
            
            input_ids_list.append(ids)
            lengths_list.append(len(ids))
        
        # pad sequences
        max_len = self.config.get('data.max_length', 512)
        padded_ids = np.zeros((len(input_ids_list), max_len), dtype=np.int64)
        
        for i, ids in enumerate(input_ids_list):
            length = min(len(ids), max_len)
            padded_ids[i, :length] = ids[:length]
        
        # convert to tensors
        input_ids = torch.LongTensor(padded_ids)
        lengths = torch.LongTensor([min(l, max_len) for l in lengths_list])
        labels_tensor = torch.LongTensor(labels)
        
        # create dataset and dataloader
        dataset = TensorDataset(input_ids, lengths, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    def create_model(self, vocab_size: int = None) -> nn.Module:
        """
        Create model based on type.
        
        Args:
            vocab_size: Vocabulary size (optional for BERT)
            
        Returns:
            Model
        """
        print(f"\nCreating {self.model_type.upper()} model...")
        
        if self.model_type == 'bert':
            model = create_bert_classifier_from_config('bert-base-uncased')
        elif self.model_type == 'roberta':
            model = create_bert_classifier_from_config('roberta-base')
        elif self.model_type == 'distilbert':
            model = create_bert_classifier_from_config('distilbert-base-uncased')
        elif self.model_type == 'lstm':
            model = create_lstm_classifier_from_config(vocab_size=vocab_size)
        elif self.model_type == 'gru':
            model = create_gru_classifier_from_config(vocab_size=vocab_size)
        elif self.model_type == 'transformer':
            model = create_transformer_classifier_from_config(vocab_size=vocab_size)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # better weight initialization for classifier head
        def init_classifier_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # apply to classifier only (not encoder/embeddings)
        if hasattr(model, 'classifier'):
            model.classifier.apply(init_classifier_weights)
            print("  ✓ Applied Xavier initialization to classifier")
            
        model = model.to(self.device)
        
        # print model info
        total_params = count_parameters(model)
        print(f"  Total parameters: {total_params:,}")
        
        return model
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict:
        """
        Train for one epoch. (handles BERT batches)
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # handle BERT vs traditional models
            if self.model_type in ['bert', 'roberta', 'distilbert']:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
            else:
                input_ids, lengths, labels = batch
                input_ids = input_ids.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.model_type == 'transformer':
                    logits = self.model(input_ids)
                else:
                    logits = self.model(input_ids, lengths)
            
            loss = self.criterion(logits, labels)
            
            # backward pass
            loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # calculate accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            
            # update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def evaluate(
        self,
        eval_loader: DataLoader,
        dataset_name: str = "Validation"
    ) -> Dict:
        """
        Evaluate model on dataset.
        
        Args:
            eval_loader: Evaluation data loader
            dataset_name: Name of dataset for logging
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for input_ids, lengths, labels in eval_loader:
                input_ids = input_ids.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                
                # forward pass
                if self.model_type == 'transformer':
                    logits = self.model(input_ids)
                else:
                    logits = self.model(input_ids, lengths)
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # get predictions
                probs = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # calculate metrics
        avg_loss = total_loss / len(eval_loader)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        metrics = calculate_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = avg_loss
        
        print(f"\n{dataset_name} Results:")
        print_metrics(metrics, title=dataset_name)
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        save_dir: str = "results/models/deep_learning"
    ) -> Dict:
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            learning_rate: Learning rate
            save_dir: Directory to save models
            
        Returns:
            Dictionary with training history
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_type.upper()} Model")
        print(f"{'='*60}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {train_loader.batch_size}")
        
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        best_val_f1 = 0
        best_epoch = 0
        
        # training loop
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            print(f"\nTraining - Loss: {train_metrics['loss']:.4f}, "
                  f"Accuracy: {train_metrics['accuracy']:.4f}")
            
            # validate
            val_metrics = self.evaluate(val_loader, "Validation")
            
            # update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            
            # save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_epoch = epoch
                
                # save checkpoint
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f"{self.model_type}_best.pt")
                
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics['loss'],
                    checkpoint_path,
                    val_f1=val_metrics['f1'],
                    val_accuracy=val_metrics['accuracy']
                )
                
                print(f"✓ Best model saved (F1: {best_val_f1:.4f})")
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Best validation F1: {best_val_f1:.4f}")
        print(f"{'='*60}")
        
        return history
    
    def train_on_dataset(
        self,
        dataset: str = 'imdb',
        batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        Train model on a dataset.
        
        Args:
            dataset: Dataset to use ('imdb' or 'twitter')
            batch_size: Batch size
            num_epochs: Number of epochs
            learning_rate: Learning rate
            
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print(f"Training on {dataset.upper()} Dataset")
        print(f"{'='*60}")
        
        # load data
        print("\nLoading dataset...")
        train_df, val_df, test_df = self.data_loader.load_dataset(dataset, use_cache=True)
        
        # build vocabulary
        print("\nBuilding vocabulary...")
        self.preprocessor.build_vocab(train_df['text'].tolist())
        
        # create model
        vocab_size = len(self.preprocessor.vocab)
        self.model = self.create_model(vocab_size)
        
        # prepare data
        print("\nPreparing data loaders...")
        train_loader = self.prepare_data(
            train_df['text'].tolist(),
            train_df['label'].values,
            batch_size=batch_size
        )
        
        val_loader = self.prepare_data(
            val_df['text'].tolist(),
            val_df['label'].values,
            batch_size=batch_size
        )
        
        test_loader = self.prepare_data(
            test_df['text'].tolist(),
            test_df['label'].values,
            batch_size=batch_size
        )
        
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # train
        save_dir = f"results/models/deep_learning/{dataset}/{self.model_type}"
        history = self.train(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            save_dir=save_dir
        )
        
        # load best model and evaluate on test set
        print(f"\n{'='*60}")
        print("Evaluating best model on test set...")
        print(f"{'='*60}")
        
        checkpoint_path = os.path.join(save_dir, f"{self.model_type}_best.pt")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = self.evaluate(test_loader, "Test")
        
        # compile results
        results = {
            'history': history,
            'test_metrics': test_metrics,
            'model_type': self.model_type,
            'dataset': dataset,
            'vocab_size': vocab_size,
            'num_params': count_parameters(self.model)
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Train end-to-end deep learning models')
    parser.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'twitter'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'gru', 'transformer', 'bert', 'roberta', 'distilbert'],
                        help='Model type')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # create trainer
    trainer = EndToEndDLTrainer(
        config_path=args.config,
        model_type=args.model
    )
    
    # train
    results = trainer.train_on_dataset(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    print("\n✓ Training complete!")
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"  Precision: {results['test_metrics']['precision']:.4f}")
    print(f"  Recall: {results['test_metrics']['recall']:.4f}")
    print(f"  F1 Score: {results['test_metrics']['f1']:.4f}")
    print(f"  ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")


if __name__ == "__main__":
    main()