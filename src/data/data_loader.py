import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Tuple, List, Optional, Dict
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import pickle
from collections import Counter
import random


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment analysis."""
    
    def __init__(
        self, 
        sequences: List[List[int]], 
        labels: List[int], 
        max_length: int = 256,
        augment: bool = False
    ):
        """
        Args:
            sequences: List of tokenized sequences (list of integers)
            labels: List of sentiment labels (0 or 1)
            max_length: Maximum sequence length for padding/truncating
            augment: Whether to apply data augmentation
        """
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.sequences)
    
    def _augment_sequence(self, sequence: List[int]) -> List[int]:
        """Apply data augmentation to a sequence."""
        if not self.augment or random.random() > 0.5:
            return sequence
        
        aug_sequence = sequence.copy()
        
        # Random deletion (10% chance to delete each token)
        aug_sequence = [token for token in aug_sequence if random.random() > 0.1]
        
        # Random swap (swap adjacent tokens with 10% chance)
        for i in range(len(aug_sequence) - 1):
            if random.random() < 0.1:
                aug_sequence[i], aug_sequence[i + 1] = aug_sequence[i + 1], aug_sequence[i]
        
        return aug_sequence if len(aug_sequence) > 0 else sequence
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Apply augmentation if enabled
        if self.augment:
            sequence = self._augment_sequence(sequence)
        
        # Pad or truncate
        if len(sequence) < self.max_length:
            sequence = sequence + [0] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        
        return {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DatasetLoader:
    """Loads and prepares datasets for sentiment analysis."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.random_seed = self.config['project']['random_seed']

        # Create directories
        os.makedirs(self.data_config['data_dir'], exist_ok=True)
        os.makedirs(self.data_config['processed_dir'], exist_ok=True)
        os.makedirs(self.data_config['embeddings_dir'], exist_ok=True)
        
        # Cache file path
        self.cache_file = os.path.join(
            self.data_config['processed_dir'], 
            f"{self.data_config['dataset_name']}_cache.pkl"
        )
        
    def _cache_exists(self) -> bool:
        """Check if cached dataset exists."""
        return os.path.exists(self.cache_file)
    
    def _save_to_cache(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save dataset splits to cache."""
        cache_data = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved dataset to cache: {self.cache_file}")
    
    def _load_from_cache(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load dataset splits from cache."""
        with open(self.cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"Loaded dataset from cache: {self.cache_file}")
        return cache_data['train'], cache_data['val'], cache_data['test']
    
    def get_data_statistics(self, df: pd.DataFrame, split_name: str = "Dataset") -> Dict:
        """
        Calculate and display statistics for a dataset split.
        
        Args:
            df: DataFrame with 'text' and 'label' columns
            split_name: Name of the split (e.g., "Train", "Val", "Test")
            
        Returns:
            Dictionary with statistics
        """
        # Calculate text lengths
        text_lengths = df['text'].apply(lambda x: len(x.split()))
        
        stats = {
            'total_samples': len(df),
            'class_distribution': df['label'].value_counts().to_dict(),
            'class_balance': df['label'].value_counts(normalize=True).to_dict(),
            'text_length': {
                'mean': text_lengths.mean(),
                'median': text_lengths.median(),
                'std': text_lengths.std(),
                'min': text_lengths.min(),
                'max': text_lengths.max(),
                'percentile_25': text_lengths.quantile(0.25),
                'percentile_75': text_lengths.quantile(0.75),
                'percentile_95': text_lengths.quantile(0.95),
                'percentile_99': text_lengths.quantile(0.99)
            }
        }
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"{split_name} Statistics")
        print(f"{'='*60}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"\nClass Distribution:")
        for label, count in stats['class_distribution'].items():
            percentage = stats['class_balance'][label] * 100
            label_name = "Positive" if label == 1 else "Negative"
            print(f"  {label_name} (label={label}): {count} ({percentage:.2f}%)")
        
        print(f"\nText Length Statistics (in words):")
        print(f"  Mean: {stats['text_length']['mean']:.2f}")
        print(f"  Median: {stats['text_length']['median']:.2f}")
        print(f"  Std Dev: {stats['text_length']['std']:.2f}")
        print(f"  Min: {stats['text_length']['min']}")
        print(f"  Max: {stats['text_length']['max']}")
        print(f"  25th percentile: {stats['text_length']['percentile_25']:.2f}")
        print(f"  75th percentile: {stats['text_length']['percentile_75']:.2f}")
        print(f"  95th percentile: {stats['text_length']['percentile_95']:.2f}")
        print(f"  99th percentile: {stats['text_length']['percentile_99']:.2f}")
        
        return stats
    
    def load_imdb(self, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load IMDb dataset.
        
        Args:
            use_cache: Whether to use cached version if available
            
        Returns:
            train_df, val_df, test_df
        """
        if use_cache and self._cache_exists():
            return self._load_from_cache()
        
        print("Loading IMDb dataset...")
        dataset = load_dataset("imdb")
        
        # Convert to DataFrame
        train_data = pd.DataFrame({
            'text': dataset['train']['text'],
            'label': dataset['train']['label']
        })
        
        test_df = pd.DataFrame({
            'text': dataset['test']['text'],
            'label': dataset['test']['label']
        })
        
        # Split train into train and validation
        val_split = self.data_config['validation_split']
        train_df, val_df = train_test_split(
            train_data, 
            test_size=val_split, 
            random_state=self.random_seed,
            stratify=train_data['label']
        )
        
        # Sample if specified
        sample_size = self.data_config.get('sample_size')
        if sample_size is not None:
            train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=self.random_seed)
            val_df = val_df.sample(n=min(int(sample_size * val_split), len(val_df)), random_state=self.random_seed)
            test_df = test_df.sample(n=min(int(sample_size * self.data_config['test_split']), len(test_df)), random_state=self.random_seed)
        
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        # Cache the dataset
        if use_cache:
            self._save_to_cache(train_df, val_df, test_df)
        
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def load_twitter(self, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load Twitter sentiment dataset (sentiment140).
        
        Args:
            use_cache: Whether to use cached version if available
            
        Returns:
            train_df, val_df, test_df
        """
        if use_cache and self._cache_exists():
            return self._load_from_cache()
        
        print("Loading Twitter sentiment140 dataset...")
        dataset = load_dataset("sentiment140", split='train')
        
        # Convert to binary sentiment (0: negative, 1: positive)
        # Original: 0 = negative, 2 = neutral, 4 = positive
        df = pd.DataFrame({
            'text': dataset['text'],
            'label': [0 if label == 0 else 1 for label in dataset['sentiment']]
        })
        
        # Remove neutral sentiments if they exist
        df = df[df['label'].isin([0, 1])]
        
        # Split into train, val, test
        test_split = self.data_config['test_split']
        val_split = self.data_config['validation_split']
        
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_split, 
            random_state=self.random_seed,
            stratify=df['label']
        )
        
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=val_split / (1 - test_split), 
            random_state=self.random_seed,
            stratify=train_val_df['label']
        )
        
        # Sample if specified
        sample_size = self.data_config.get('sample_size')
        if sample_size is not None:
            train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=self.random_seed)
            val_df = val_df.sample(n=min(int(sample_size * val_split), len(val_df)), random_state=self.random_seed)
            test_df = test_df.sample(n=min(int(sample_size * test_split), len(test_df)), random_state=self.random_seed)
        
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        # Cache the dataset
        if use_cache:
            self._save_to_cache(train_df, val_df, test_df)
        
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def load_custom(self, train_path: str, test_path: str = None, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load custom CSV dataset.
        Expected columns: 'text', 'label'
        
        Args:
            train_path: Path to training CSV
            test_path: Optional path to test CSV
            use_cache: Whether to use cached version if available
            
        Returns:
            train_df, val_df, test_df
        """
        if use_cache and self._cache_exists():
            return self._load_from_cache()
        
        print(f"Loading custom dataset from {train_path}...")
        
        train_data = pd.read_csv(train_path)
        
        if test_path:
            test_df = pd.read_csv(test_path)
            
            # Split train into train and validation
            val_split = self.data_config['validation_split']
            train_df, val_df = train_test_split(
                train_data, 
                test_size=val_split, 
                random_state=self.random_seed,
                stratify=train_data['label']
            )
        else:
            # Split into train, val, test
            test_split = self.data_config['test_split']
            val_split = self.data_config['validation_split']
            
            train_val_df, test_df = train_test_split(
                train_data, 
                test_size=test_split, 
                random_state=self.random_seed,
                stratify=train_data['label']
            )
            
            train_df, val_df = train_test_split(
                train_val_df, 
                test_size=val_split / (1 - test_split), 
                random_state=self.random_seed,
                stratify=train_val_df['label']
            )
        
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        # Cache the dataset
        if use_cache:
            self._save_to_cache(train_df, val_df, test_df)
        
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def load_dataset(self, dataset_name: str = None, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load dataset based on configuration.
        
        Args:
            dataset_name: Override dataset name from config
            use_cache: Whether to use cached version if available
            
        Returns:
            train_df, val_df, test_df
        """
        if dataset_name is None:
            dataset_name = self.data_config['dataset_name']
        
        if dataset_name.lower() == 'imdb':
            return self.load_imdb(use_cache=use_cache)
        elif dataset_name.lower() == 'twitter':
            return self.load_twitter(use_cache=use_cache)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Use 'imdb', 'twitter', or provide custom paths.")
    
    def create_dataloaders(
        self,
        train_sequences: List[List[int]],
        train_labels: List[int],
        val_sequences: List[List[int]],
        val_labels: List[int],
        test_sequences: List[List[int]],
        test_labels: List[int],
        batch_size: int = None,
        augment_train: bool = False
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for train, validation, and test sets.
        
        Args:
            train_sequences: Training sequences
            train_labels: Training labels
            val_sequences: Validation sequences
            val_labels: Validation labels
            test_sequences: Test sequences
            test_labels: Test labels
            batch_size: Override batch size from config
            augment_train: Apply data augmentation to training set
            
        Returns:
            train_loader, val_loader, test_loader
        """
        if batch_size is None:
            batch_size = self.config['training']['batch_size']
        
        max_length = self.data_config['max_length']
        
        train_dataset = SentimentDataset(train_sequences, train_labels, max_length, augment=augment_train)
        val_dataset = SentimentDataset(val_sequences, val_labels, max_length, augment=False)
        test_dataset = SentimentDataset(test_sequences, test_labels, max_length, augment=False)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader
    
    def save_splits(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ):
        """Save train/val/test splits to CSV files."""
        save_dir = self.data_config['processed_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(save_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(save_dir, 'test.csv'), index=False)
        
        print(f"Saved splits to {save_dir}")
    
    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load previously saved train/val/test splits."""
        save_dir = self.data_config['processed_dir']
        
        train_df = pd.read_csv(os.path.join(save_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(save_dir, 'val.csv'))
        test_df = pd.read_csv(os.path.join(save_dir, 'test.csv'))
        
        print(f"Loaded splits from {save_dir}")
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
        
        return train_df, val_df, test_df


if __name__ == "__main__":
    # Test the data loader
    loader = DatasetLoader()
    
    # Load IMDb dataset (will use cache if available)
    train_df, val_df, test_df = loader.load_dataset('imdb', use_cache=True)
    
    # Get statistics for each split
    train_stats = loader.get_data_statistics(train_df, "Training Set")
    val_stats = loader.get_data_statistics(val_df, "Validation Set")
    test_stats = loader.get_data_statistics(test_df, "Test Set")
    
    # Save splits
    loader.save_splits(train_df, val_df, test_df)
    
    # Display sample
    print("\nSample from training set:")
    print(train_df.head(3))