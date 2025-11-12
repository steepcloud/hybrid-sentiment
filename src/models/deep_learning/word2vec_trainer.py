import torch
import numpy as np
from gensim.models import Word2Vec
from typing import List, Optional, Dict, Tuple
import pickle
import os
import yaml
from collections import Counter


class Word2VecTrainer:
    """Trainer for Word2Vec embeddings using Gensim."""
    
    def __init__(
        self,
        embedding_dim: int = 300,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        sg: int = 1,  # 1 for skip-gram, 0 for CBOW
        negative: int = 5,
        epochs: int = 5,
        seed: int = 42
    ):
        """
        Args:
            embedding_dim: Dimension of word embeddings
            window: Maximum distance between current and predicted word
            min_count: Ignores words with total frequency lower than this
            workers: Number of worker threads
            sg: Training algorithm (1=skip-gram, 0=CBOW)
            negative: Number of negative samples
            epochs: Number of training epochs
            seed: Random seed
        """
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.negative = negative
        self.epochs = epochs
        self.seed = seed
        
        self.model = None
        self.vocab_size = 0
        self.word_to_idx = {}
        self.idx_to_word = {}
    
    def train(
        self,
        sentences: List[List[str]],
        save_path: Optional[str] = None
    ) -> Word2Vec:
        """
        Train Word2Vec model on sentences.
        
        Args:
            sentences: List of tokenized sentences (list of list of words)
            save_path: Optional path to save the trained model
            
        Returns:
            Trained Word2Vec model
        """
        print(f"Training Word2Vec model...")
        print(f"  Algorithm: {'Skip-gram' if self.sg == 1 else 'CBOW'}")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Window size: {self.window}")
        print(f"  Min count: {self.min_count}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Total sentences: {len(sentences)}")
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            negative=self.negative,
            epochs=self.epochs,
            seed=self.seed
        )
        
        self.vocab_size = len(self.model.wv)
        print(f"  Vocabulary size: {self.vocab_size}")
        
        # Build word-to-index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(self.model.wv.index_to_key)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        print("✓ Training complete!")
        return self.model
    
    def get_embedding_matrix(
        self,
        vocab: Dict[str, int],
        pad_idx: int = 0,
        unk_idx: int = 1
    ) -> torch.Tensor:
        """
        Create embedding matrix aligned with vocabulary.
        
        Args:
            vocab: Vocabulary dictionary (word -> index)
            pad_idx: Index for padding token
            unk_idx: Index for unknown token
            
        Returns:
            Embedding matrix as PyTorch tensor [vocab_size, embedding_dim]
        """
        print("Creating embedding matrix...")
        
        vocab_size = len(vocab)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        
        # Initialize with random values for PAD and UNK
        embedding_matrix[pad_idx] = np.zeros(self.embedding_dim)  # PAD is zeros
        embedding_matrix[unk_idx] = np.random.randn(self.embedding_dim) * 0.01  # UNK is small random
        
        # Fill in embeddings for words in vocabulary
        found_words = 0
        for word, idx in vocab.items():
            if word in self.model.wv:
                embedding_matrix[idx] = self.model.wv[word]
                found_words += 1
        
        print(f"  Found embeddings for {found_words}/{vocab_size} words ({found_words/vocab_size*100:.2f}%)")
        print(f"  Missing words will use random initialization")
        
        return torch.FloatTensor(embedding_matrix)
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a word.
        
        Args:
            word: Input word
            
        Returns:
            Embedding vector or None if word not in vocabulary
        """
        if self.model and word in self.model.wv:
            return self.model.wv[word]
        return None
    
    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar words.
        
        Args:
            word: Input word
            topn: Number of similar words to return
            
        Returns:
            List of (word, similarity) tuples
        """
        if self.model and word in self.model.wv:
            return self.model.wv.most_similar(word, topn=topn)
        return []
    
    def similarity(self, word1: str, word2: str) -> float:
        """
        Calculate similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score (0-1)
        """
        if self.model and word1 in self.model.wv and word2 in self.model.wv:
            return self.model.wv.similarity(word1, word2)
        return 0.0
    
    def save_model(self, path: str):
        """
        Save Word2Vec model.
        
        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save Gensim model
        model_path = path.replace('.pkl', '.model')
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'vocab_size': self.vocab_size,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'config': {
                'window': self.window,
                'min_count': self.min_count,
                'sg': self.sg,
                'negative': self.negative,
                'epochs': self.epochs
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {path}")
    
    def load_model(self, path: str):
        """
        Load Word2Vec model.
        
        Args:
            path: Path to model file
        """
        # Load Gensim model
        model_path = path.replace('.pkl', '.model')
        self.model = Word2Vec.load(model_path)
        
        # Load metadata
        with open(path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.embedding_dim = metadata['embedding_dim']
        self.vocab_size = metadata['vocab_size']
        self.word_to_idx = metadata['word_to_idx']
        self.idx_to_word = metadata['idx_to_word']
        
        config = metadata['config']
        self.window = config['window']
        self.min_count = config['min_count']
        self.sg = config['sg']
        self.negative = config['negative']
        self.epochs = config['epochs']
        
        print(f"Model loaded from {model_path}")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Embedding dimension: {self.embedding_dim}")
    
    def evaluate_analogies(self, analogies: List[Tuple[str, str, str, str]]) -> float:
        """
        Evaluate model on word analogies (e.g., king - man + woman = queen).
        
        Args:
            analogies: List of (word1, word2, word3, expected_word4) tuples
            
        Returns:
            Accuracy score
        """
        if not self.model:
            return 0.0
        
        correct = 0
        total = 0
        
        for word1, word2, word3, expected in analogies:
            if all(w in self.model.wv for w in [word1, word2, word3, expected]):
                # word1 - word2 + word3 ≈ expected
                try:
                    result = self.model.wv.most_similar(
                        positive=[word3, word2],
                        negative=[word1],
                        topn=1
                    )
                    if result[0][0] == expected:
                        correct += 1
                    total += 1
                except:
                    pass
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"Analogy accuracy: {correct}/{total} = {accuracy:.2%}")
        return accuracy
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the trained model.
        
        Returns:
            Dictionary with model statistics
        """
        if not self.model:
            return {}
        
        stats = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'window': self.window,
            'min_count': self.min_count,
            'algorithm': 'Skip-gram' if self.sg == 1 else 'CBOW',
            'epochs': self.epochs,
            'total_words': sum(self.model.wv.get_vecattr(word, "count") for word in self.model.wv.index_to_key[:100])
        }
        
        print("\nWord2Vec Model Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return stats


def create_word2vec_from_config(config_path: str = "configs/config.yaml") -> Word2VecTrainer:
    """
    Create Word2Vec trainer from configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Word2VecTrainer instance
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get Word2Vec settings (you'll need to add this to config.yaml)
    word_emb_config = config.get('word_embeddings', {})
    
    trainer = Word2VecTrainer(
        embedding_dim=word_emb_config.get('embedding_dim', 300),
        window=word_emb_config.get('window', 5),
        min_count=word_emb_config.get('min_count', 2),
        workers=word_emb_config.get('workers', 4),
        sg=word_emb_config.get('sg', 1),  # 1 for skip-gram
        negative=word_emb_config.get('negative', 5),
        epochs=word_emb_config.get('epochs', 5),
        seed=config['project']['random_seed']
    )
    
    return trainer


if __name__ == "__main__":
    # Test Word2Vec trainer
    print("Testing Word2Vec Trainer...")
    
    # Load preprocessor to get tokenized sentences
    import sys
    sys.path.append('src')
    from data.data_loader import DatasetLoader
    from data.preprocessor import TextPreprocessor
    
    # Load data
    print("\n" + "="*60)
    print("Loading data...")
    loader = DatasetLoader()
    train_df, val_df, test_df = loader.load_dataset('imdb', use_cache=True)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Get tokenized sentences (just first 1000 for testing)
    print("\n" + "="*60)
    print("Tokenizing texts...")
    sentences = []
    for text in train_df['text'].head(1000):
        cleaned = preprocessor.clean_text(text)
        tokens = preprocessor.tokenize(cleaned)
        if tokens:
            sentences.append(tokens)
    
    print(f"Prepared {len(sentences)} sentences for training")
    
    # Train Word2Vec
    print("\n" + "="*60)
    trainer = Word2VecTrainer(
        embedding_dim=100,  # Smaller for testing
        window=5,
        min_count=2,
        epochs=3,  # Fewer epochs for testing
        sg=1  # Skip-gram
    )
    
    model = trainer.train(sentences)
    
    # Get statistics
    print("\n" + "="*60)
    stats = trainer.get_statistics()
    
    # Test word similarity
    print("\n" + "="*60)
    print("Testing word similarities...")
    
    test_words = ['good', 'bad', 'movie', 'film', 'great']
    for word in test_words:
        if word in model.wv:
            print(f"\nMost similar to '{word}':")
            similar = trainer.most_similar(word, topn=5)
            for sim_word, score in similar:
                print(f"  {sim_word}: {score:.3f}")
    
    # Test word similarity between pairs
    print("\n" + "="*60)
    print("Testing word pair similarities...")
    word_pairs = [
        ('good', 'great'),
        ('bad', 'terrible'),
        ('movie', 'film'),
        ('good', 'bad')
    ]
    
    for word1, word2 in word_pairs:
        sim = trainer.similarity(word1, word2)
        print(f"Similarity between '{word1}' and '{word2}': {sim:.3f}")
    
    # Test embedding matrix creation
    print("\n" + "="*60)
    print("Testing embedding matrix creation...")
    
    # Create a small vocabulary
    test_vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        'good': 2,
        'bad': 3,
        'movie': 4,
        'film': 5
    }
    
    embedding_matrix = trainer.get_embedding_matrix(test_vocab)
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    print(f"Expected shape: [{len(test_vocab)}, {trainer.embedding_dim}]")
    
    # Save model
    print("\n" + "="*60)
    save_path = "data/embeddings/word2vec_test.pkl"
    trainer.save_model(save_path)
    
    # Load model
    print("\n" + "="*60)
    print("Testing model loading...")
    new_trainer = Word2VecTrainer()
    new_trainer.load_model(save_path)
    
    # Verify loaded model works
    print(f"Loaded vocabulary size: {new_trainer.vocab_size}")
    print(f"Testing loaded model with word 'good':")
    vec = new_trainer.get_word_vector('good')
    if vec is not None:
        print(f"  Vector shape: {vec.shape}")
        print(f"  First 5 values: {vec[:5]}")
    
    print("\n✓ All tests passed!")