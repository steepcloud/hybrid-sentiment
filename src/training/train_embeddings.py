import numpy as np
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_loader import DatasetLoader
from src.data.preprocessor import TextPreprocessor
from src.models.deep_learning.word2vec_trainer import Word2VecTrainer, create_word2vec_from_config
from src.utils.helpers import set_seed, save_results
from src.utils.config import Config


class EmbeddingTrainer:
    """Trainer for word embeddings (Word2Vec, GloVe, etc.)."""
    
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        embedding_type: str = "word2vec"
    ):
        """
        Args:
            config_path: Path to configuration file
            embedding_type: Type of embedding ('word2vec', 'glove', etc.)
        """
        # load config
        self.config = Config(config_path)
        self.embedding_type = embedding_type
        
        # set random seed
        set_seed(self.config.get('project.random_seed', 42))
        
        # initialize components
        self.data_loader = DatasetLoader()
        self.preprocessor = TextPreprocessor()
        
        # set vocab size from config
        self.preprocessor.max_vocab_size = self.config.get('data.vocab_size', 20000)
        self.preprocessor.max_length = self.config.get('data.max_length', 512)
        
        self.embedding_model = None
        
        print(f"Initialized EmbeddingTrainer")
        print(f"  Embedding type: {embedding_type}")
    
    def prepare_corpus(
        self,
        texts: List[str],
        save_vocab: bool = True
    ) -> List[List[str]]:
        """
        Prepare text corpus for embedding training.
        
        Args:
            texts: List of text strings
            save_vocab: Whether to save vocabulary
            
        Returns:
            List of tokenized sentences
        """
        print("\nPreparing corpus...")
        print(f"  Total texts: {len(texts)}")
        
        # build vocabulary
        print("  Building vocabulary...")
        self.preprocessor.build_vocab(texts)
        
        if save_vocab:
            vocab_path = f"results/embeddings/{self.embedding_type}/vocab.pkl"
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            self.preprocessor.save_vocab(vocab_path)
        
        # tokenize all texts
        print("  Tokenizing texts...")
        corpus = []
        for i, text in enumerate(texts):
            cleaned = self.preprocessor.clean_text(text)
            tokens = self.preprocessor.tokenize(cleaned)
            if tokens:  # only add non-empty token lists
                corpus.append(tokens)
            
            if (i + 1) % 5000 == 0:
                print(f"    Processed {i + 1}/{len(texts)} texts")
        
        print(f"✓ Corpus prepared: {len(corpus)} documents")
        print(f"  Vocabulary size: {len(self.preprocessor.vocab)}")
        
        return corpus
    
    def train_word2vec(
        self,
        corpus: List[List[str]],
        save_model: bool = True
    ) -> Word2VecTrainer:
        """
        Train Word2Vec embeddings.
        
        Args:
            corpus: List of tokenized sentences
            save_model: Whether to save trained model
            
        Returns:
            Trained Word2VecTrainer
        """
        print("\n" + "="*60)
        print("Training Word2Vec Embeddings")
        print("="*60)
        
        # create Word2Vec model from config
        self.embedding_model = create_word2vec_from_config()
        
        # train
        self.embedding_model.train(corpus, save_path=None)
        
        # get statistics
        stats = self.embedding_model.get_statistics()
        
        print(f"\n✓ Training complete!")
        print(f"  Vocabulary size: {stats['vocab_size']}")
        print(f"  Embedding dimension: {stats['embedding_dim']}")
        
        # save model
        if save_model:
            save_dir = f"results/embeddings/{self.embedding_type}"
            os.makedirs(save_dir, exist_ok=True)
            
            model_path = os.path.join(save_dir, "word2vec.pkl")
            self.embedding_model.save_model(model_path)
        
        return self.embedding_model
    
    def evaluate_embeddings(
        self,
        test_words: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate trained embeddings.
        
        Args:
            test_words: Optional list of words to test
            
        Returns:
            Dictionary with evaluation results
        """
        print("\n" + "="*60)
        print("Evaluating Embeddings")
        print("="*60)
        
        if self.embedding_model is None:
            raise ValueError("No embedding model trained")
        
        # default test words
        if test_words is None:
            test_words = [
                'good', 'bad', 'excellent', 'terrible', 'love', 'hate',
                'happy', 'sad', 'amazing', 'awful', 'best', 'worst'
            ]
        
        results = {}
        
        # test word similarities
        print("\nWord Similarities:")
        for word in test_words:
            similar = self.embedding_model.most_similar(word, topn=5)
            if similar:
                print(f"\n  '{word}' similar to:")
                for similar_word, score in similar:
                    print(f"    {similar_word}: {score:.4f}")
                results[word] = similar
            else:
                print(f"\n  '{word}' not in vocabulary")
                results[word] = []
        
        # test word similarities between pairs
        print("\n" + "-"*60)
        print("Word Pair Similarities:")
        
        word_pairs = [
            ('good', 'great'),
            ('bad', 'terrible'),
            ('love', 'like'),
            ('hate', 'dislike'),
            ('good', 'bad'),
            ('love', 'hate')
        ]
        
        for word1, word2 in word_pairs:
            sim = self.embedding_model.similarity(word1, word2)
            print(f"  {word1} <-> {word2}: {sim:.4f}")
        
        # test sentiment-related words
        print("\n" + "-"*60)
        print("Sentiment Word Clusters:")
        
        sentiment_words = {
            'positive': ['good', 'great', 'excellent', 'amazing', 'wonderful'],
            'negative': ['bad', 'terrible', 'awful', 'horrible', 'worst']
        }
        
        for sentiment, words in sentiment_words.items():
            print(f"\n  {sentiment.upper()} words:")
            available_words = [w for w in words if self.embedding_model.get_word_vector(w) is not None]
            if len(available_words) >= 2:
                # compute average pairwise similarity
                similarities = []
                for i, w1 in enumerate(available_words):
                    for w2 in available_words[i+1:]:
                        sim = self.embedding_model.similarity(w1, w2)
                        similarities.append(sim)
                        print(f"    {w1} <-> {w2}: {sim:.4f}")
                
                if similarities:
                    avg_sim = np.mean(similarities)
                    print(f"    Average similarity: {avg_sim:.4f}")
                    results[f'{sentiment}_cluster_similarity'] = avg_sim
        
        return results
    
    def train_on_dataset(
        self,
        dataset: str = 'imdb',
        save_model: bool = True,
        evaluate: bool = True
    ) -> Dict:
        """
        Train embeddings on a dataset.
        
        Args:
            dataset: Dataset to use ('imdb' or 'twitter')
            save_model: Whether to save trained model
            evaluate: Whether to evaluate embeddings
            
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print(f"Training Embeddings on {dataset.upper()}")
        print(f"{'='*60}")
        
        # load data
        print("\nLoading dataset...")
        train_df, val_df, test_df = self.data_loader.load_dataset(dataset, use_cache=True)
        
        # combine all texts for training embeddings
        all_texts = (
            train_df['text'].tolist() +
            val_df['text'].tolist() +
            test_df['text'].tolist()
        )
        
        print(f"Total texts for training: {len(all_texts)}")
        
        # prepare corpus
        corpus = self.prepare_corpus(all_texts, save_vocab=True)
        
        # train embeddings based on type
        if self.embedding_type == 'word2vec':
            self.train_word2vec(corpus, save_model=save_model)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
        
        # evaluate
        results = {}
        if evaluate:
            eval_results = self.evaluate_embeddings()
            results['evaluation'] = eval_results
        
        # save metadata
        results['metadata'] = {
            'dataset': dataset,
            'embedding_type': self.embedding_type,
            'corpus_size': len(corpus),
            'vocab_size': len(self.preprocessor.vocab),
            'embedding_dim': self.embedding_model.embedding_dim if self.embedding_model else None,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # save results
        results_path = f"results/embeddings/{self.embedding_type}/training_results.pkl"
        save_results(results, results_path)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Train word embeddings')
    parser.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'twitter'],
                        help='Dataset to use')
    parser.add_argument('--embedding', type=str, default='word2vec', choices=['word2vec'],
                        help='Embedding type')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save trained model')
    parser.add_argument('--no-eval', action='store_true',
                        help='Do not evaluate embeddings')
    
    args = parser.parse_args()
    
    # create trainer
    trainer = EmbeddingTrainer(
        config_path=args.config,
        embedding_type=args.embedding
    )
    
    # train embeddings
    results = trainer.train_on_dataset(
        dataset=args.dataset,
        save_model=not args.no_save,
        evaluate=not args.no_eval
    )
    
    print("\n✓ Training complete!")
    print(f"\nMetadata:")
    for key, value in results['metadata'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()