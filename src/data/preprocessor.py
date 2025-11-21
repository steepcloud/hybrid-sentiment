import re
import string
from typing import List, Dict, Optional, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import os
import yaml
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """Preprocesses text data for sentiment analysis."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocess_config = self.config['preprocessing']
        self.data_config = self.config['data']
        
        # Preprocessing settings
        self.lowercase = self.preprocess_config['lowercase']
        self.remove_punctuation = self.preprocess_config['remove_punctuation']
        self.remove_stopwords = self.preprocess_config['remove_stopwords']
        self.remove_html = self.preprocess_config['remove_html']
        self.remove_urls = self.preprocess_config['remove_urls']
        self.remove_mentions = self.preprocess_config['remove_mentions']
        self.remove_hashtags = self.preprocess_config['remove_hashtags']
        self.min_word_freq = self.preprocess_config['min_word_freq']
        self.tokenizer_type = self.preprocess_config['tokenizer']
        
        # Vocabulary settings
        self.max_vocab_size = self.data_config['vocab_size']
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english')) if self.remove_stopwords else set()
        
        # Vocabulary
        self.vocab = {}
        self.idx_to_word = {}
        self.word_freq = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove HTML tags
        if self.remove_html:
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'&[a-z]+;', '', text)  # Remove HTML entities like &nbsp;
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions (for tweets)
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (for tweets)
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        else:
            # Remove just the # symbol but keep the word
            text = re.sub(r'#', '', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        text = re.sub(r'([.,!?;:\-\'\"])', r' \1 ', text)

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.tokenizer_type == 'word':
            tokens = word_tokenize(text)
        elif self.tokenizer_type == 'simple':
            # Simple whitespace tokenization
            tokens = re.findall(r"\b\w+(?:'\w+)?\b", text)
        elif self.tokenizer_type == 'char':
            # Character-level tokenization
            tokens = list(text)
        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")
        
        if self.lowercase:
            tokens = [token.lower() for token in tokens]
            
        # Remove stopwords if configured
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        # Remove empty tokens
        tokens = [token for token in tokens if token.strip() and not all(c in string.punctuation for c in token)]
        
        return tokens
    
    def build_vocab(self, texts: List[str], save_path: str = None) -> Dict[str, int]:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
            save_path: Optional path to save vocabulary
            
        Returns:
            Vocabulary dictionary mapping words to indices
        """
        print("Building vocabulary...")
        
        # Count word frequencies
        for text in texts:
            cleaned = self.clean_text(text)
            tokens = self.tokenize(cleaned)
            self.word_freq.update(tokens)
        
        # Filter by minimum frequency
        filtered_words = [
            word for word, freq in self.word_freq.items() 
            if freq >= self.min_word_freq
        ]
        
        # Sort by frequency (most common first)
        sorted_words = sorted(
            filtered_words, 
            key=lambda w: self.word_freq[w], 
            reverse=True
        )
        
        # Limit vocabulary size
        if self.max_vocab_size:
            sorted_words = sorted_words[:self.max_vocab_size - 2]  # -2 for PAD and UNK
        
        # Build vocabulary
        self.vocab = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        
        for idx, word in enumerate(sorted_words, start=2):
            self.vocab[word] = idx
        
        # Build reverse vocabulary
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Most common words: {sorted_words[:10]}")
        
        # Save vocabulary if path provided
        if save_path:
            self.save_vocab(save_path)
        
        return self.vocab
    
    def text_to_sequence(self, text: str) -> List[int]:
        """
        Convert text to sequence of integers.
        
        Args:
            text: Input text
            
        Returns:
            List of integer token IDs
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        
        # Convert tokens to indices
        sequence = [self.vocab.get(token, self.UNK_IDX) for token in tokens]
        
        return sequence
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """
        Convert sequence of integers back to text.
        
        Args:
            sequence: List of integer token IDs
            
        Returns:
            Text string
        """
        tokens = [self.idx_to_word.get(idx, self.UNK_TOKEN) for idx in sequence]
        # Remove padding tokens
        tokens = [token for token in tokens if token != self.PAD_TOKEN]
        return ' '.join(tokens)
    
    def preprocess_texts(self, texts: List[str], build_vocab: bool = False) -> List[List[int]]:
        """
        Preprocess a list of texts.
        
        Args:
            texts: List of text strings
            build_vocab: Whether to build vocabulary from these texts
            
        Returns:
            List of sequences (list of lists of integers)
        """
        if build_vocab:
            self.build_vocab(texts)
        
        sequences = [self.text_to_sequence(text) for text in texts]
        
        return sequences
    
    def pad_sequences(
        self, 
        sequences: List[List[int]], 
        max_length: int = None, 
        padding: str = 'post', 
        truncating: str = 'post'
    ) -> np.ndarray:
        """
        Pad sequences to the same length.
        
        Args:
            sequences: List of sequences
            max_length: Maximum sequence length (default from config)
            padding: 'pre' or 'post' padding
            truncating: 'pre' or 'post' truncation
            
        Returns:
            Padded sequences as numpy array
        """
        if max_length is None:
            max_length = self.data_config['max_length']
        
        padded = np.zeros((len(sequences), max_length), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                continue
            
            # Truncate if necessary
            if len(seq) > max_length:
                if truncating == 'post':
                    seq = seq[:max_length]
                else:  # 'pre'
                    seq = seq[-max_length:]
            
            # Pad if necessary
            if padding == 'post':
                padded[i, :len(seq)] = seq
            else:  # 'pre'
                padded[i, -len(seq):] = seq
        
        return padded
    
    def save_vocab(self, path: str):
        """
        Save vocabulary to file.
        
        Args:
            path: Path to save vocabulary
        """
        vocab_data = {
            'vocab': self.vocab,
            'idx_to_word': self.idx_to_word,
            'word_freq': dict(self.word_freq),
            'config': {
                'lowercase': self.lowercase,
                'remove_punctuation': self.remove_punctuation,
                'remove_stopwords': self.remove_stopwords,
                'remove_html': self.remove_html,
                'remove_urls': self.remove_urls,
                'remove_mentions': self.remove_mentions,
                'remove_hashtags': self.remove_hashtags,
                'min_word_freq': self.min_word_freq,
                'tokenizer_type': self.tokenizer_type,
                'max_vocab_size': self.max_vocab_size
            }
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"Vocabulary saved to {path}")
    
    def load_vocab(self, path: str):
        """
        Load vocabulary from file.
        
        Args:
            path: Path to vocabulary file
        """
        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.vocab = vocab_data['vocab']
        self.idx_to_word = vocab_data['idx_to_word']
        self.word_freq = Counter(vocab_data['word_freq'])
        
        # Load config (optional - can override current settings)
        config = vocab_data['config']
        self.lowercase = config.get('lowercase', self.lowercase)
        self.remove_punctuation = config.get('remove_punctuation', self.remove_punctuation)
        self.remove_stopwords = config.get('remove_stopwords', self.remove_stopwords)
        
        print(f"Vocabulary loaded from {path}")
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def get_vocab_stats(self) -> Dict:
        """
        Get statistics about the vocabulary.
        
        Returns:
            Dictionary with vocabulary statistics
        """
        stats = {
            'vocab_size': len(self.vocab),
            'total_words': sum(self.word_freq.values()),
            'unique_words': len(self.word_freq),
            'avg_word_freq': np.mean(list(self.word_freq.values())),
            'most_common': self.word_freq.most_common(10),
            'least_common': self.word_freq.most_common()[-10:] if len(self.word_freq) >= 10 else []
        }
        
        print(f"\nVocabulary Statistics:")
        print(f"  Vocabulary size: {stats['vocab_size']}")
        print(f"  Total words (including duplicates): {stats['total_words']}")
        print(f"  Unique words: {stats['unique_words']}")
        print(f"  Average word frequency: {stats['avg_word_freq']:.2f}")
        print(f"\n  Most common words:")
        for word, freq in stats['most_common']:
            print(f"    {word}: {freq}")
        
        return stats


if __name__ == "__main__":
    # Test the preprocessor
    from data_loader import DatasetLoader
    
    # Load data
    loader = DatasetLoader()
    train_df, val_df, test_df = loader.load_dataset('imdb', use_cache=True)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Test cleaning
    sample_text = train_df.iloc[0]['text']
    print("Original text:")
    print(sample_text[:200])
    print("\nCleaned text:")
    print(preprocessor.clean_text(sample_text)[:200])
    
    # Build vocabulary from training data
    print("\n" + "="*60)
    train_sequences = preprocessor.preprocess_texts(
        train_df['text'].tolist(), 
        build_vocab=True
    )
    
    # Get vocabulary statistics
    preprocessor.get_vocab_stats()
    
    # Save vocabulary
    vocab_path = "data/processed/vocab.pkl"
    preprocessor.save_vocab(vocab_path)
    
    # Test preprocessing
    print("\n" + "="*60)
    print("Sample preprocessed sequences:")
    for i in range(3):
        print(f"\nOriginal: {train_df.iloc[i]['text'][:100]}...")
        print(f"Sequence (first 20 tokens): {train_sequences[i][:20]}")
        print(f"Reconstructed: {preprocessor.sequence_to_text(train_sequences[i][:20])}")
    
    # Process validation and test sets
    print("\n" + "="*60)
    print("Processing validation and test sets...")
    val_sequences = preprocessor.preprocess_texts(val_df['text'].tolist())
    test_sequences = preprocessor.preprocess_texts(test_df['text'].tolist())
    
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val sequences: {len(val_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")
    
    # Test padding
    print("\n" + "="*60)
    print("Testing sequence padding...")
    padded_sequences = preprocessor.pad_sequences(train_sequences[:5])
    print(f"Padded shape: {padded_sequences.shape}")
    print(f"First padded sequence:\n{padded_sequences[0]}")