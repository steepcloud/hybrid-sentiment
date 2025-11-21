import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import yaml

from src.data.preprocessor import TextPreprocessor
from src.models.deep_learning.lstm_encoder import LSTMClassifier
from src.models.deep_learning.gru_encoder import GRUClassifier
from src.models.deep_learning.transformer_encoder import TransformerClassifier


class HybridSentimentPredictor:
    """
    Predictor using trained encoder + classifier.
    Supports: LSTM/GRU/Transformer + Logistic/RF/XGBoost
    """
    
    def __init__(
        self,
        encoder_path: str,
        classifier_path: str,
        config_path: str = 'configs/config.yaml'
    ):
        """
        Initialize predictor with trained models.
        
        Args:
            encoder_path: Path to trained encoder (.pt file)
            classifier_path: Path to trained classifier (.pkl file)
            config_path: Path to config file
        """
        self.encoder_path = Path(encoder_path)
        self.classifier_path = Path(classifier_path)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(config_path=config_path)
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load models
        self._load_encoder()
        self._load_classifier()
        
        print(f"✓ Models loaded successfully!")
        print(f"  Encoder: {self.encoder_path.name}")
        print(f"  Classifier: {self.classifier_path.name}")
    
    def _load_encoder(self):
        """Load the encoder model (LSTM/GRU/Transformer)."""
        print(f"Loading encoder from {self.encoder_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(self.encoder_path, map_location=self.device)
        
        # Infer model type from file path if not in checkpoint
        model_type = checkpoint.get('model_type')
        if not model_type:
            path_lower = str(self.encoder_path).lower()
            if 'gru' in path_lower:
                model_type = 'gru'
            elif 'transformer' in path_lower:
                model_type = 'transformer'
            elif 'lstm' in path_lower:
                model_type = 'lstm'
            else:
                raise ValueError(f"Cannot determine model type from path: {self.encoder_path}")
        
        print(f"  Model type: {model_type}")

        # Get config from checkpoint or use defaults
        model_config = checkpoint.get('config', {})

        # If config is empty, infer from model_state_dict
        if not model_config:
            state_dict = checkpoint['model_state_dict']
            embedding_key = 'encoder.embedding.weight'
            if embedding_key in state_dict:
                vocab_size, embedding_dim = state_dict[embedding_key].shape
                if model_type in ['lstm', 'gru']:
                    rnn_key = f'encoder.{model_type}.weight_hh_l0'
                    if rnn_key in state_dict:
                        hidden_dim = state_dict[rnn_key].shape[0] // (4 if model_type == 'lstm' else 3)
                    else:
                        hidden_dim = 256
                else:
                    hidden_dim = 256
                
                layer_keys = [k for k in state_dict.keys() if f'encoder.{model_type}.weight_hh_l' in k and '_reverse' not in k]
                num_layers = len(layer_keys)
                
                model_config = {
                    'vocab_size': vocab_size,
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'dropout': 0.5
                }
                print(f"  Inferred config: {model_config}")
        
        # Initialize encoder
        if model_type == 'lstm':
            self.encoder = LSTMClassifier(
                vocab_size=model_config['vocab_size'],
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                dropout=model_config.get('dropout', 0.5)
            )
        elif model_type == 'gru':
            self.encoder = GRUClassifier(
                vocab_size=model_config['vocab_size'],
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                dropout=model_config.get('dropout', 0.5)
            )
        elif model_type == 'transformer':
            self.encoder = TransformerClassifier(
                vocab_size=model_config['vocab_size'],
                embedding_dim=model_config['embedding_dim'],
                num_heads=model_config.get('num_heads', 8),
                num_layers=model_config['num_layers'],
                dropout=model_config.get('dropout', 0.1)
            )
        
        # Load weights
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # Store vocab (if available)
        self.vocab = checkpoint.get('vocab', {})
        self.word_to_idx = checkpoint.get('word_to_idx', self.vocab)

        # If no vocab in checkpoint, load from embeddings folder
        if not self.word_to_idx:
            print("  [!] No vocab in checkpoint, loading from embeddings...")
            if 'imdb' in str(self.encoder_path).lower():
                vocab_path = Path('results/embeddings/imdb/word2vec/vocab.pkl')
            elif 'twitter' in str(self.encoder_path).lower():
                vocab_path = Path('results/embeddings/twitter/word2vec/vocab.pkl')
            else:
                raise ValueError("Cannot determine dataset from encoder path")
            
            if vocab_path.exists():
                with open(vocab_path, 'rb') as f:
                    vocab_data = pickle.load(f)
                    
                    # FIX: vocab_data['vocab'] IS the word_to_idx mapping!
                    if isinstance(vocab_data, dict) and 'vocab' in vocab_data:
                        self.word_to_idx = vocab_data['vocab']
                    else:
                        raise ValueError(f"Unexpected vocab format in {vocab_path}")
                        
                print(f"  ✓ Loaded vocab from {vocab_path}")
            else:
                raise ValueError(f"Vocab file not found: {vocab_path}")
        
        print(f"  ✓ Encoder loaded: {model_type}")
        print(f"  Vocab size: {len(self.word_to_idx)}")
    
    def _load_classifier(self):
        """Load the classical ML classifier."""
        print(f"Loading classifier from {self.classifier_path}...")
        
        classifier_name = self.classifier_path.stem.lower()
        
        # Load with joblib (all classifiers use this format)
        try:
            import joblib
            model_data = joblib.load(self.classifier_path)
            
            # Extract the actual model from the dict
            if isinstance(model_data, dict) and 'model' in model_data:
                self.classifier = model_data['model']
            else:
                # Fallback: assume it's the model directly
                self.classifier = model_data
            
            print(f"  ✓ Classifier loaded: {type(self.classifier).__name__}")
            
        except Exception as e:
            raise ValueError(f"Failed to load classifier from {self.classifier_path}: {e}")
    
    def _text_to_indices(self, text: str, max_len: int = 256) -> torch.Tensor:
        """
        Convert text to indices.
        
        Args:
            text: Input text
            max_len: Maximum sequence length
            
        Returns:
            Tensor of shape (1, max_len)
        """
        # Preprocess text
        tokens = self.preprocessor.tokenize(text)
        
        # Convert to indices
        indices = [self.word_to_idx.get(token, self.word_to_idx.get('<UNK>', 1)) 
                   for token in tokens[:max_len]]
        
        # Pad or truncate
        if len(indices) < max_len:
            indices += [self.word_to_idx.get('<PAD>', 0)] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        
        return torch.tensor([indices], dtype=torch.long)
    
    def _extract_features(self, text: str) -> np.ndarray:
        """
        Extract features using encoder.
        
        Args:
            text: Input text
            
        Returns:
            Feature vector (numpy array)
        """
        # Convert text to indices
        indices = self._text_to_indices(text).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.encoder.get_embeddings(indices)
        
        return features.cpu().numpy().flatten()
    
    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Dict with keys:
                - sentiment: 'positive' or 'negative'
                - confidence: float (0-1)
                - probabilities: [neg_prob, pos_prob]
                - processing_time_ms: float
        """
        start_time = time.time()
        
        print(f"\n🔍 INFERENCE DEBUG:")
        print(f"   Input text: {text}")
        
        # Extract features
        features = self._extract_features(text)
        print(f"   Features shape: {features.shape}")
        print(f"   Features sample (first 10): {features[:10]}")
        
        # Predict with classifier
        prediction = self.classifier.predict([features])[0]
        print(f"   Raw prediction: {prediction} (0=negative, 1=positive)")
        
        # Get probabilities if available
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba([features])[0]
        else:
            # For models without predict_proba, use binary prediction
            probabilities = [1 - prediction, prediction]
        
        print(f"   Probabilities: {probabilities}")
        print(f"   Probability[0] (negative): {probabilities[0]:.4f}")
        print(f"   Probability[1] (positive): {probabilities[1]:.4f}")
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        sentiment = 'positive' if prediction == 1 else 'negative'
        confidence = max(probabilities)
        
        print(f"   Final sentiment: {sentiment}")
        print(f"   Final confidence: {confidence:.4f}\n")

        return {
            'sentiment': sentiment,
            'confidence': float(confidence),
            'probabilities': [float(p) for p in probabilities],
            'processing_time_ms': processing_time
        }
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dicts
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Extract features for batch
            batch_features = []
            for text in batch_texts:
                features = self._extract_features(text)
                batch_features.append(features)
            
            batch_features = np.array(batch_features)
            
            # Predict
            predictions = self.classifier.predict(batch_features)
            
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(batch_features)
            else:
                probabilities = [[1 - p, p] for p in predictions]
            
            # Format results
            for pred, probs in zip(predictions, probabilities):
                sentiment = 'positive' if pred == 1 else 'negative'
                confidence = max(probs)
                
                results.append({
                    'sentiment': sentiment,
                    'confidence': float(confidence),
                    'probabilities': [float(p) for p in probs]
                })
        
        return results


class EndToEndPredictor:
    """
    Predictor using end-to-end deep learning model.
    Supports: LSTM/GRU/Transformer trained end-to-end.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str = 'configs/config.yaml'
    ):
        """
        Initialize end-to-end predictor.
        
        Args:
            model_path: Path to trained model (.pt file)
            config_path: Path to config file
        """
        self.model_path = Path(model_path)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(config_path=config_path)
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self._load_model()
        
        print(f"✓ End-to-end model loaded: {self.model_path.name}")
    
    def _load_model(self):
        """Load the end-to-end model."""
        print(f"Loading end-to-end model from {self.model_path}...")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Infer model type from file path if not in checkpoint
        model_type = checkpoint.get('model_type')
        if not model_type:
            path_lower = str(self.model_path).lower()
            if 'gru' in path_lower:
                model_type = 'gru'
            elif 'transformer' in path_lower:
                model_type = 'transformer'
            elif 'lstm' in path_lower:
                model_type = 'lstm'
            else:
                raise ValueError(f"Cannot determine model type from path: {self.model_path}")
        
        print(f"  Model type: {model_type}")
        
        # Get config from checkpoint or infer it
        model_config = checkpoint.get('config', {})
        
        # If config is empty, infer from model_state_dict (same logic as HybridSentimentPredictor)
        if not model_config:
            state_dict = checkpoint['model_state_dict']
            embedding_key = 'encoder.embedding.weight'
            if embedding_key in state_dict:
                vocab_size, embedding_dim = state_dict[embedding_key].shape
                
                if model_type in ['lstm', 'gru']:
                    rnn_key = f'encoder.{model_type}.weight_hh_l0'
                    if rnn_key in state_dict:
                        hidden_dim = state_dict[rnn_key].shape[0] // (4 if model_type == 'lstm' else 3)
                    else:
                        hidden_dim = 256
                else:
                    hidden_dim = 256
                
                layer_keys = [k for k in state_dict.keys() if f'encoder.{model_type}.weight_hh_l' in k and '_reverse' not in k]
                num_layers = len(layer_keys)
                
                model_config = {
                    'vocab_size': vocab_size,
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'dropout': 0.5
                }
                print(f"  Inferred config: {model_config}")
        
        # Initialize model
        if model_type == 'lstm':
            self.model = LSTMClassifier(
                vocab_size=model_config['vocab_size'],
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                dropout=model_config.get('dropout', 0.5)
            )
        elif model_type == 'gru':
            self.model = GRUClassifier(
                vocab_size=model_config['vocab_size'],
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                dropout=model_config.get('dropout', 0.5)
            )
        elif model_type == 'transformer':
            self.model = TransformerClassifier(
                vocab_size=model_config['vocab_size'],
                embedding_dim=model_config['embedding_dim'],
                num_heads=model_config.get('num_heads', 8),
                num_layers=model_config['num_layers'],
                dropout=model_config.get('dropout', 0.1)
            )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load vocab (same logic as HybridSentimentPredictor)
        self.vocab = checkpoint.get('vocab', {})
        self.word_to_idx = checkpoint.get('word_to_idx', self.vocab)
        
        if not self.word_to_idx:
            print("  [!] No vocab in checkpoint, loading from embeddings...")
            if 'imdb' in str(self.model_path).lower():
                vocab_path = Path('results/embeddings/imdb/word2vec/vocab.pkl')
            elif 'twitter' in str(self.model_path).lower():
                vocab_path = Path('results/embeddings/twitter/word2vec/vocab.pkl')
            else:
                raise ValueError("Cannot determine dataset from model path")
            
            if vocab_path.exists():
                with open(vocab_path, 'rb') as f:
                    vocab_data = pickle.load(f)
                    
                    if isinstance(vocab_data, dict) and 'vocab' in vocab_data:
                        self.word_to_idx = vocab_data['vocab']
                    else:
                        raise ValueError(f"Unexpected vocab format in {vocab_path}")
                
                print(f"  ✓ Loaded vocab from {vocab_path}")
            else:
                raise ValueError(f"Vocab file not found: {vocab_path}")
        
        print(f"  ✓ Model loaded: {model_type}")
        print(f"  Vocab size: {len(self.word_to_idx)}")
    
    def _text_to_indices(self, text: str, max_len: int = 256) -> torch.Tensor:
        """Convert text to indices."""
        tokens = self.preprocessor.tokenize(text)
        
        print(f"🔍 TOKENIZATION DEBUG:")
        print(f"   Raw text: {text}")
        print(f"   Tokens: {tokens[:20]}")
        
        indices = [self.word_to_idx.get(token, self.word_to_idx.get('<UNK>', 1)) 
                   for token in tokens[:max_len]]
        
        print(f"   Indices (first 20): {indices[:20]}")
        print(f"   Vocab check - 'love': {self.word_to_idx.get('love', 'NOT FOUND')}")
        print(f"   Vocab check - 'trash': {self.word_to_idx.get('trash', 'NOT FOUND')}")
        print(f"   Vocab check - '<PAD>': {self.word_to_idx.get('<PAD>', 'NOT FOUND')}")
        print(f"   Vocab check - '<UNK>': {self.word_to_idx.get('<UNK>', 'NOT FOUND')}\n")
        
        if len(indices) < max_len:
            indices += [self.word_to_idx.get('<PAD>', 0)] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        
        return torch.tensor([indices], dtype=torch.long)
    
    def predict(self, text: str) -> Dict:
        """Predict sentiment for a single text."""
        start_time = time.time()
        
        # Convert to indices
        indices = self._text_to_indices(text).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(indices)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probabilities)
        
        processing_time = (time.time() - start_time) * 1000

        # DEBUG: Print raw values
        print(f"\n🔍 END-TO-END DEBUG:")
        print(f"   Text: {text}")
        print(f"   Logits: {logits.cpu().numpy()[0]}")
        print(f"   Probabilities: {probabilities}")
        print(f"   Prediction (argmax): {prediction}")
        print(f"   Prob[0]: {probabilities[0]:.4f}, Prob[1]: {probabilities[1]:.4f}\n")
        
        sentiment = 'positive' if prediction == 1 else 'negative'
        confidence = float(probabilities[prediction])
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': [float(p) for p in probabilities],
            'processing_time_ms': processing_time
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment for multiple texts."""
        results = []
        
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Test predictor
    print("Testing Hybrid Sentiment Predictor")
    print("=" * 60)
    
    # Example usage
    predictor = HybridSentimentPredictor(
        encoder_path='results/models/deep_learning/imdb/lstm/lstm_best.pt',
        classifier_path='results/models/classical_ml/imdb/lstm/xgboost.pkl'
    )
    
    test_texts = [
        "This movie is absolutely amazing! I loved every minute.",
        "Terrible waste of time. Don't watch this garbage.",
        "It's okay, nothing special."
    ]
    
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']} ({result['confidence']:.1%})")