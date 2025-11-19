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
        
        # Get model config
        model_config = checkpoint.get('config', {})
        model_type = checkpoint.get('model_type', 'lstm')
        
        # Initialize encoder
        if model_type == 'lstm':
            from src.models.deep_learning.lstm_encoder import create_lstm_classifier_from_config
            self.encoder = create_lstm_classifier_from_config(model_config)
        elif model_type == 'gru':
            from src.models.deep_learning.gru_encoder import create_gru_classifier_from_config
            self.encoder = create_gru_classifier_from_config(model_config)
        elif model_type == 'transformer':
            from src.models.deep_learning.transformer_encoder import create_transformer_classifier_from_config
            self.encoder = create_transformer_classifier_from_config(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # Store vocab
        self.vocab = checkpoint.get('vocab', {})
        self.word_to_idx = checkpoint.get('word_to_idx', self.vocab)
        
        print(f"  ✓ Encoder loaded: {model_type}")
        print(f"  Vocab size: {len(self.word_to_idx)}")
    
    def _load_classifier(self):
        """Load the classical ML classifier."""
        print(f"Loading classifier from {self.classifier_path}...")
        
        with open(self.classifier_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        classifier_name = type(self.classifier).__name__
        print(f"  ✓ Classifier loaded: {classifier_name}")
    
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
        tokens = self.preprocessor.preprocess(text)
        
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
            features = self.encoder.encode(indices)
        
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
        
        # Extract features
        features = self._extract_features(text)
        
        # Predict with classifier
        prediction = self.classifier.predict([features])[0]
        
        # Get probabilities if available
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba([features])[0]
        else:
            # For models without predict_proba, use binary prediction
            probabilities = [1 - prediction, prediction]
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        sentiment = 'positive' if prediction == 1 else 'negative'
        confidence = max(probabilities)
        
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
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        model_config = checkpoint.get('config', {})
        model_type = checkpoint.get('model_type', 'lstm')
        
        # Initialize model
        if model_type == 'lstm':
            from src.models.deep_learning.lstm_encoder import create_lstm_classifier_from_config
            self.model = create_lstm_classifier_from_config(model_config)
        elif model_type == 'gru':
            from src.models.deep_learning.gru_encoder import create_gru_classifier_from_config
            self.model = create_gru_classifier_from_config(model_config)
        elif model_type == 'transformer':
            from src.models.deep_learning.transformer_encoder import create_transformer_classifier_from_config
            self.model = create_transformer_classifier_from_config(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Store vocab
        self.vocab = checkpoint.get('vocab', {})
        self.word_to_idx = checkpoint.get('word_to_idx', self.vocab)
    
    def _text_to_indices(self, text: str, max_len: int = 256) -> torch.Tensor:
        """Convert text to indices."""
        tokens = self.preprocessor.preprocess(text)
        
        indices = [self.word_to_idx.get(token, self.word_to_idx.get('<UNK>', 1)) 
                   for token in tokens[:max_len]]
        
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