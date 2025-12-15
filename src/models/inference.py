import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
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
        
        # load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # initialize preprocessor
        self.preprocessor = TextPreprocessor(config_path=config_path)
        
        # determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # load models
        self._load_encoder()
        self._load_classifier()
        
        print(f"✓ Models loaded successfully!")
        print(f"  Encoder: {self.encoder_path.name}")
        print(f"  Classifier: {self.classifier_path.name}")
    
    def _load_encoder(self):
        """Load the encoder model (LSTM/GRU/Transformer/BERT/ROBERTA/DISTILBERT)."""
        print(f"Loading encoder from {self.encoder_path}...")
        
        # load checkpoint
        checkpoint = torch.load(self.encoder_path, map_location=self.device)
        
        # infer model type from file path if not in checkpoint
        model_type = checkpoint.get('model_type')
        if not model_type:
            path_lower = str(self.encoder_path).lower()
            if 'distilbert' in path_lower:
                model_type = 'distilbert'
            elif 'roberta' in path_lower:
                model_type = 'roberta'
            elif 'bert' in path_lower:
                model_type = 'bert'
            elif 'gru' in path_lower:
                model_type = 'gru'
            elif 'transformer' in path_lower:
                model_type = 'transformer'
            elif 'lstm' in path_lower:
                model_type = 'lstm'
            else:
                raise ValueError(f"Cannot determine model type from path: {self.encoder_path}")
        
        print(f"  Model type: {model_type}")

        # get config from checkpoint or use defaults
        model_config = checkpoint.get('config', {})

        # if config is empty, infer from model_state_dict
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
                
                    layer_keys = [k for k in state_dict.keys() if f'encoder.{model_type}.weight_hh_l' in k and '_reverse' not in k]
                    num_layers = len(layer_keys)
                
                    model_config = {
                        'vocab_size': vocab_size,
                        'embedding_dim': embedding_dim,
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers,
                        'dropout': 0.5
                    }

                elif model_type == 'transformer':
                    hidden_dim = 256  # Default
            
                    # infer num_layers from transformer layers
                    layer_keys = [k for k in state_dict.keys() if 'encoder.transformer_encoder.layers.' in k and 'self_attn.in_proj_weight' in k]
                    num_layers = len(set(k.split('.')[3] for k in layer_keys)) if layer_keys else 3
                    
                    # infer max_len from pos_encoder
                    pos_encoder_key = 'encoder.pos_encoder.pe'
                    max_len = state_dict[pos_encoder_key].shape[1] if pos_encoder_key in state_dict else 512
                    
                    model_config = {
                        'vocab_size': vocab_size,
                        'embedding_dim': embedding_dim,
                        'num_heads': 4,  # Fixed to match training
                        'num_layers': num_layers,
                        'max_len': max_len,  # Added
                        'dropout': 0.1
                    }

                print(f"  Inferred config: {model_config}")

        # initialize encoder
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
                num_heads=model_config.get('num_heads', 4),
                num_layers=model_config['num_layers'],
                max_seq_length=model_config.get('max_len', 512),
                dropout=model_config.get('dropout', 0.1)
            )
        elif model_type in ['bert', 'roberta', 'distilbert']:
            from src.models.deep_learning.bert_encoder import BERTClassifier
            model_name_map = {
                'bert': 'bert-base-uncased',
                'roberta': 'roberta-base',
                'distilbert': 'distilbert-base-uncased'
            }
            model_name = model_name_map[model_type]
            self.encoder = BERTClassifier(model_name=model_name)
        
        # load weights
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # store vocab (if available)
        self.vocab = checkpoint.get('vocab', {})
        self.word_to_idx = checkpoint.get('word_to_idx', self.vocab)

        # if no vocab in checkpoint, load from embeddings folder
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
        
        # load with joblib (all classifiers use this format)
        try:
            import joblib
            model_data = joblib.load(self.classifier_path)
            
            # extract the actual model from the dict
            if isinstance(model_data, dict) and 'model' in model_data:
                self.classifier = model_data['model']
            else:
                # fallback: assume it's the model directly
                self.classifier = model_data
            
            print(f"  ✓ Classifier loaded: {type(self.classifier).__name__}")
            
        except Exception as e:
            raise ValueError(f"Failed to load classifier from {self.classifier_path}: {e}")
    
    def _text_to_indices(self, text: str, max_len: int = 256) -> Tuple[torch.Tensor, int]:
        """
        Convert text to indices.
        """
        # preprocess text
        tokens = self.preprocessor.tokenize(text)
        
        # convert to indices
        indices = [self.word_to_idx.get(token, self.word_to_idx.get('<UNK>', 1)) 
                   for token in tokens[:max_len]]
        
        actual_length = len(indices)
        
        # pad or truncate
        if len(indices) < max_len:
            indices += [self.word_to_idx.get('<PAD>', 0)] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        
        return torch.tensor([indices], dtype=torch.long), actual_length
    
    def _extract_features(self, text: str) -> np.ndarray:
        """
        Extract features using encoder.
        
        Args:
            text: Input text
            
        Returns:
            Feature vector (numpy array)
        """
        if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'tokenizer'):
            # BERT tokenization
            tokenizer = self.encoder.encoder.tokenizer
            encodings = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # extract features
            with torch.no_grad():
                features = self.encoder.get_embeddings(input_ids, attention_mask)
            
            return features.cpu().numpy().flatten()
        else:
            # convert text to indices
            indices, actual_length = self._text_to_indices(text)
            
            if indices.dim() == 1:
                # this is the correct action if _text_to_indices returns 1D [L]
                indices = indices.unsqueeze(0).to(self.device)
            elif indices.dim() == 2:
                indices = indices.to(self.device)

            # extract features
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
        
        # extract features
        features = self._extract_features(text)
        print(f"   Features shape: {features.shape}")
        print(f"   Features sample (first 10): {features[:10]}")
        
        # predict with classifier
        prediction = self.classifier.predict([features])[0]
        print(f"   Raw prediction: {prediction} (0=negative, 1=positive)")
        
        # get probabilities if available
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba([features])[0]
        else:
            # for models without predict_proba, use binary prediction
            probabilities = [1 - prediction, prediction]
        
        print(f"   Probabilities: {probabilities}")
        print(f"   Probability[0] (negative): {probabilities[0]:.4f}")
        print(f"   Probability[1] (positive): {probabilities[1]:.4f}")
        
        is_bert = hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'tokenizer')
    
        if is_bert:
            # BERT: 0=positive, 1=negative
            sentiment = 'positive' if prediction == 1 else 'negative'
            # swap probabilities to [neg, pos] format
            probabilities_swapped = [float(probabilities[1]), float(probabilities[0])]
        else:
            # Traditional: 0=negative, 1=positive
            sentiment = 'positive' if prediction == 0 else 'negative'
            probabilities_swapped = [float(probabilities[0]), float(probabilities[1])]
        
        processing_time = (time.time() - start_time) * 1000  # convert to ms

        confidence = float(max(probabilities_swapped))
        
        print(f"   Encoder type: {'BERT' if is_bert else 'Traditional'}")
        print(f"   Final sentiment: {sentiment}")
        print(f"   Final confidence: {confidence:.4f}\n")

        return {
            'sentiment': sentiment,
            'confidence': float(confidence),
            'probabilities': [float(p) for p in probabilities_swapped],
            'processing_time_ms': float(processing_time)
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
            
            # extract features for batch
            batch_features = []
            for text in batch_texts:
                features = self._extract_features(text)
                batch_features.append(features)
            
            batch_features = np.array(batch_features)
            
            # predict
            predictions = self.classifier.predict(batch_features)
            
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(batch_features)
            else:
                probabilities = [[1 - p, p] for p in predictions]
            
            # format results
            for pred, probs in zip(predictions, probabilities):
                sentiment = 'positive' if pred == 0 else 'negative'
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
        
        #load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # initialize preprocessor
        self.preprocessor = TextPreprocessor(config_path=config_path)
        
        # determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # load model
        self._load_model()
        
        print(f"✓ End-to-end model loaded: {self.model_path.name}")
    
    def _load_model(self):
        """Load the end-to-end model."""
        print(f"Loading end-to-end model from {self.model_path}...")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # infer model type from file path if not in checkpoint
        model_type = checkpoint.get('model_type')
        if not model_type:
            path_lower = str(self.model_path).lower()
            if 'distilbert' in path_lower:
                model_type = 'distilbert'
            elif 'roberta' in path_lower:
                model_type = 'roberta'
            elif 'bert' in path_lower:
                model_type = 'bert'
            elif 'gru' in path_lower:
                model_type = 'gru'
            elif 'transformer' in path_lower:
                model_type = 'transformer'
            elif 'lstm' in path_lower:
                model_type = 'lstm'
            else:
                raise ValueError(f"Cannot determine model type from path: {self.model_path}")
        
        print(f"  Model type: {model_type}")
        
        # get config from checkpoint or infer it
        model_config = checkpoint.get('config', {})
        
        # if config is empty, infer from model_state_dict (same logic as HybridSentimentPredictor)
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
        
        # initialize model
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
        elif model_type in ['bert', 'roberta', 'distilbert']:
            # load BERT-based model
            from src.models.deep_learning.bert_encoder import BERTClassifier
            model_name_map = {
                'bert': 'bert-base-uncased',
                'roberta': 'roberta-base',
                'distilbert': 'distilbert-base-uncased'
            }
            model_name = model_name_map[model_type]
            self.model = BERTClassifier(model_name=model_name)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # load vocab (same logic as HybridSentimentPredictor)
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
    
    def _text_to_indices(self, text: str, max_len: int = 256) -> Tuple[torch.Tensor, int]:
        """Convert text to indices."""
        tokens = self.preprocessor.tokenize(text)
        
        print(f"🔍 TOKENIZATION DEBUG:")
        print(f"   Raw text: {text}")
        print(f"   Tokens: {tokens[:20]}")
        
        indices = [self.word_to_idx.get(token, self.word_to_idx.get('<UNK>', 1)) 
                   for token in tokens[:max_len]]
        
        actual_length = len(indices)
        
        print(f"   Indices (first 20): {indices[:20]}")
        print(f"   Vocab check - 'love': {self.word_to_idx.get('love', 'NOT FOUND')}")
        print(f"   Vocab check - 'trash': {self.word_to_idx.get('trash', 'NOT FOUND')}")
        print(f"   Vocab check - '<PAD>': {self.word_to_idx.get('<PAD>', 'NOT FOUND')}")
        print(f"   Vocab check - '<UNK>': {self.word_to_idx.get('<UNK>', 'NOT FOUND')}\n")
        
        if len(indices) < max_len:
            indices += [self.word_to_idx.get('<PAD>', 0)] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        
        return torch.tensor([indices], dtype=torch.long), actual_length
    
    def predict(self, text: str) -> Dict:
        """Predict sentiment for a single text."""
        start_time = time.time()

        model_type = None
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'tokenizer'):
            model_type = 'bert' # BERT/RoBERTa/DistilBERT
        
        if model_type == 'bert':
            # BERT tokenization
            tokenizer = self.model.encoder.tokenizer
            encodings = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # predict
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                prediction = np.argmax(probabilities)
            
            sentiment = 'positive' if prediction == 1 else 'negative'
            confidence = float(probabilities[prediction])
            # swap probabilities to match traditional format [neg, pos]
            probabilities_swapped = [float(probabilities[1]), float(probabilities[0])]
        else:
            # convert to indices
            indices, actual_length = self._text_to_indices(text)
            indices = indices.to(self.device)

            sequence_length = torch.tensor([actual_length], dtype=torch.long).to(self.device)
            
            # predict
            with torch.no_grad():
                logits = self.model(indices, sequence_length)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                prediction = np.argmax(probabilities)
            
            sentiment = 'positive' if prediction == 0 else 'negative'
            confidence = float(probabilities[prediction])
            probabilities_swapped = [float(probabilities[0]), float(probabilities[1])]  # Already [neg, pos]
        
        processing_time = (time.time() - start_time) * 1000

        # DEBUG: Print raw values
        print(f"\nEND-TO-END DEBUG:")
        print(f"   Text: {text}")
        print(f"   Logits: {logits.cpu().numpy()[0]}")
        print(f"   Probabilities: {probabilities}")
        print(f"   Prediction (argmax): {prediction}")
        print(f"   Prob[0]: {probabilities[0]:.4f}, Prob[1]: {probabilities[1]:.4f}\n")
        
        return {
            'sentiment': sentiment,
            'confidence': float(confidence),
            'probabilities': probabilities_swapped,
            'processing_time_ms': float(processing_time)
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