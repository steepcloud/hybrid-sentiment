import numpy as np
import os
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Tuple
import joblib


class LogisticRegressionClassifier:
    """Logistic Regression classifier for sentiment analysis using embeddings."""
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = 'lbfgs',
        penalty: str = 'l2',
        class_weight: Optional[str] = None,
        random_state: int = 42,
        normalize: bool = True
    ):
        """
        Args:
            C: Inverse of regularization strength (smaller = stronger regularization)
            max_iter: Maximum number of iterations
            solver: Algorithm to use ('lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga')
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
            class_weight: Weights for classes ('balanced' or None)
            random_state: Random seed
            normalize: Whether to normalize features with StandardScaler
        """
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.penalty = penalty
        self.class_weight = class_weight
        self.random_state = random_state
        self.normalize = normalize
        
        # initialize model
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            penalty=penalty,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1  # use all CPU cores
        )
        
        # scaler for normalization
        self.scaler = StandardScaler() if normalize else None
        
        # training info
        self.is_trained = False
        self.feature_dim = None
        self.classes = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train the logistic regression model.
        
        Args:
            X_train: Training embeddings [n_samples, embedding_dim]
            y_train: Training labels [n_samples]
            X_val: Optional validation embeddings
            y_val: Optional validation labels
            
        Returns:
            Dictionary with training metrics
        """
        print("Training Logistic Regression...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Feature dimension: {X_train.shape[1]}")
        print(f"  C (regularization): {self.C}")
        print(f"  Penalty: {self.penalty}")
        print(f"  Solver: {self.solver}")
        print(f"  Normalize: {self.normalize}")
        
        self.feature_dim = X_train.shape[1]
        
        # normalize features if specified
        if self.normalize:
            X_train = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val = self.scaler.transform(X_val)
        
        # train model
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        self.classes = self.model.classes_
        
        # calculate metrics
        metrics = {
            'train_accuracy': self.model.score(X_train, y_train),
            'n_iterations': self.model.n_iter_[0] if hasattr(self.model, 'n_iter_') else None
        }
        
        if X_val is not None and y_val is not None:
            metrics['val_accuracy'] = self.model.score(X_val, y_val)
        
        print(f"✓ Training complete!")
        print(f"  Train accuracy: {metrics['train_accuracy']:.4f}")
        if 'val_accuracy' in metrics:
            print(f"  Validation accuracy: {metrics['val_accuracy']:.4f}")
        if metrics['n_iterations']:
            print(f"  Iterations: {metrics['n_iterations']}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input embeddings [n_samples, embedding_dim]
            
        Returns:
            Predicted labels [n_samples]
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        # normalize if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input embeddings [n_samples, embedding_dim]
            
        Returns:
            Class probabilities [n_samples, n_classes]
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        # normalize if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Input embeddings
            y: True labels
            
        Returns:
            Accuracy score
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        # normalize if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.score(X, y)
    
    def get_feature_importance(self, top_n: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get feature importance (coefficients).
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Tuple of (feature_indices, coefficients)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        # get coefficients
        coef = self.model.coef_[0]  # [feature_dim]
        
        # get indices sorted by absolute value
        indices = np.argsort(np.abs(coef))[::-1][:top_n]
        values = coef[indices]
        
        return indices, values
    
    def save_model(self, path: str):
        """
        Save model to file.
        
        Args:
            path: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_dim': self.feature_dim,
            'classes': self.classes,
            'config': {
                'C': self.C,
                'max_iter': self.max_iter,
                'solver': self.solver,
                'penalty': self.penalty,
                'class_weight': self.class_weight,
                'random_state': self.random_state,
                'normalize': self.normalize
            }
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model from file.
        
        Args:
            path: Path to model file
        """
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.feature_dim = model_data['feature_dim']
        self.classes = model_data['classes']
        
        config = model_data['config']
        self.C = config['C']
        self.max_iter = config['max_iter']
        self.solver = config['solver']
        self.penalty = config['penalty']
        self.class_weight = config['class_weight']
        self.random_state = config['random_state']
        self.normalize = config['normalize']
        
        print(f"Model loaded from {path}")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Classes: {self.classes}")


def create_logistic_regression_from_config(
    config_path: str = "configs/config.yaml"
) -> LogisticRegressionClassifier:
    """
    Create Logistic Regression classifier from configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        LogisticRegressionClassifier instance
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    lr_config = config['classical_ml']['logistic_regression']
    
    C = lr_config.get('C', 1.0)
    if isinstance(C, list):
        C = C[0]
    
    penalty = lr_config.get('penalty', 'l2')
    if isinstance(penalty, list):
        penalty = penalty[0]
    
    solver = lr_config.get('solver', 'lbfgs')
    if isinstance(solver, list):
        solver = solver[0]
    
    classifier = LogisticRegressionClassifier(
        C=C,
        max_iter=lr_config.get('max_iter', 1000),
        solver=solver,
        penalty=penalty,
        class_weight=lr_config.get('class_weight', None),
        random_state=config['project']['random_seed'],
        normalize=lr_config.get('normalize', True)
    )
    
    return classifier


if __name__ == "__main__":
    # Test Logistic Regression classifier
    print("Testing Logistic Regression Classifier...")
    
    # Create dummy embeddings
    np.random.seed(42)
    
    n_train = 1000
    n_val = 200
    n_test = 200
    embedding_dim = 256
    
    # Generate random embeddings (in practice, these come from encoders)
    X_train = np.random.randn(n_train, embedding_dim)
    y_train = np.random.randint(0, 2, n_train)
    
    X_val = np.random.randn(n_val, embedding_dim)
    y_val = np.random.randint(0, 2, n_val)
    
    X_test = np.random.randn(n_test, embedding_dim)
    y_test = np.random.randint(0, 2, n_test)
    
    # Add some signal to the data
    X_train[y_train == 1] += 0.5
    X_val[y_val == 1] += 0.5
    X_test[y_test == 1] += 0.5
    
    print("\n" + "="*60)
    print("Dataset info:")
    print(f"  Train: {X_train.shape}, labels: {np.bincount(y_train)}")
    print(f"  Val: {X_val.shape}, labels: {np.bincount(y_val)}")
    print(f"  Test: {X_test.shape}, labels: {np.bincount(y_test)}")
    
    # Create and train classifier
    print("\n" + "="*60)
    classifier = LogisticRegressionClassifier(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        penalty='l2',
        normalize=True
    )
    
    metrics = classifier.fit(X_train, y_train, X_val, y_val)
    
    # Test predictions
    print("\n" + "="*60)
    print("Testing predictions...")
    
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    test_accuracy = classifier.score(X_test, y_test)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Probabilities shape: {y_proba.shape}")
    print(f"Sample predictions: {y_pred[:10]}")
    print(f"Sample probabilities:\n{y_proba[:5]}")
    
    # Get feature importance
    print("\n" + "="*60)
    print("Top 10 most important features:")
    indices, values = classifier.get_feature_importance(top_n=10)
    for idx, val in zip(indices, values):
        print(f"  Feature {idx}: {val:.4f}")
    
    # Test save/load
    print("\n" + "="*60)
    save_path = "results/models/test_logistic_regression.pkl"
    classifier.save_model(save_path)
    
    print("\n" + "="*60)
    print("Testing model loading...")
    new_classifier = LogisticRegressionClassifier()
    new_classifier.load_model(save_path)
    
    # Verify loaded model
    loaded_accuracy = new_classifier.score(X_test, y_test)
    print(f"Loaded model test accuracy: {loaded_accuracy:.4f}")
    
    assert abs(test_accuracy - loaded_accuracy) < 1e-6, "Loaded model accuracy mismatch!"
    
    # Test different configurations
    print("\n" + "="*60)
    print("Testing different regularization strengths...")
    
    C_values = [0.01, 0.1, 1.0, 10.0]
    for C in C_values:
        clf = LogisticRegressionClassifier(C=C, normalize=True)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print(f"  C={C:5.2f}: Test accuracy = {acc:.4f}")
    
    print("\n✓ All tests passed!")