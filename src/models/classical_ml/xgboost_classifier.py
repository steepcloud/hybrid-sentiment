import numpy as np
import os
import yaml
from xgboost import XGBClassifier
from typing import Optional, Dict, Tuple
import joblib


class XGBoostClassifier:
    """XGBoost classifier for sentiment analysis using embeddings."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        use_label_encoder: bool = False,
        eval_metric: str = 'logloss'
    ):
        """
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            min_child_weight: Minimum sum of instance weight in a child
            gamma: Minimum loss reduction required for split
            reg_alpha: L1 regularization on weights
            reg_lambda: L2 regularization on weights
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 = use all cores)
            use_label_encoder: Whether to use label encoder (deprecated)
            eval_metric: Evaluation metric
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_label_encoder = use_label_encoder
        self.eval_metric = eval_metric
        
        # Initialize model
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            use_label_encoder=use_label_encoder,
            eval_metric=eval_metric
        )
        
        # Training info
        self.is_trained = False
        self.feature_dim = None
        self.classes = None
        self.evals_result = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training embeddings [n_samples, embedding_dim]
            y_train: Training labels [n_samples]
            X_val: Optional validation embeddings
            y_val: Optional validation labels
            early_stopping_rounds: Stop if no improvement for this many rounds
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training metrics
        """
        print("Training XGBoost...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Feature dimension: {X_train.shape[1]}")
        print(f"  Number of estimators: {self.n_estimators}")
        print(f"  Max depth: {self.max_depth}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Subsample: {self.subsample}")
        print(f"  Colsample by tree: {self.colsample_bytree}")
        
        self.feature_dim = X_train.shape[1]
        
        # Prepare eval set for early stopping
        eval_set = None
        callbacks = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

            if early_stopping_rounds:
                from xgboost.callback import EarlyStopping
                callbacks = [EarlyStopping(rounds=early_stopping_rounds, save_best=True)]
        
        # Train model
        fit_params = {
            'verbose': verbose
        }
        
        if eval_set:
            fit_params['eval_set'] = eval_set
        
        if callbacks:
            fit_params['callbacks'] = callbacks
        
        self.model.fit(X_train, y_train, **fit_params)
        
        # Get evaluation results if available
        if hasattr(self.model, 'evals_result_'):
            self.evals_result = self.model.evals_result_
        
        self.is_trained = True
        self.classes = self.model.classes_
        
        # Calculate metrics
        metrics = {
            'train_accuracy': self.model.score(X_train, y_train),
            'n_estimators_used': self.model.best_iteration + 1 if hasattr(self.model, 'best_iteration') and self.model.best_iteration else self.n_estimators
        }
        
        if X_val is not None and y_val is not None:
            metrics['val_accuracy'] = self.model.score(X_val, y_val)
        
        print(f"✓ Training complete!")
        print(f"  Train accuracy: {metrics['train_accuracy']:.4f}")
        if 'val_accuracy' in metrics:
            print(f"  Validation accuracy: {metrics['val_accuracy']:.4f}")
        if hasattr(self.model, 'best_iteration'):
            print(f"  Best iteration: {self.model.best_iteration + 1}")
        
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
        
        return self.model.score(X, y)
    
    def get_feature_importance(
        self,
        importance_type: str = 'weight',
        top_n: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get feature importance from XGBoost.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')
            top_n: Number of top features to return
            
        Returns:
            Tuple of (feature_indices, importance_values)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get feature importances
        importances = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Convert to array (handle missing features)
        importance_array = np.zeros(self.feature_dim)
        for feat_name, importance in importances.items():
            feat_idx = int(feat_name.replace('f', ''))
            importance_array[feat_idx] = importance
        
        # Get indices sorted by importance
        indices = np.argsort(importance_array)[::-1][:top_n]
        values = importance_array[indices]
        
        return indices, values
    
    def get_training_history(self) -> Optional[Dict]:
        """
        Get training history (if early stopping was used).
        
        Returns:
            Dictionary with training/validation metrics per iteration
        """
        return self.evals_result
    
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
            'is_trained': self.is_trained,
            'feature_dim': self.feature_dim,
            'classes': self.classes,
            'evals_result': self.evals_result,
            'config': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'min_child_weight': self.min_child_weight,
                'gamma': self.gamma,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'eval_metric': self.eval_metric
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
        self.is_trained = model_data['is_trained']
        self.feature_dim = model_data['feature_dim']
        self.classes = model_data['classes']
        self.evals_result = model_data.get('evals_result')
        
        config = model_data['config']
        self.n_estimators = config['n_estimators']
        self.max_depth = config['max_depth']
        self.learning_rate = config['learning_rate']
        self.subsample = config['subsample']
        self.colsample_bytree = config['colsample_bytree']
        self.min_child_weight = config['min_child_weight']
        self.gamma = config['gamma']
        self.reg_alpha = config['reg_alpha']
        self.reg_lambda = config['reg_lambda']
        self.random_state = config['random_state']
        self.n_jobs = config['n_jobs']
        self.eval_metric = config['eval_metric']
        
        print(f"Model loaded from {path}")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Classes: {self.classes}")
        print(f"  Number of estimators: {self.n_estimators}")


def create_xgboost_from_config(
    config_path: str = "configs/config.yaml"
) -> XGBoostClassifier:
    """
    Create XGBoost classifier from configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        XGBoostClassifier instance
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    xgb_config = config['classical_ml']['xgboost']
    
    # Handle single values or lists (for hyperparameter tuning)
    n_estimators = xgb_config.get('n_estimators', 100)
    if isinstance(n_estimators, list):
        n_estimators = n_estimators[0]
    
    max_depth = xgb_config.get('max_depth', 6)
    if isinstance(max_depth, list):
        max_depth = max_depth[0]
    
    learning_rate = xgb_config.get('learning_rate', 0.1)
    if isinstance(learning_rate, list):
        learning_rate = learning_rate[0]
    
    subsample = xgb_config.get('subsample', 1.0)
    if isinstance(subsample, list):
        subsample = subsample[0]
    
    colsample_bytree = xgb_config.get('colsample_bytree', 1.0)
    if isinstance(colsample_bytree, list):
        colsample_bytree = colsample_bytree[0]
    
    classifier = XGBoostClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=xgb_config.get('min_child_weight', 1),
        gamma=xgb_config.get('gamma', 0.0),
        reg_alpha=xgb_config.get('reg_alpha', 0.0),
        reg_lambda=xgb_config.get('reg_lambda', 1.0),
        random_state=config['project']['random_seed'],
        n_jobs=-1
    )
    
    return classifier


if __name__ == "__main__":
    # Test XGBoost classifier
    print("Testing XGBoost Classifier...")
    
    # Create dummy embeddings
    np.random.seed(42)
    
    n_train = 1000
    n_val = 200
    n_test = 200
    embedding_dim = 256
    
    # Generate random embeddings
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
    classifier = XGBoostClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
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
    print("Top 10 most important features (by weight):")
    indices, values = classifier.get_feature_importance(importance_type='weight', top_n=10)
    for idx, val in zip(indices, values):
        print(f"  Feature {idx}: {val:.2f}")
    
    print("\nTop 10 most important features (by gain):")
    indices, values = classifier.get_feature_importance(importance_type='gain', top_n=10)
    for idx, val in zip(indices, values):
        print(f"  Feature {idx}: {val:.4f}")
    
    # Test save/load
    print("\n" + "="*60)
    save_path = "results/models/test_xgboost.pkl"
    classifier.save_model(save_path)
    
    print("\n" + "="*60)
    print("Testing model loading...")
    new_classifier = XGBoostClassifier()
    new_classifier.load_model(save_path)
    
    # Verify loaded model
    loaded_accuracy = new_classifier.score(X_test, y_test)
    print(f"Loaded model test accuracy: {loaded_accuracy:.4f}")
    
    assert abs(test_accuracy - loaded_accuracy) < 1e-6, "Loaded model accuracy mismatch!"
    
    # Test different learning rates
    print("\n" + "="*60)
    print("Testing different learning rates...")
    
    learning_rates = [0.01, 0.05, 0.1, 0.3]
    for lr in learning_rates:
        clf = XGBoostClassifier(n_estimators=100, learning_rate=lr)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print(f"  learning_rate={lr:.2f}: Test accuracy = {acc:.4f}")
    
    print("\n✓ All tests passed!")