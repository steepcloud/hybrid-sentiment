import numpy as np
import os
import yaml
from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from typing import Optional, Dict, Tuple
import joblib


class RandomForestClassifier:
    """Random Forest classifier for sentiment analysis using embeddings."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        bootstrap: bool = True,
        class_weight: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap samples
            class_weight: Weights for classes ('balanced' or None)
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 = use all cores)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Initialize model
        self.model = SKRandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        # Training info
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
        Train the random forest model.
        
        Args:
            X_train: Training embeddings [n_samples, embedding_dim]
            y_train: Training labels [n_samples]
            X_val: Optional validation embeddings
            y_val: Optional validation labels
            
        Returns:
            Dictionary with training metrics
        """
        print("Training Random Forest...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Feature dimension: {X_train.shape[1]}")
        print(f"  Number of trees: {self.n_estimators}")
        print(f"  Max depth: {self.max_depth if self.max_depth else 'unlimited'}")
        print(f"  Max features: {self.max_features}")
        
        self.feature_dim = X_train.shape[1]
        
        # Train model
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        self.classes = self.model.classes_
        
        # Calculate metrics
        metrics = {
            'train_accuracy': self.model.score(X_train, y_train),
            'n_trees': len(self.model.estimators_)
        }
        
        if X_val is not None and y_val is not None:
            metrics['val_accuracy'] = self.model.score(X_val, y_val)
        
        print(f"✓ Training complete!")
        print(f"  Train accuracy: {metrics['train_accuracy']:.4f}")
        if 'val_accuracy' in metrics:
            print(f"  Validation accuracy: {metrics['val_accuracy']:.4f}")
        print(f"  Number of trees: {metrics['n_trees']}")
        
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
    
    def get_feature_importance(self, top_n: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get feature importance from the random forest.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Tuple of (feature_indices, importance_values)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Get indices sorted by importance
        indices = np.argsort(importances)[::-1][:top_n]
        values = importances[indices]
        
        return indices, values
    
    def get_tree_depths(self) -> Dict:
        """
        Get statistics about tree depths.
        
        Returns:
            Dictionary with depth statistics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        depths = [tree.get_depth() for tree in self.model.estimators_]
        
        return {
            'mean': np.mean(depths),
            'median': np.median(depths),
            'min': np.min(depths),
            'max': np.max(depths)
        }
    
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
            'config': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'class_weight': self.class_weight,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs
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
        
        config = model_data['config']
        self.n_estimators = config['n_estimators']
        self.max_depth = config['max_depth']
        self.min_samples_split = config['min_samples_split']
        self.min_samples_leaf = config['min_samples_leaf']
        self.max_features = config['max_features']
        self.bootstrap = config['bootstrap']
        self.class_weight = config['class_weight']
        self.random_state = config['random_state']
        self.n_jobs = config['n_jobs']
        
        print(f"Model loaded from {path}")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Classes: {self.classes}")
        print(f"  Number of trees: {self.n_estimators}")


def create_random_forest_from_config(
    config_path: str = "configs/config.yaml"
) -> RandomForestClassifier:
    """
    Create Random Forest classifier from configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        RandomForestClassifier instance
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    rf_config = config['classical_ml']['random_forest']
    
    # Handle single values or lists (for hyperparameter tuning)
    n_estimators = rf_config.get('n_estimators', 100)
    if isinstance(n_estimators, list):
        n_estimators = n_estimators[0]
    
    max_depth = rf_config.get('max_depth', None)
    if isinstance(max_depth, list):
        max_depth = max_depth[0]
    
    min_samples_split = rf_config.get('min_samples_split', 2)
    if isinstance(min_samples_split, list):
        min_samples_split = min_samples_split[0]
    
    min_samples_leaf = rf_config.get('min_samples_leaf', 1)
    if isinstance(min_samples_leaf, list):
        min_samples_leaf = min_samples_leaf[0]
    
    max_features = rf_config.get('max_features', 'sqrt')
    if isinstance(max_features, list):
        max_features = max_features[0]
    
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=rf_config.get('bootstrap', True),
        class_weight=rf_config.get('class_weight', None),
        random_state=config['project']['random_seed'],
        n_jobs=-1
    )
    
    return classifier


if __name__ == "__main__":
    # Test Random Forest classifier
    print("Testing Random Forest Classifier...")
    
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
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt'
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
    
    # Get tree depth statistics
    print("\n" + "="*60)
    print("Tree depth statistics:")
    depth_stats = classifier.get_tree_depths()
    for key, value in depth_stats.items():
        print(f"  {key.capitalize()}: {value:.2f}")
    
    # Test save/load
    print("\n" + "="*60)
    save_path = "results/models/test_random_forest.pkl"
    classifier.save_model(save_path)
    
    print("\n" + "="*60)
    print("Testing model loading...")
    new_classifier = RandomForestClassifier()
    new_classifier.load_model(save_path)
    
    # Verify loaded model
    loaded_accuracy = new_classifier.score(X_test, y_test)
    print(f"Loaded model test accuracy: {loaded_accuracy:.4f}")
    
    assert abs(test_accuracy - loaded_accuracy) < 1e-6, "Loaded model accuracy mismatch!"
    
    # Test different configurations
    print("\n" + "="*60)
    print("Testing different number of trees...")
    
    n_estimators_values = [10, 50, 100, 200]
    for n_est in n_estimators_values:
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=10)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print(f"  n_estimators={n_est:3d}: Test accuracy = {acc:.4f}")
    
    print("\n✓ All tests passed!")