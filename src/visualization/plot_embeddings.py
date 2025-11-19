"""
Embedding Visualization Module
Visualize Word2Vec embeddings, model representations, and attention weights.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple
import torch
from pathlib import Path
from wordcloud import WordCloud
import pandas as pd


class EmbeddingVisualizer:
    """Visualize embeddings and learned representations."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        sns.set_style("whitegrid")
        
    def plot_word_embeddings_tsne(
        self,
        embeddings: np.ndarray,
        words: List[str],
        n_words: int = 500,
        perplexity: int = 30,
        save_path: Optional[str] = None
    ):
        """
        Visualize word embeddings using t-SNE.
        
        Args:
            embeddings: Word embedding matrix (vocab_size, embedding_dim)
            words: List of words corresponding to embeddings
            n_words: Number of words to plot
            perplexity: t-SNE perplexity parameter
            save_path: Path to save figure
        """
        # Select most common words
        embeddings = embeddings[:n_words]
        words = words[:n_words]
        
        # Apply t-SNE
        print(f"Applying t-SNE to {len(embeddings)} word embeddings...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=30)
        
        # Annotate sample words
        step = max(1, n_words // 50)  # Show ~50 labels
        for i in range(0, len(words), step):
            ax.annotate(
                words[i],
                xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=9,
                alpha=0.7
            )
        
        ax.set_title(f't-SNE Visualization of Word Embeddings (n={n_words})', fontsize=14)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
    def plot_word_embeddings_pca(
        self,
        embeddings: np.ndarray,
        words: List[str],
        n_words: int = 500,
        save_path: Optional[str] = None
    ):
        """
        Visualize word embeddings using PCA (faster than t-SNE).
        
        Args:
            embeddings: Word embedding matrix
            words: List of words
            n_words: Number of words to plot
            save_path: Path to save figure
        """
        embeddings = embeddings[:n_words]
        words = words[:n_words]
        
        # Apply PCA
        print(f"Applying PCA to {len(embeddings)} word embeddings...")
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=30)
        
        # Annotate
        step = max(1, n_words // 50)
        for i in range(0, len(words), step):
            ax.annotate(
                words[i],
                xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=9,
                alpha=0.7
            )
        
        ax.set_title(f'PCA Visualization of Word Embeddings (n={n_words})', fontsize=14)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_word_similarities(
        self,
        word: str,
        similar_words: List[Tuple[str, float]],
        top_k: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot most similar words to a given word.
        
        Args:
            word: Query word
            similar_words: List of (word, similarity_score) tuples
            top_k: Number of similar words to show
            save_path: Path to save figure
        """
        similar_words = similar_words[:top_k]
        words, scores = zip(*similar_words)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(words))
        
        bars = ax.barh(y_pos, scores, color=plt.cm.viridis(np.array(scores)))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel('Cosine Similarity')
        ax.set_title(f'Top {top_k} Words Similar to "{word}"', fontsize=14)
        ax.set_xlim([0, 1])
        
        # Add value labels
        for i, (w, s) in enumerate(zip(words, scores)):
            ax.text(s + 0.01, i, f'{s:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_sentiment_word_cloud(
        self,
        texts: List[str],
        labels: List[int],
        save_path: Optional[str] = None
    ):
        """
        Create word clouds for positive and negative sentiments.
        
        Args:
            texts: List of text samples
            labels: List of labels (0=negative, 1=positive)
            save_path: Path to save figure
        """
        # Separate by sentiment
        positive_texts = ' '.join([t for t, l in zip(texts, labels) if l == 1])
        negative_texts = ' '.join([t for t, l in zip(texts, labels) if l == 0])
        
        # Create word clouds
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Positive
        wc_pos = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='Greens',
            max_words=100
        ).generate(positive_texts)
        
        axes[0].imshow(wc_pos, interpolation='bilinear')
        axes[0].set_title('Positive Sentiment Words', fontsize=14)
        axes[0].axis('off')
        
        # Negative
        wc_neg = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='Reds',
            max_words=100
        ).generate(negative_texts)
        
        axes[1].imshow(wc_neg, interpolation='bilinear')
        axes[1].set_title('Negative Sentiment Words', fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_encoder_representations(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        method: str = 'tsne',
        save_path: Optional[str] = None
    ):
        """
        Visualize encoder representations (LSTM/GRU/Transformer outputs).
        
        Args:
            embeddings: Encoder outputs (n_samples, hidden_dim)
            labels: Sentiment labels (0/1)
            method: 'tsne' or 'pca'
            save_path: Path to save figure
        """
        # Reduce to 2D
        if method == 'tsne':
            print(f"Applying t-SNE to {len(embeddings)} samples...")
            reducer = TSNE(n_components=2, random_state=42)
        else:
            print(f"Applying PCA to {len(embeddings)} samples...")
            reducer = PCA(n_components=2, random_state=42)
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for label, color, name in [(0, 'red', 'Negative'), (1, 'green', 'Positive')]:
            mask = labels == label
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=color,
                label=name,
                alpha=0.5,
                s=20
            )
        
        ax.set_title(f'Encoder Representations ({method.upper()})', fontsize=14)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        save_path: Optional[str] = None
    ):
        """
        Visualize attention weights from Transformer.
        
        Args:
            attention_weights: Attention matrix (seq_len, seq_len)
            tokens: List of tokens
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            attention_weights,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            annot=False,
            fmt='.2f',
            square=True,
            cbar_kws={'label': 'Attention Weight'},
            ax=ax
        )
        
        ax.set_title('Attention Weight Heatmap', fontsize=14)
        ax.set_xlabel('Target Tokens')
        ax.set_ylabel('Source Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# Convenience functions
def visualize_word2vec_embeddings(
    embedding_path: str,
    n_words: int = 500,
    method: str = 'tsne'
):
    """
    Load and visualize Word2Vec embeddings.
    
    Args:
        embedding_path: Path to .model file
        n_words: Number of words to visualize
        method: 'tsne' or 'pca'
    """
    from gensim.models import Word2Vec
    
    # Load model
    model = Word2Vec.load(embedding_path)
    
    # Get embeddings and words
    words = list(model.wv.index_to_key[:n_words])
    embeddings = np.array([model.wv[w] for w in words])
    
    # Visualize
    viz = EmbeddingVisualizer()
    
    if method == 'tsne':
        viz.plot_word_embeddings_tsne(embeddings, words, n_words)
    else:
        viz.plot_word_embeddings_pca(embeddings, words, n_words)


if __name__ == "__main__":
    # Example usage
    print("Embedding Visualizer")
    print("=" * 50)
    
    # Test with dummy data
    viz = EmbeddingVisualizer()
    
    # Dummy embeddings
    np.random.seed(42)
    dummy_embeddings = np.random.randn(100, 50)
    dummy_words = [f"word_{i}" for i in range(100)]
    
    print("\nGenerating t-SNE visualization...")
    viz.plot_word_embeddings_tsne(dummy_embeddings, dummy_words, n_words=100)
    
    print("\nVisualization complete!")