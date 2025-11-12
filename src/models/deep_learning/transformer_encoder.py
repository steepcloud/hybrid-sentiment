import torch
import torch.nn as nn
import yaml
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, embedding_dim: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            embedding_dim: Dimension of embeddings
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, embedding_dim]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_length, embedding_dim]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer-based encoder for generating text embeddings."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        num_heads: int = 6,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings (must be divisible by num_heads)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            pretrained_embeddings: Optional pretrained embedding matrix
            freeze_embeddings: Whether to freeze embedding layer
            padding_idx: Index for padding token
        """
        super(TransformerEncoder, self).__init__()
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        
        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # Freeze embeddings if specified
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, max_seq_length, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        self.output_dim = embedding_dim
    
    def _create_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for attention.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            
        Returns:
            Mask tensor [batch_size, seq_length]
        """
        return (input_ids == self.padding_idx)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            mask: Optional attention mask [batch_size, seq_length]
            return_sequence: If True, return full sequence; if False, return pooled output
            
        Returns:
            embeddings: [batch_size, output_dim] or [batch_size, seq_length, output_dim]
        """
        # Create padding mask if not provided
        if mask is None:
            mask = self._create_padding_mask(input_ids)
        
        # Get embeddings
        x = self.embedding(input_ids) * math.sqrt(self.embedding_dim)  # Scale embeddings
        x = self.pos_encoder(x)
        
        # Pass through transformer
        # Note: src_key_padding_mask expects True for positions to be masked
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        if return_sequence:
            # Return full sequence
            return self.dropout(output)
        else:
            # Pool: take mean of non-padded tokens
            # Invert mask for pooling (True where we want to keep values)
            pooling_mask = (~mask).unsqueeze(-1).float()  # [batch_size, seq_length, 1]
            
            # Sum and average
            sum_embeddings = (output * pooling_mask).sum(dim=1)
            avg_embeddings = sum_embeddings / pooling_mask.sum(dim=1).clamp(min=1)
            
            return self.dropout(avg_embeddings)  # [batch_size, output_dim]
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of the output embeddings."""
        return self.output_dim


class TransformerClassifier(nn.Module):
    """Transformer-based classifier for end-to-end training (baseline)."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        num_heads: int = 6,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            num_classes: Number of output classes
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            pretrained_embeddings: Optional pretrained embedding matrix
            freeze_embeddings: Whether to freeze embedding layer
            padding_idx: Index for padding token
        """
        super(TransformerClassifier, self).__init__()
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=max_seq_length,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
            padding_idx=padding_idx
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            mask: Optional attention mask
            
        Returns:
            logits: [batch_size, num_classes]
        """
        # Get embeddings from encoder
        embeddings = self.encoder(input_ids, mask, return_sequence=False)
        
        # Classify
        logits = self.classifier(embeddings)
        
        return logits
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get embeddings without classification.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            mask: Optional attention mask
            
        Returns:
            embeddings: [batch_size, output_dim]
        """
        return self.encoder(input_ids, mask, return_sequence=False)


def create_transformer_encoder_from_config(
    config_path: str = "configs/config.yaml",
    vocab_size: int = None
) -> TransformerEncoder:
    """
    Create Transformer encoder from configuration file.
    
    Args:
        config_path: Path to configuration file
        vocab_size: Vocabulary size (required)
        
    Returns:
        TransformerEncoder model
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    transformer_config = config['deep_learning']['transformer']
    
    if vocab_size is None:
        vocab_size = config['data']['vocab_size']
    
    model = TransformerEncoder(
        vocab_size=vocab_size,
        embedding_dim=transformer_config['embedding_dim'],
        num_heads=transformer_config['num_heads'],
        num_layers=transformer_config['num_layers'],
        dim_feedforward=transformer_config['dim_feedforward'],
        dropout=transformer_config['dropout'],
        max_seq_length=config['data']['max_length']
    )
    
    return model


def create_transformer_classifier_from_config(
    config_path: str = "configs/config.yaml",
    vocab_size: int = None
) -> TransformerClassifier:
    """
    Create Transformer classifier from configuration file.
    
    Args:
        config_path: Path to configuration file
        vocab_size: Vocabulary size (required)
        
    Returns:
        TransformerClassifier model
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    transformer_config = config['deep_learning']['transformer']
    
    if vocab_size is None:
        vocab_size = config['data']['vocab_size']
    
    model = TransformerClassifier(
        vocab_size=vocab_size,
        embedding_dim=transformer_config['embedding_dim'],
        num_heads=transformer_config['num_heads'],
        num_layers=transformer_config['num_layers'],
        dim_feedforward=transformer_config['dim_feedforward'],
        num_classes=2,
        dropout=transformer_config['dropout'],
        max_seq_length=config['data']['max_length']
    )
    
    return model


if __name__ == "__main__":
    # Test the Transformer encoder
    print("Testing Transformer Encoder...")
    
    # Create dummy data
    batch_size = 4
    seq_length = 20
    vocab_size = 10000
    
    # Random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Create encoder
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        embedding_dim=300,
        num_heads=6,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1
    )
    
    print(f"\nEncoder architecture:")
    print(encoder)
    print(f"\nTotal parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    embeddings = encoder(input_ids)
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Expected shape: [batch_size={batch_size}, output_dim={encoder.output_dim}]")
    
    # Test sequence output
    sequence_output = encoder(input_ids, return_sequence=True)
    print(f"\nSequence output shape: {sequence_output.shape}")
    
    # Test classifier
    print("\n" + "="*60)
    print("Testing Transformer Classifier...")
    
    classifier = TransformerClassifier(
        vocab_size=vocab_size,
        embedding_dim=300,
        num_heads=6,
        num_layers=3,
        dim_feedforward=512,
        num_classes=2,
        dropout=0.1
    )
    
    print(f"\nClassifier architecture:")
    print(classifier)
    print(f"\nTotal parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    
    # Test forward pass
    logits = classifier(input_ids)
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: [batch_size={batch_size}, num_classes=2]")
    
    # Test embeddings extraction
    embeddings = classifier.get_embeddings(input_ids)
    print(f"\nExtracted embeddings shape: {embeddings.shape}")
    
    # Compare with LSTM and GRU
    print("\n" + "="*60)
    print("Comparing model parameters...")
    
    from lstm_encoder import LSTMEncoder
    from gru_encoder import GRUEncoder
    
    lstm_encoder = LSTMEncoder(vocab_size, 300, 128, 2, bidirectional=True)
    gru_encoder = GRUEncoder(vocab_size, 300, 128, 2, bidirectional=True)
    transformer_encoder = TransformerEncoder(vocab_size, 300, 6, 3, 512)
    
    lstm_params = sum(p.numel() for p in lstm_encoder.parameters())
    gru_params = sum(p.numel() for p in gru_encoder.parameters())
    transformer_params = sum(p.numel() for p in transformer_encoder.parameters())
    
    print(f"LSTM parameters: {lstm_params:,}")
    print(f"GRU parameters: {gru_params:,}")
    print(f"Transformer parameters: {transformer_params:,}")
    
    print("\n✓ All tests passed!")