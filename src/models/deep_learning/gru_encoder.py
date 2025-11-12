import torch
import torch.nn as nn
import yaml
from typing import Tuple, Optional


class GRUEncoder(nn.Module):
    """GRU-based encoder for generating text embeddings."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of GRU hidden state
            num_layers: Number of GRU layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
            pretrained_embeddings: Optional pretrained embedding matrix
            freeze_embeddings: Whether to freeze embedding layer
            padding_idx: Index for padding token
        """
        super(GRUEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
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
        
        # GRU layer
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension (for embeddings)
        self.output_dim = hidden_dim * self.num_directions
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            lengths: Actual lengths of sequences (for packing)
            return_sequence: If True, return full sequence output; if False, return final embedding
            
        Returns:
            embeddings: [batch_size, output_dim] or [batch_size, seq_length, output_dim]
        """
        # Get embeddings
        embedded = self.embedding(input_ids)  # [batch_size, seq_length, embedding_dim]
        embedded = self.dropout(embedded)
        
        # Pack sequences if lengths provided
        if lengths is not None:
            # Sort by length (required for pack_padded_sequence)
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            embedded_sorted = embedded[sorted_idx]
            
            # Pack
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded_sorted, 
                lengths_sorted.cpu(), 
                batch_first=True,
                enforce_sorted=True
            )
            
            # GRU forward
            packed_output, hidden = self.gru(packed)
            
            # Unpack
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # Unsort
            _, unsorted_idx = sorted_idx.sort()
            output = output[unsorted_idx]
            hidden = hidden[:, unsorted_idx, :]
        else:
            # Standard GRU forward
            output, hidden = self.gru(embedded)
        
        # output: [batch_size, seq_length, hidden_dim * num_directions]
        # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        
        if return_sequence:
            # Return full sequence output
            return self.dropout(output)
        else:
            # Return final embedding (concatenate final hidden states)
            if self.bidirectional:
                # Concatenate forward and backward final hidden states
                # hidden[-2]: forward, hidden[-1]: backward
                final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                final_hidden = hidden[-1]
            
            return self.dropout(final_hidden)  # [batch_size, output_dim]
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of the output embeddings."""
        return self.output_dim


class GRUClassifier(nn.Module):
    """GRU-based classifier for end-to-end training (baseline)."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of GRU hidden state
            num_layers: Number of GRU layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
            pretrained_embeddings: Optional pretrained embedding matrix
            freeze_embeddings: Whether to freeze embedding layer
            padding_idx: Index for padding token
        """
        super(GRUClassifier, self).__init__()
        
        # GRU encoder
        self.encoder = GRUEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
            padding_idx=padding_idx
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            lengths: Actual lengths of sequences
            
        Returns:
            logits: [batch_size, num_classes]
        """
        # Get embeddings from encoder
        embeddings = self.encoder(input_ids, lengths, return_sequence=False)
        
        # Classify
        logits = self.classifier(embeddings)
        
        return logits
    
    def get_embeddings(self, input_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get embeddings without classification.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            lengths: Actual lengths of sequences
            
        Returns:
            embeddings: [batch_size, output_dim]
        """
        return self.encoder(input_ids, lengths, return_sequence=False)


def create_gru_encoder_from_config(config_path: str = "configs/config.yaml", vocab_size: int = None) -> GRUEncoder:
    """
    Create GRU encoder from configuration file.
    
    Args:
        config_path: Path to configuration file
        vocab_size: Vocabulary size (required)
        
    Returns:
        GRUEncoder model
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    gru_config = config['deep_learning']['gru']
    
    if vocab_size is None:
        vocab_size = config['data']['vocab_size']
    
    model = GRUEncoder(
        vocab_size=vocab_size,
        embedding_dim=gru_config['embedding_dim'],
        hidden_dim=gru_config.get('hidden_dim', 128),
        num_layers=gru_config.get('num_layers', 2),
        dropout=gru_config.get('dropout', 0.3),
        bidirectional=gru_config['bidirectional']
    )
    
    return model


def create_gru_classifier_from_config(config_path: str = "configs/config.yaml", vocab_size: int = None) -> GRUClassifier:
    """
    Create GRU classifier from configuration file.
    
    Args:
        config_path: Path to configuration file
        vocab_size: Vocabulary size (required)
        
    Returns:
        GRUClassifier model
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    gru_config = config['deep_learning']['gru']
    
    if vocab_size is None:
        vocab_size = config['data']['vocab_size']
    
    model = GRUClassifier(
        vocab_size=vocab_size,
        embedding_dim=gru_config['embedding_dim'],
        hidden_dim=gru_config.get('hidden_dim', 128),
        num_layers=gru_config.get('num_layers', 2),
        num_classes=2,  # Binary classification
        dropout=gru_config.get('dropout', 0.3),
        bidirectional=gru_config['bidirectional']
    )
    
    return model


if __name__ == "__main__":
    # Test the GRU encoder
    print("Testing GRU Encoder...")
    
    # Create dummy data
    batch_size = 4
    seq_length = 20
    vocab_size = 10000
    
    # Random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    lengths = torch.tensor([20, 15, 10, 5])
    
    # Create encoder
    encoder = GRUEncoder(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_dim=128,
        num_layers=2,
        bidirectional=True,
        dropout=0.3
    )
    
    print(f"\nEncoder architecture:")
    print(encoder)
    print(f"\nTotal parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    embeddings = encoder(input_ids, lengths)
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Expected shape: [batch_size={batch_size}, output_dim={encoder.output_dim}]")
    
    # Test sequence output
    sequence_output = encoder(input_ids, lengths, return_sequence=True)
    print(f"\nSequence output shape: {sequence_output.shape}")
    
    # Test classifier
    print("\n" + "="*60)
    print("Testing GRU Classifier...")
    
    classifier = GRUClassifier(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_dim=128,
        num_layers=2,
        num_classes=2,
        bidirectional=True,
        dropout=0.3
    )
    
    print(f"\nClassifier architecture:")
    print(classifier)
    print(f"\nTotal parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    
    # Test forward pass
    logits = classifier(input_ids, lengths)
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: [batch_size={batch_size}, num_classes=2]")
    
    # Test embeddings extraction
    embeddings = classifier.get_embeddings(input_ids, lengths)
    print(f"\nExtracted embeddings shape: {embeddings.shape}")
    
    # Compare with LSTM
    print("\n" + "="*60)
    print("Comparing GRU vs LSTM parameters...")
    
    from lstm_encoder import LSTMEncoder, LSTMClassifier
    
    lstm_encoder = LSTMEncoder(vocab_size, 300, 128, 2, bidirectional=True)
    gru_encoder = GRUEncoder(vocab_size, 300, 128, 2, bidirectional=True)
    
    lstm_params = sum(p.numel() for p in lstm_encoder.parameters())
    gru_params = sum(p.numel() for p in gru_encoder.parameters())
    
    print(f"LSTM parameters: {lstm_params:,}")
    print(f"GRU parameters: {gru_params:,}")
    print(f"Difference: {lstm_params - gru_params:,} ({(1 - gru_params/lstm_params)*100:.1f}% fewer in GRU)")
    
    print("\n✓ All tests passed!")