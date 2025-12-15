import torch
import torch.nn as nn
import yaml
from typing import Optional


class LSTMEncoder(nn.Module):
    """LSTM-based encoder for generating text embeddings."""
    
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
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            pretrained_embeddings: Optional pretrained embedding matrix
            freeze_embeddings: Whether to freeze embedding layer
            padding_idx: Index for padding token
        """
        super(LSTMEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.padding_idx = padding_idx
        
        # embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        # load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # freeze embeddings if specified
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # output dimension (for embeddings)
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
        # get embeddings
        embedded = self.embedding(input_ids)  # [batch_size, seq_length, embedding_dim]
        embedded = self.dropout(embedded)
        
        # pack sequences if lengths provided
        if lengths is not None:
            # sort by length (required for pack_padded_sequence)
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            embedded_sorted = embedded[sorted_idx]
            
            # pack
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded_sorted, 
                lengths_sorted.cpu(), 
                batch_first=True,
                enforce_sorted=True
            )
            
            # LSTM forward
            packed_output, (hidden, cell) = self.lstm(packed)
            
            # unpack
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # unsort
            _, unsorted_idx = sorted_idx.sort()
            output = output[unsorted_idx]
            hidden = hidden[:, unsorted_idx, :]
            cell = cell[:, unsorted_idx, :]
        else:
            # standard LSTM forward
            output, (hidden, cell) = self.lstm(embedded)
        
        # output: [batch_size, seq_length, hidden_dim * num_directions]
        # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        
        if return_sequence:
            # return full sequence output
            return self.dropout(output)
        else:
            # return final embedding (concatenate final hidden states)
            if self.bidirectional:
                # concatenate forward and backward final hidden states
                # hidden[-2]: forward, hidden[-1]: backward
                final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                final_hidden = hidden[-1]
            
            return self.dropout(final_hidden)  # [batch_size, output_dim]
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of the output embeddings."""
        return self.output_dim


class LSTMClassifier(nn.Module):
    """LSTM-based classifier for end-to-end training (baseline)."""
    
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
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            pretrained_embeddings: Optional pretrained embedding matrix
            freeze_embeddings: Whether to freeze embedding layer
            padding_idx: Index for padding token
        """
        super(LSTMClassifier, self).__init__()
        
        # LSTM encoder
        self.encoder = LSTMEncoder(
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
        
        # classification head
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
        # get embeddings from encoder
        embeddings = self.encoder(input_ids, lengths, return_sequence=False)
        
        # classify
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


def create_lstm_encoder_from_config(config_path: str = "configs/config.yaml", vocab_size: int = None) -> LSTMEncoder:
    """
    Create LSTM encoder from configuration file.
    
    Args:
        config_path: Path to configuration file
        vocab_size: Vocabulary size (required)
        
    Returns:
        LSTMEncoder model
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    lstm_config = config['deep_learning']['lstm']
    
    if vocab_size is None:
        vocab_size = config['data']['vocab_size']
    
    model = LSTMEncoder(
        vocab_size=vocab_size,
        embedding_dim=lstm_config['embedding_dim'],
        hidden_dim=lstm_config.get('hidden_dim', 128),
        num_layers=lstm_config.get('num_layers', 2),
        dropout=lstm_config.get('dropout', 0.3),
        bidirectional=lstm_config['bidirectional']
    )
    
    return model


def create_lstm_classifier_from_config(config_path: str = "configs/config.yaml", vocab_size: int = None) -> LSTMClassifier:
    """
    Create LSTM classifier from configuration file.
    
    Args:
        config_path: Path to configuration file
        vocab_size: Vocabulary size (required)
        
    Returns:
        LSTMClassifier model
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    lstm_config = config['deep_learning']['lstm']
    
    if vocab_size is None:
        vocab_size = config['data']['vocab_size']
    
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=lstm_config['embedding_dim'],
        hidden_dim=lstm_config.get('hidden_dim', 128),
        num_layers=lstm_config.get('num_layers', 2),
        num_classes=2,  # Binary classification
        dropout=lstm_config.get('dropout', 0.3),
        bidirectional=lstm_config['bidirectional']
    )
    
    return model


if __name__ == "__main__":
    # Test the LSTM encoder
    print("Testing LSTM Encoder...")
    
    # Create dummy data
    batch_size = 4
    seq_length = 20
    vocab_size = 10000
    
    # Random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    lengths = torch.tensor([20, 15, 10, 5])
    
    # Create encoder
    encoder = LSTMEncoder(
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
    print("Testing LSTM Classifier...")
    
    classifier = LSTMClassifier(
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
    
    print("\n✓ All tests passed!")