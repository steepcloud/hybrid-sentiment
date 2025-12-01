import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional, Tuple


class BERTEncoder(nn.Module):
    """BERT/RoBERTa encoder for sentiment analysis."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_bert: bool = False,
        max_length: int = 512
    ):
        """
        Args:
            model_name: HuggingFace model name ('bert-base-uncased', 'roberta-base', etc.)
            num_classes: Number of output classes
            dropout: Dropout rate
            freeze_bert: Whether to freeze BERT parameters
            max_length: Maximum sequence length
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        
        # Load pre-trained model
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Get hidden size from model config
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            return_embeddings: If True, return embeddings instead of logits
            
        Returns:
            Logits [batch_size, num_classes] or embeddings [batch_size, hidden_size]
        """
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        if return_embeddings:
            return cls_embedding
        
        # Classification
        pooled = self.dropout(cls_embedding)
        logits = self.classifier(pooled)
        
        return logits
    
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get embeddings for hybrid models.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Embeddings [batch_size, hidden_size]
        """
        return self.forward(input_ids, attention_mask, return_embeddings=True)


class BERTClassifier(nn.Module):
    """Wrapper for end-to-end BERT classification."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_bert: bool = False
    ):
        super().__init__()
        self.encoder = BERTEncoder(model_name, num_classes, dropout, freeze_bert)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(input_ids, attention_mask)
    
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder.get_embeddings(input_ids, attention_mask)


def create_bert_classifier_from_config(model_name: str = "bert-base-uncased") -> BERTClassifier:
    """
    Create BERT classifier from config.
    
    Args:
        model_name: Model name ('bert-base-uncased', 'roberta-base', 'distilbert-base-uncased')
        
    Returns:
        BERTClassifier
    """
    return BERTClassifier(
        model_name=model_name,
        num_classes=2,
        dropout=0.3,
        freeze_bert=False  # Fine-tune all layers
    )