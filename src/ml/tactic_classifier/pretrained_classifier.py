from typing import List

import torch
import torch.nn as nn

class PretrainedTacticClassifier(nn.Module):
    """
    Pretrained neural network classifier adapted for CSGO tactic prediction.
    
    Architecture:
    - Input layer: Accepts normalized feature vectors
    - Hidden layers: Multiple fully-connected layers with ReLU activations
    - Dropout: Regularization to prevent overfitting
    - Output layer: Softmax over tactic classes
    
    This is implemented as a feature-based classifier that takes learned
    representations rather than raw images, making it ideal for our graph data.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        num_classes: int = 10,
        dropout_rate: float = 0.3
    ):
        """
        Initialize the pretrained tactic classifier.
        
        Args:
            input_dim: Dimension of input feature vectors
            hidden_dims: List of hidden layer dimensions. Defaults to [256, 128, 64]
            num_classes: Number of tactic classes to predict
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        
        # Set default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Build feature extraction layers: progressively reduce dimension
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear transformation
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Batch normalization for stable training
            layers.append(nn.BatchNorm1d(hidden_dim))
            # ReLU activation for non-linearity
            layers.append(nn.ReLU())
            # Dropout for regularization and preventing overfitting
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Classification head: map learned features to class probabilities
        self.classifier = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            x: Input features of shape (batch_size, input_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Extract features through hidden layers
        features = self.feature_extractor(x)
        
        # Classify based on learned features
        logits = self.classifier(features)
        
        return logits