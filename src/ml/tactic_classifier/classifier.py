import os
import joblib
import numpy as np
import torch
import torch.nn as nn

from collections import Counter
from typing import Dict, Tuple, Optional
from pretrained_classifier import PretrainedTacticClassifier
from tactic_feature_dataset import TacticFeatureDataset
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, F1Score, Precision, Recall

class TacticClassifierTrainer:
    """
    Handles training, validation, and evaluation of the tactic classifier.
    
    Responsibilities:
    - Loading and splitting data
    - Managing model training with early stopping
    - Computing evaluation metrics
    - Saving and loading model checkpoints
    - Logging training progress
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        checkpoint_dir: str = "research_project/mlmodels"
    ):
        """
        Initialize trainer.
        
        Args:
            model: The PyTorch model to train
            device: Device to train on (cuda or cpu)
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model
        self.device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        print(f"Using device: {self.device}")
    
    def train_with_finetuning(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        class_weights: Optional[torch.Tensor] = None,
        patience: int = 10,
    ) -> Dict:
        """
        Finetune the pretrained model on labeled data.
        
        Args:
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set
            test_loader: DataLoader for test set
            num_epochs: Maximum number of training epochs
            learning_rate: Initial learning rate for optimizer
            weight_decay: L2 regularization strength
            class_weights: Tensor of class weights for handling imbalance
            patience: Number of epochs without improvement before early stopping
            
        Returns:
            Dictionary with training history and metrics
        """
        print("\n" + "="*60)
        print("Starting Model Finetuning on Labeled Tactic Data")
        print("="*60)
        
        # Setup optimizer: Adam is good for fine-tuning pretrained models
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup learning rate scheduler: reduce LR if validation plateaus
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3
        )
        
        # Loss function with class weights to handle imbalanced data
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        # Tracking metrics throughout training
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "best_val_acc": 0,
            "best_epoch": 0
        }
        
        no_improvement_count = 0
        
        # Main training loop
        for epoch in range(1, num_epochs + 1):
            # Training phase: update model weights
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, loss_fn
            )
            
            # Validation phase: evaluate on held-out data
            val_loss, val_metrics = self._validate_epoch(val_loader, loss_fn)
            
            # Extract validation accuracy
            val_acc = val_metrics["accuracy"]
            
            # Update learning rate based on validation performance
            scheduler.step(val_acc)
            
            # Track history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            
            # Check for improvement
            if val_acc > history["best_val_acc"]:
                history["best_val_acc"] = val_acc
                history["best_epoch"] = epoch
                no_improvement_count = 0
                # Save best model checkpoint
                self._save_checkpoint(epoch, is_best=True)
            else:
                no_improvement_count += 1
            
            # Print progress every epoch
            if epoch % 1 == 0:
                print(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.2%} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.2%}"
                )
            
            # Early stopping: stop if no improvement for 'patience' epochs
            if no_improvement_count >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch}. "
                    f"No improvement for {patience} epochs."
                )
                break
        
        # Evaluate on test set with best model
        print("\n" + "="*60)
        print("Evaluating Best Model on Test Set")
        print("="*60)
        test_metrics = self._evaluate(test_loader)
        history["test_metrics"] = test_metrics
        
        return history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module
    ) -> Tuple[float, float]:
        """
        Execute one training epoch.
        
        Updates model weights using backpropagation on training data.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer for updating weights
            loss_fn: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()  # Set model to training mode (enables dropout, etc.)
        total_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            # Move data to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass: compute predictions
            logits = self.model(features)
            
            # Compute loss
            loss = loss_fn(logits, labels)
            
            # Backward pass: compute gradients
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Backpropagate error
            optimizer.step()  # Update weights
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        loss_fn: nn.Module
    ) -> Tuple[float, Dict]:
        """
        Execute validation phase without updating weights.
        
        Args:
            val_loader: Validation data loader
            loss_fn: Loss function
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()  # Set model to evaluation mode (disables dropout, etc.)
        total_loss = 0.0
        correct = 0
        total = 0
        
        # No gradient computation during validation
        with torch.no_grad():
            for features, labels in val_loader:
                # Move data to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(features)
                
                # Compute loss
                loss = loss_fn(logits, labels)
                
                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0
        
        metrics = {"accuracy": accuracy, "loss": avg_loss}
        
        return avg_loss, metrics
    
    def _evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Comprehensive evaluation on test set with multiple metrics.
        
        Computes: accuracy, precision, recall, F1-score for multi-class classification.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        # Collect predictions on entire test set
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                logits = self.model(features)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        num_classes = self.model.num_classes
        
        # Compute metrics
        accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes)
        precision_metric = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        recall_metric = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        f1_metric = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        
        metrics = {
            "accuracy": accuracy_metric(all_predictions, all_labels).item(),
            "precision": precision_metric(all_predictions, all_labels).item(),
            "recall": recall_metric(all_predictions, all_labels).item(),
            "f1_score": f1_metric(all_predictions, all_labels).item(),
        }
        
        # Print metrics
        print(f"Test Accuracy: {metrics['accuracy']:.2%}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")
        print(f"Test F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint for resuming training or deployment.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        filename = os.path.join(
            self.checkpoint_dir,
            f"classifier_best.pt" if is_best else f"classifier_epoch_{epoch}.pt"
        )
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.model.input_dim,
            "hidden_dims": self.model.hidden_dims,
            "num_classes": self.model.num_classes,
            "dropout_rate": self.model.dropout_rate,
        }
        
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint: {filename}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from saved checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")


def train_pretrained_classifier(
    graph_root_dir: str = "",
    tactics_json_path: str = "",
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    train_split: float = 0.7,
    val_split: float = 0.15,
):
    """
    Args:
        graph_root_dir: Directory containing graph data
        tactics_json_path: Path to tactic definitions
        num_epochs: Maximum training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        train_split: Proportion of data for training (0.0-1.0)
        val_split: Proportion of remaining data for validation
    """
    print("\n" + "="*60)
    print("Initializing Pretrained Classifier Training Pipeline")
    print("="*60)
    
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    print("\nLoading raw graph data with labels...")
    dataset = TacticFeatureDataset(
        data_root_dir=graph_root_dir,
        tactics_json_path=tactics_json_path
    )
    
    # Print dataset statistics
    print(f"Total samples: {len(dataset)}")
    labels = [label for _, label in dataset]
    label_counts = Counter(labels)
    print(f"Label distribution: {dict(label_counts)}")
    
    # Compute class weights for handling imbalance
    unique_labels = np.array(sorted(set(labels)))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_labels,
        y=labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print(f"Class weights: {class_weights}")
    
    # Split dataset into train/val/test
    print("\nSplitting data into train/val/test sets...")
    train_size = int(train_split * len(dataset))
    remaining_size = len(dataset) - train_size
    val_size = int(val_split * remaining_size)
    test_size = remaining_size - val_size
    
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size]
    )
    
    print(f"Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    print("\nInitializing pretrained model...")
    input_dim = dataset[0][0].shape[0]  # Get feature dimension (max_nodes * 7 raw features per node)
    num_classes = len(dataset.label_to_id)
    
    # Scale hidden dimensions based on input size
    # Hidden dims should be progressively smaller than input_dim
    hidden_dims = [
        max(128, int(input_dim * 0.5)),  # First hidden layer: ~50% of input
        max(64, int(input_dim * 0.25)),   # Second hidden layer: ~25% of input
        32  # Final hidden layer before classification
    ]
    
    model = PretrainedTacticClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout_rate=0.3
    )
    
    print(f"Input dimension: {input_dim} (raw node features: max_nodes={dataset.max_nodes} * 7 features/node)")
    print(f"Hidden dimensions: {hidden_dims}")
    print(f"Number of classes: {num_classes}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = TacticClassifierTrainer(model, device=device)
    
    # Train with finetuning
    history = trainer.train_with_finetuning(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        class_weights=class_weights_tensor,
        patience=10
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    return model, dataset, history


# def predict_on_raw_data(
#     model: nn.Module,
#     feature_scaler,
#     label_to_id: Dict[str, int],
#     graph_data: Dict,
#     max_nodes: int,
#     device: torch.device = None
# ) -> Tuple[str, float]:
#     """
#     Predict tactic for a single raw graph sample using raw node data.
    
#     Args:
#         model: Trained classifier model
#         feature_scaler: Fitted StandardScaler for feature normalization
#         label_to_id: Mapping from strategy names to IDs
#         graph_data: Raw graph dictionary with node and edge data
#         max_nodes: Maximum number of nodes (used for padding)
#         device: Device to run prediction on
        
#     Returns:
#         Tuple of (predicted_tactic_name, confidence_score)
#     """
#     device = device or (
#         torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     )
#     model = model.to(device)
#     model.eval()
    
#     # Extract raw node features from the graph and pad to max_nodes
#     raw_features = _extract_raw_node_features_inference(graph_data, max_nodes)
    
#     # Normalize features using the fitted scaler
#     normalized_features = feature_scaler.transform(raw_features.reshape(1, -1))[0]
#     features_tensor = torch.tensor(
#         normalized_features, dtype=torch.float32
#     ).unsqueeze(0).to(device)
    
#     # Make prediction
#     with torch.no_grad():
#         logits = model(features_tensor)
#         probabilities = F.softmax(logits, dim=1)
#         predicted_id = torch.argmax(logits, dim=1).item()
#         confidence = probabilities[0, predicted_id].item()
    
#     # Map ID back to tactic name
#     id_to_label = {v: k for k, v in label_to_id.items()}
#     predicted_tactic = id_to_label.get(predicted_id, "unknown")
    
#     return predicted_tactic, confidence


# def _extract_raw_node_features_inference(graph_dict: Dict, max_nodes: int) -> np.ndarray:
#     """
#     Extract and pad raw node features for inference.
    
#     Args:
#         graph_dict: Graph data dictionary with nodes_data
#         max_nodes: Maximum nodes to pad to
        
#     Returns:
#         1D array of padded raw node features
#     """
#     nodes = graph_dict.get("nodes_data", {})
    
#     if not nodes:
#         return np.zeros(max_nodes * 7, dtype=np.float32)
    
#     # Extract raw features for each node
#     node_features_list = []
    
#     for node in nodes.values():
#         features = [
#             float(node.get("hp", 0)),
#             float(node.get("armor", 0)),
#             float(node.get("totalUtility", 0)),
#             float(node.get("x", 0)),
#             float(node.get("y", 0)),
#             float(node.get("isAlive", 0)),
#             float(node.get("hasBomb", 0)),
#         ]
#         node_features_list.append(features)
    
#     # Convert to array
#     node_array = np.array(node_features_list, dtype=np.float32)
    
#     # Pad to max_nodes
#     num_nodes = len(node_features_list)
#     padding_needed = max_nodes - num_nodes
    
#     if padding_needed > 0:
#         padding = np.zeros((padding_needed, 7), dtype=np.float32)
#         node_array = np.vstack([node_array, padding])
    
#     return node_array.flatten()


if __name__ == "__main__":
    model, dataset, history = train_pretrained_classifier(
        graph_root_dir = "data/preprocessed/de_dust2",
        tactics_json_path = "data/tactic_labels/de_dust2_tactics.json",
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
