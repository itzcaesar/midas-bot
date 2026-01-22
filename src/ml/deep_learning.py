"""
Deep Learning Models for Time-Series Trading
LSTM and GRU models using PyTorch.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
sys.path.append('../..')

from core.logger import get_logger

logger = get_logger("mt5bot.ml.deep_learning")

# Import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed")


class LSTMModel(nn.Module):
    """LSTM model for time-series classification."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # LSTM output
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        out = self.fc(lstm_out[:, -1, :])
        return out


class GRUModel(nn.Module):
    """GRU model for time-series classification."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # GRU output
        gru_out, h_n = self.gru(x)
        
        # Use last hidden state
        out = self.fc(gru_out[:, -1, :])
        return out


class DeepLearningTrainer:
    """Trainer for deep learning models."""
    
    def __init__(
        self,
        model_type: str = 'lstm',
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        model_dir: str = "models"
    ):
        """
        Initialize trainer.
        
        Args:
            model_type: 'lstm' or 'gru'
            sequence_length: Number of time steps in input sequence
            hidden_size: LSTM/GRU hidden size
            num_layers: Number of LSTM/GRU layers
            learning_rate: Learning rate
            batch_size: Training batch size
            epochs: Number of training epochs
            model_dir: Directory for saving models
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")
        
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_mean = None
        self.scaler_std = None
    
    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/GRU input.
        
        Args:
            X: Features array
            y: Labels array
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features."""
        if fit:
            self.scaler_mean = np.mean(X, axis=(0, 1), keepdims=True)
            self.scaler_std = np.std(X, axis=(0, 1), keepdims=True) + 1e-8
        
        return (X - self.scaler_mean) / self.scaler_std
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history
        """
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        
        # Normalize
        X_train_seq = self.normalize_features(X_train_seq, fit=True)
        
        # Map labels: -1,0,1 -> 0,1,2
        y_train_seq = y_train_seq + 1
        
        # Create model
        input_size = X_train_seq.shape[2]
        if self.model_type == 'lstm':
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers
            )
        else:
            self.model = GRUModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers
            )
        
        self.model.to(self.device)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_tensor = torch.LongTensor(y_train_seq).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)  # No shuffle for time-series
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        history = {'loss': [], 'accuracy': []}
        
        logger.info(f"Training {self.model_type.upper()} model for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            epoch_loss = total_loss / len(dataloader)
            epoch_acc = correct / total
            
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2%}")
        
        # Validation
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            history['val_accuracy'] = val_metrics['accuracy']
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features array
            
        Returns:
            Predictions (-1, 0, 1)
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        
        # Create sequences for the last sequence_length points
        if len(X.shape) == 2:
            # Single prediction - need sequence
            X_seq = X[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        else:
            X_seq = X
        
        X_seq = self.normalize_features(X_seq)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        # Map back: 0,1,2 -> -1,0,1
        return predicted.cpu().numpy() - 1
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features array
            
        Returns:
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        
        if len(X.shape) == 2:
            X_seq = X[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        else:
            X_seq = X
        
        X_seq = self.normalize_features(X_seq)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs, dim=1)
        
        return proba.cpu().numpy()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        X_seq, y_seq = self.create_sequences(X_test, y_test)
        X_seq = self.normalize_features(X_seq)
        y_seq = y_seq + 1  # Map to 0,1,2
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.LongTensor(y_seq).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        y_pred = predicted.cpu().numpy()
        y_true = y_seq
        
        from sklearn.metrics import accuracy_score, f1_score
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def save(self, filename: str = None):
        """Save model to disk."""
        filename = filename or f"{self.model_type}_model.pt"
        filepath = self.model_dir / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filename: str = None) -> bool:
        """Load model from disk."""
        filename = filename or f"{self.model_type}_model.pt"
        filepath = self.model_dir / filename
        
        if not filepath.exists():
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.scaler_mean = checkpoint['scaler_mean']
        self.scaler_std = checkpoint['scaler_std']
        self.model_type = checkpoint['model_type']
        self.sequence_length = checkpoint['sequence_length']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        
        # Recreate model (need input_size from data)
        # This is a limitation - consider saving input_size
        
        logger.info(f"Model loaded from {filepath}")
        return True
