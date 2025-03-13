import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class EmotionPredictor:
    def __init__(self, embeddings, labels):
        """
        Args:
            embeddings (np.array): Pre-computed embeddings of shape (n_samples, embedding_dim).
            labels (list or array-like): Corresponding emotion labels.
        """
        self.embeddings = np.array(embeddings)
        self.labels = labels

        # Encode categorical labels
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(labels)
        
        # Check that the number of samples match
        assert self.embeddings.shape[0] == len(self.y), "Mismatch between embeddings and labels."
        
        # Prepare training and testing datasets
        self._prepare_dataset()
        
        # Default hyperparameters and model save path
        self.batch_size = 8
        self.model = None
        self.model_path = "emotion_model.pth"

    def _prepare_dataset(self):
        """Splits the data into training and testing sets and creates TensorDatasets."""
        indices = np.arange(self.embeddings.shape[0])
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        X_train, X_test = self.embeddings[train_idx], self.embeddings[test_idx]
        y_train, y_test = self.y[train_idx], self.y[test_idx]
        
        self.train_data = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        self.test_data = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )

    # Define a more complex neural network architecture that accepts hidden sizes.
    class ComplexEmotionNet(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_emotions):
            """
            Args:
                input_size (int): Dimension of the input embeddings.
                hidden_size1 (int): Number of neurons for the first layer.
                hidden_size2 (int): Number of neurons for the second layer.
                hidden_size3 (int): Number of neurons for the third layer.
                num_emotions (int): Number of unique emotion classes.
            """
            super().__init__()
            # First block: fc1 -> BN -> Dropout -> ReLU
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.bn1 = nn.BatchNorm1d(hidden_size1)
            self.dropout1 = nn.Dropout(0.3)
            
            # Second block: fc2 -> BN -> Dropout -> ReLU
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.bn2 = nn.BatchNorm1d(hidden_size2)
            self.dropout2 = nn.Dropout(0.3)
            
            # Third block: fc3 -> BN -> Dropout -> ReLU
            self.fc3 = nn.Linear(hidden_size2, hidden_size3)
            self.bn3 = nn.BatchNorm1d(hidden_size3)
            self.dropout3 = nn.Dropout(0.3)
            
            # If dimensions differ for residual connection, create a projection layer
            if hidden_size2 != hidden_size3:
                self.res_proj = nn.Linear(hidden_size2, hidden_size3)
            else:
                self.res_proj = None

            # Fourth block: combine residual output, then fc4 -> BN -> ReLU
            # We'll set hidden_size4 to be half of hidden_size3 (or 1 if hidden_size3 is very small)
            hidden_size4 = max(1, hidden_size3 // 2)
            self.fc4 = nn.Linear(hidden_size3, hidden_size4)
            self.bn4 = nn.BatchNorm1d(hidden_size4)
            
            # Output layer
            self.fc_out = nn.Linear(hidden_size4, num_emotions)

        def forward(self, x):
            # First block
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout1(x)
            
            # Second block
            x2 = F.relu(self.bn2(self.fc2(x)))
            x2 = self.dropout2(x2)
            
            # Third block
            x3 = F.relu(self.bn3(self.fc3(x2)))
            x3 = self.dropout3(x3)
            
            # Residual connection: project x2 if necessary then add to x3
            if self.res_proj:
                x2_proj = self.res_proj(x2)
            else:
                x2_proj = x2
            x_res = x2_proj + x3
            
            # Fourth block
            x4 = F.relu(self.bn4(self.fc4(x_res)))
            
            logits = self.fc_out(x4)
            return logits

    def define_model(self, hidden_size1=100, hidden_size2=50, hidden_size3=25, load=False, model_path=None):
        """
        Initializes (or loads) the neural network model.
        
        Args:
            hidden_size1 (int): Number of neurons for the first layer.
            hidden_size2 (int): Number of neurons for the second layer.
            hidden_size3 (int): Number of neurons for the third layer.
            load (bool): Whether to load an existing model.
            model_path (str): Path to the model file.
        """
        num_emotions = len(np.unique(self.y))
        input_size = self.embeddings.shape[1]
        self.model = self.ComplexEmotionNet(input_size, hidden_size1, hidden_size2, hidden_size3, num_emotions)
        if model_path is not None:
            self.model_path = model_path
        if load and os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            print("Loaded saved model.")
        else:
            print(f"Initialized complex model with {num_emotions} emotion classes.")

    def train_model(self, epochs=500, lr=0.01):
        """
        Train the neural network on the emotion classification task.
        """
        if self.model is None:
            raise ValueError("Model is not defined. Call define_model() first.")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        train_loader = DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size)

        for epoch in range(epochs):
            running_loss = 0.0
            self.model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                logits = self.model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def predict(self, embeddings):
        """
        Predicts probability scores for each emotion for given embeddings.
        
        Args:
            embeddings (np.array): Pre-computed embeddings of shape (n_samples, embedding_dim) 
                                   or a single embedding (1D array).
        
        Returns:
            np.array: An array of probability scores (shape: [n_samples, num_emotions]).
        """
        self.model.eval()
        x_tensor = torch.tensor(embeddings, dtype=torch.float32)
        if len(x_tensor.shape) == 1:
            x_tensor = x_tensor.unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x_tensor)
            probabilities = F.softmax(logits, dim=1)
        return probabilities.cpu().numpy()

    def evaluate_model(self):
        """
        Evaluate the model on the test set and print the classification accuracy.
        """
        self.model.eval()
        correct = 0
        total = 0
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size)
        with torch.no_grad():
            for inputs, labels in test_loader:
                logits = self.model(inputs)
                _, predictions = torch.max(logits, 1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy*100:.2f}%")
