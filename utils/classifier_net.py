import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class ClassifierNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[64, 32],
        dropout=0.3,
        lr=1e-3,
        epochs=10,
        optimizer_fn=torch.optim.Adam,
        device=None
    ):
        """
        Classifier Neural Network with training and evaluation support.

        Args:
            input_dim (int): Input feature size.
            output_dim (int): Number of classes (1 for binary).
            hidden_dims (list): List of hidden layer sizes.
            dropout (float): Dropout rate.
            lr (float): Learning rate.
            epochs (int): Training epochs.
            optimizer_fn (torch.optim.Optimizer): Optimizer class.
            device (str or torch.device): 'cpu' or 'cuda' or torch.device object.
        """
        super(ClassifierNet, self).__init__()
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.optimizer_fn = optimizer_fn
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            logits = self.forward(x)
            if self.output_dim == 1:
                return torch.sigmoid(logits)
            else:
                return F.softmax(logits, dim=1)

    def train_model(self, train_loader, val_loader=None):
        self.to(self.device)
        optimizer = self.optimizer_fn(self.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss() if self.output_dim == 1 else nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.forward(X_batch).squeeze()

                if self.output_dim == 1:
                    y_batch = y_batch.float()
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {epoch_loss / len(train_loader):.4f}")

            if val_loader:
                val_acc = self.evaluate(val_loader)
                print(f"           Validation Accuracy: {val_acc:.2f}%")

    def evaluate(self, data_loader):
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.forward(X_batch)
                if self.output_dim == 1:
                    preds = torch.sigmoid(logits).squeeze() > 0.5
                else:
                    preds = torch.argmax(logits, dim=1)

                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        return 100 * correct / total
