"""
Direct MLP Baseline

Maps time directly to SH coefficients: hour -> 27 SH coefficients
Uses a simple feed-forward network with ReLU activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DirectMLPBaseline(nn.Module):
    """Direct MLP: hour -> SH coefficients"""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 27)
        )
        self.trained = False

    def forward(self, hours):
        """
        Forward pass

        Args:
            hours: [N, 1] tensor of hours

        Returns:
            [N, 27] tensor of predicted SH coefficients
        """
        return self.mlp(hours)

    def train_model(self, hours, sh_gt, epochs=1000, lr=0.001):
        """
        Train MLP (renamed to avoid conflict with nn.Module.train)

        Args:
            hours: [N] array of training hours
            sh_gt: [N, 27] array of ground truth SH coefficients
            epochs: Number of training epochs
            lr: Learning rate
        """
        hours_tensor = torch.tensor(hours, dtype=torch.float32).unsqueeze(1)
        sh_tensor = torch.tensor(sh_gt, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()  # Set to training mode
        for epoch in range(epochs):
            sh_pred = self(hours_tensor)
            loss = F.mse_loss(sh_pred, sh_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 200 == 0:
                print(f"      Epoch {epoch+1}/{epochs}: Loss = {loss.item():.6f}")

        self.trained = True
        self.eval()  # Set to eval mode

    def predict(self, hours_query):
        """
        Predict SH at query hours

        Args:
            hours_query: [M] array of query hours

        Returns:
            [M, 27] array of predicted SH coefficients
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train_model() first.")

        with torch.no_grad():
            hours_tensor = torch.tensor(hours_query, dtype=torch.float32).unsqueeze(1)
            return self(hours_tensor).numpy()
