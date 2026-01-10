"""
Time Distance Metric Learning (TDML) Baseline

Uses TimeEncoder with Fourier features to encode time, then decodes to SH coefficients.
This baseline tests whether learned time representations improve upon direct time input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path


class TDMLBaseline:
    """Time Distance Metric Learning baseline"""

    def __init__(self):
        # Add parent directories to path to import models
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

        from models.time_encoder import TimeEncoder

        self.time_encoder = TimeEncoder(embed_dim=16, use_fourier_features=True)
        self.decoder = nn.Linear(16, 27)
        self.trained = False

    def train_model(self, hours, sh_gt, epochs=1000, lr=0.001):
        """
        Train TDML

        Args:
            hours: [N] array of training hours
            sh_gt: [N, 27] array of ground truth SH coefficients
            epochs: Number of training epochs
            lr: Learning rate
        """
        hours_tensor = torch.tensor(hours, dtype=torch.float32)
        sh_tensor = torch.tensor(sh_gt, dtype=torch.float32)

        optimizer = torch.optim.Adam(
            list(self.time_encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr
        )

        self.time_encoder.train()
        self.decoder.train()

        for epoch in range(epochs):
            time_embed = self.time_encoder(hours_tensor)
            sh_pred = self.decoder(time_embed)
            loss = F.mse_loss(sh_pred, sh_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 200 == 0:
                print(f"      Epoch {epoch+1}/{epochs}: Loss = {loss.item():.6f}")

        self.trained = True
        self.time_encoder.eval()
        self.decoder.eval()

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
            hours_tensor = torch.tensor(hours_query, dtype=torch.float32)
            time_embed = self.time_encoder(hours_tensor)
            return self.decoder(time_embed).numpy()
