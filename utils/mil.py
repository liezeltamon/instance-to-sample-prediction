from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GatedAttention(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 32):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.V(x)
        u = self.U(x)
        scores = self.attention_weights(v * u).transpose(1, 0)
        weights = F.softmax(scores, dim=-1)
        z = torch.matmul(weights, x)
        return z, weights


class MILClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, attn_dim: int = 32):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.attn = GatedAttention(hidden_dim, attn_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, bag_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.feature_extractor(x)
        bag_ids, inverse_indices = torch.unique(bag_index, return_inverse=True)
        if bag_ids.numel() == 0:
            return torch.empty(0), torch.empty(0), bag_ids

        bag_representations = []
        attention_scores = torch.zeros(x.shape[0], device=x.device)
        for bag_idx in range(bag_ids.shape[0]):
            mask = inverse_indices == bag_idx
            bag_x = x[mask]
            z, weights = self.attn(bag_x)
            bag_representations.append(z)
            attention_scores[mask] = weights.squeeze(0)

        bag_tensor = torch.cat(bag_representations, dim=0)
        logits = self.classifier(bag_tensor).squeeze(-1)
        return logits, attention_scores, bag_ids


def train_mil_model(
    model: MILClassifier,
    x: np.ndarray,
    bag_index: np.ndarray,
    bag_labels: np.ndarray,
    n_epochs: int = 25,
    lr: float = 1e-3,
    device: str | None = None,
) -> tuple[MILClassifier, dict[str, float]]:
    device = get_device(device)
    model = model.to(device)

    x_tensor = torch.from_numpy(x).to(device)
    bag_index_tensor = torch.from_numpy(bag_index).long().to(device)
    bag_labels_tensor = torch.from_numpy(bag_labels).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {"loss": []}
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _, _ = model(x_tensor, bag_index_tensor)
        loss = criterion(logits, bag_labels_tensor)
        loss.backward()
        optimizer.step()
        history["loss"].append(loss.item())
    return model, {"final_loss": history["loss"][-1] if history["loss"] else 0.0}


def predict_mil(
    model: MILClassifier,
    x: np.ndarray,
    bag_index: np.ndarray,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = get_device(device)
    model = model.to(device)
    model.eval()

    x_tensor = torch.from_numpy(x).to(device)
    bag_index_tensor = torch.from_numpy(bag_index).long().to(device)
    with torch.no_grad():
        logits, attention, bag_ids = model(x_tensor, bag_index_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        attention = attention.cpu().numpy().astype(np.float32)
        bag_ids = bag_ids.cpu().numpy().astype(np.int64)
    return probabilities, attention, bag_ids
