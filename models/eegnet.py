"""PyTorch EEGNet-8,2 for binary motor imagery at 128 Hz."""

from copy import deepcopy

import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class EEGNet(nn.Module):
    def __init__(self, n_channels, n_samples, dropout):
        super().__init__()
        self.register_buffer("channel_mean", torch.zeros(1, n_channels, 1))
        self.register_buffer("channel_scale", torch.ones(1, n_channels, 1))
        self.classes_ = None

        self.temporal = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, 8, kernel_size=(1, 64), bias=False),
            nn.BatchNorm2d(8, eps=1e-3, momentum=0.01),
        )

        self.spatial = nn.Conv2d(
            8, 16, kernel_size=(n_channels, 1), groups=8, bias=False
        )
        self.spatial_pool = nn.Sequential(
            nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.separable = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(16, 16, kernel_size=(1, 16), groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(16 * (n_samples // 32), 2)

    def forward(self, x):
        # Standardize [trial, electrode, time] and add the Conv2d channel dimension.
        x = ((x - self.channel_mean) / self.channel_scale).unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial_pool(self.spatial(x))
        x = self.separable(x)
        return self.classifier(x.flatten(1))

    @torch.inference_mode()
    def predict(self, X):
        # Predict in batches and map class indices back to the original labels.
        self.eval()
        inputs = torch.as_tensor(X, dtype=torch.float32)
        device = self.channel_mean.device
        predictions = [
            self(batch.to(device)).argmax(dim=1).cpu()
            for batch in inputs.split(128)
        ]
        return self.classes_[torch.cat(predictions).numpy()]


def build_eegnet(n_channels, n_samples, *, dropout=0.5):
    return EEGNet(n_channels, n_samples, dropout)


def fit_eegnet(
    X_train, y_train, X_val, y_val, *, device, seed, dropout,
    batch_size, max_epochs, patience, learning_rate, weight_decay,
):
    torch.manual_seed(seed)
    labels = LabelEncoder().fit(y_train)

    train_x = torch.as_tensor(X_train, dtype=torch.float32)
    val_x = torch.as_tensor(X_val, dtype=torch.float32)
    model = build_eegnet(*train_x.shape[1:], dropout=dropout).to(device)
    model.classes_ = labels.classes_

    scale, mean = torch.std_mean(train_x, dim=(0, 2), correction=0, keepdim=True)
    model.channel_mean.copy_(mean)
    model.channel_scale.copy_(scale.masked_fill(scale == 0, 1.0))

    train_loader = DataLoader(
        TensorDataset(train_x, torch.from_numpy(labels.transform(y_train)).long()),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        TensorDataset(val_x, torch.from_numpy(labels.transform(y_val)).long()),
        batch_size=batch_size,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for _ in range(max_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.spatial.weight.renorm_(p=2, dim=0, maxnorm=1.0)
                model.classifier.weight.renorm_(p=2, dim=0, maxnorm=0.25)

        model.eval()
        with torch.inference_mode():
            validation_loss = torch.zeros((), device=device)
            for x, y in val_loader:
                validation_loss += F.cross_entropy(
                    model(x.to(device)), y.to(device), reduction="sum"
                )
            validation_loss = (validation_loss / len(val_x)).item()

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("EEGNet training produced no finite loss.")
    model.load_state_dict(best_state)
    return model
