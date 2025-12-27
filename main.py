from pathlib import Path
from typing import cast

import mlflow
import mlflow.pytorch as ml_torch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TimeSeriesWindowDataset(Dataset):
    @staticmethod
    def make_windows(signal: np.ndarray, window_length: int, stride: int):
        windows = []
        for start in range(0, len(signal) - window_length + 1, stride):
            windows.append(signal[start : start + window_length])
        return np.stack(windows)

    def __init__(self, windows: np.ndarray):
        self.windows = torch.from_numpy(windows).float()

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        x = self.windows[idx].unsqueeze(0)
        return x


class Conv1dEncoder(nn.Module):
    def __init__(self, input_length: int, latent_dims: int):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(1, 16, 5, 2, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, 5, 2, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(32, 64, 3, 2, padding=1), nn.ReLU())
        self.flatten = nn.Flatten()

        self.flatten_out_length = self.__calculate_flatten_output(input_length)
        self.latent = nn.Linear(self.flatten_out_length, latent_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.latent(x)
        return x

    def __calculate_flatten_output(self, input_length: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, input_length)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flatten(x)
            return x.shape[1]


class Conv1dDecoder(nn.Module):
    def __init__(self, latent_dims: int, flattened_size: int):
        super().__init__()

        self.latent = nn.Linear(latent_dims, flattened_size)
        self.deconv3 = nn.Sequential(nn.ConvTranspose1d(64, 32, 3, 2, padding=1), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose1d(32, 16, 5, 2, padding=1), nn.ReLU())
        self.deconv1 = nn.Sequential(nn.ConvTranspose1d(16, 1, 5, 2, padding=1, output_padding=1), nn.ReLU())

        deconv3 = cast(nn.ConvTranspose1d, self.deconv3[0])
        self.feature_length = flattened_size // deconv3.in_channels
        assert self.feature_length * deconv3.in_channels == flattened_size, "Flattened size must be divisible by feature channels"
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(deconv3.in_channels, self.feature_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.latent(x)
        x = self.unflatten(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        return x


class Conv1dAutoencoder(nn.Module):
    def __init__(self, input_length: int, latent_dims: int) -> None:
        super().__init__()
        self.encoder = Conv1dEncoder(input_length, latent_dims)
        self.decoder = Conv1dDecoder(latent_dims, self.encoder.flatten_out_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder(
    model: Conv1dAutoencoder,
    train_loader: torch.utils.data.DataLoader[TimeSeriesWindowDataset],
    val_loader: torch.utils.data.DataLoader[TimeSeriesWindowDataset],
    input_length: int,
    latent_dims: int,
    num_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, list]:

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="mean")

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    with mlflow.start_run():
        # --------------------
        # Log hyperparameters
        # --------------------
        mlflow.log_params(
            {
                "model": "Conv1dAutoencoder",
                "input_length": input_length,
                "latent_dims": latent_dims,
                "batch_size": batch_size,
                "learning_rate": lr,
                "epochs": num_epochs,
                "optimizer": "Adam",
                "loss": "MSE",
            }
        )

        # Log model summary–style info
        mlflow.log_param("encoder_flattened_size", model.encoder.flatten_out_length)

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            # --------------------
            # Training
            # --------------------
            model.train()
            train_loss = 0.0

            for x in train_loader:
                x = x.to(device)

                optimizer.zero_grad()
                x_hat = model(x)
                loss = criterion(x_hat, x)
                loss.backward()

                # Optional but good for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                train_loss += loss.item() * x.size(0)

            train_loss /= len(train_loader.dataset)
            history["train_loss"].append(train_loss)

            # --------------------
            # Validation
            # --------------------
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for x in val_loader:
                    x = x.to(device)
                    x_hat = model(x)
                    loss = criterion(x_hat, x)
                    val_loss += loss.item() * x.size(0)

            val_loss /= len(val_loader.dataset)
            history["val_loss"].append(val_loss)

            # --------------------
            # MLflow logging
            # --------------------
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                step=epoch,
            )

            pbar.set_postfix({"Train": train_loss, "Val": val_loss})

        # --------------------
        # Save trained model
        # --------------------
        ml_torch.log_model(
            model,
            name="model",
            pip_requirements=[
                "torch==2.9.1+cu126",
            ],
        )

    return history


def main():
    LATENT_DIMS = 32
    WINDOW = 1024
    STRIDE = WINDOW // 2

    encoder = Conv1dAutoencoder(WINDOW, LATENT_DIMS)

    df = pd.read_csv(Path(__file__).parent / "data" / "baseline.csv", index_col=0)
    # We only want normal data for now
    normal_df = df[(df["system_anomalies"] == "()") & (df["observation_anomalies"] == "()")].reset_index(drop=True)

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_df, test_df = train_test_split(  # pyright: ignore
        normal_df,
        test_size=0.3,
        shuffle=False,  # IMPORTANT for time series
    )

    train_signal = train_df["observed"].to_numpy(dtype=np.float32)
    test_signal = test_df["observed"].to_numpy(dtype=np.float32)

    train_windows = TimeSeriesWindowDataset.make_windows(train_signal, WINDOW, STRIDE)
    test_windows = TimeSeriesWindowDataset.make_windows(test_signal, WINDOW, STRIDE)

    train_dataset = TimeSeriesWindowDataset(train_windows)
    test_dataset = TimeSeriesWindowDataset(test_windows)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,  # OK here: windows are independent
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("conv1d_autoencoder_anomaly_baseline")
    train_autoencoder(encoder, train_loader, test_loader, WINDOW, LATENT_DIMS, 50, lr=1e-3)


if __name__ == "__main__":
    main()
