# __init__.py
from .utils import check_dependencies
from .data import load_data, preprocess_data, split_data
from .eda import perform_eda
from .model import VAE, build_vae_model
from .training import train_vae, validate_vae, plot_loss_curve, visualize_latent_space
from .generate import generate_samples, postprocess_generated_data
from .compare import compare_distributions

__all__ = [
    "check_dependencies",
    "load_data",
    "preprocess_data",
    "split_data",
    "perform_eda",
    "build_vae_model",
    "train_vae",
    "validate_vae",
    "plot_loss_curve",
    "visualize_latent_space",
    "generate_samples",
    "postprocess_generated_data",
    "compare_distributions",
] 

# utils.py
import importlib
import pkg_resources

def check_dependencies():
    required = {
        'pandas': '1.0.0', 'numpy': '1.18.0', 'torch': '1.6.0', 'scikit-learn': '0.22.0',
        'matplotlib': '3.1.0', 'seaborn': '0.10.0'
    }
    missing = []
    for lib, ver in required.items():
        try:
            pkg = importlib.import_module(lib)
            if pkg_resources.parse_version(pkg.__version__) < pkg_resources.parse_version(ver):
                print(f"{lib} version {pkg.__version__} is outdated, please upgrade to >= {ver}")
        except ImportError:
            missing.append(lib)
    if missing:
        raise ImportError(f"Missing libraries: {', '.join(missing)}. Please install them before proceeding.")

# data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df, drop_negative_only_cols=True, fill_method='median', log_transform_cols=None):
    if drop_negative_only_cols:
        df = df.drop(columns=[col for col in df.columns if (df[col] == -1).all()])
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if fill_method == 'median':
                val = df[df[col] != -1][col].median()
            elif fill_method == 'mode':
                val = df[df[col] != -1][col].mode()[0]
            else:
                val = 0
            df[col] = df[col].replace(-1, val)
    if log_transform_cols:
        for col in log_transform_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return df, torch.FloatTensor(scaled_data), scaler

def split_data(tensor_data, test_size=0.2, batch_size=128):
    train, val = train_test_split(tensor_data, test_size=test_size, random_state=42)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    return train_loader, val_loader

# eda.py
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df, log_transformed_cols=None):
    import math
    plt.figure(figsize=(20, 15))
    for i, col in enumerate(df.columns):
        plt.subplot(math.ceil(len(df.columns)/4), 4, i+1)
        if df[col].nunique() > 20:
            sns.histplot(df[col], kde=True, bins=30)
        else:
            sns.countplot(x=df[col])
        plt.title(col)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.show()
    if log_transformed_cols:
        sample_cols = log_transformed_cols[:4]  # limit for readability
        sns.pairplot(df[sample_cols].sample(min(500, len(df))))
        plt.show()

# model.py
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], latent_dim=20):
        super(VAE, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def build_vae_model(input_dim, hidden_dims=[256, 128, 64], latent_dim=20, device='cpu'):
    model = VAE(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
    return model.to(device)

# training.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

def train_vae(model, train_loader, optimizer, device, epoch, beta):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss = loss_function(recon, batch, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader.dataset)

def validate_vae(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            val_loss += loss_function(recon, batch, mu, logvar).item()
    return val_loss / len(val_loader.dataset)

def plot_loss_curve(train_losses, val_losses):
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.legend()
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def visualize_latent_space(model, data_loader, device):
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            latents.append(mu.cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    if latents.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=latents[:, 0], y=latents[:, 1], alpha=0.5)
        plt.title('2D Latent Space Representation')
        plt.show()

# generate.py
import torch
import pandas as pd
import numpy as np

def generate_samples(model, num_samples, latent_dim, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        generated = model.decode(z).cpu().numpy()
    return generated

def postprocess_generated_data(generated_data, scaler, columns, log_transformed_cols=None, int_cols=None):
    df = pd.DataFrame(scaler.inverse_transform(generated_data), columns=columns)
    if log_transformed_cols:
        for col in log_transformed_cols:
            if col in df.columns:
                df[col] = np.expm1(df[col])
    if int_cols:
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].round().astype(int)
    for col in ['Submit Time', 'Wait Time', 'Run Time', 'Requested Time']:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    return df

# compare.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compare_distributions(original_df, generated_df):
    for df in [original_df, generated_df]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
    common_cols = original_df.select_dtypes(include=[np.number]).columns.intersection(generated_df.columns)
    num_cols = len(common_cols)
    cols_per_row = 3
    rows = (num_cols + cols_per_row - 1) // cols_per_row
    plt.figure(figsize=(6 * cols_per_row, 4 * rows))
    for i, col in enumerate(common_cols):
        plt.subplot(rows, cols_per_row, i + 1)
        sns.kdeplot(original_df[col], label='Real', fill=True, color='blue')
        sns.kdeplot(generated_df[col], label='Generated', fill=True, color='orange')
        plt.title(f'Distribution of {col}')
        plt.legend()
    plt.tight_layout()
    plt.show()
