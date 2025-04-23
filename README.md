# 🚀 workload_vae

> A modular and extensible Python library for processing and generating synthetic workload data/logs of real parallel workloads from production systems using Variational Autoencoders (VAEs) 🔬📊

---

## 📚 Overview

`workload_vae` provides an end-to-end pipeline to:

- ✅ Load and clean CSV workload logs in SWF (Standard Workload Format)
- 📊 Perform exploratory data analysis (EDA)
- 🧠 Build and train deep VAE models for learning latent representations
- 🧪 Generate synthetic workload data
- 📈 Compare real vs. synthetic distributions for statistical fidelity

Whether you're a researcher, engineer, or data scientist working in performance modeling, resource scheduling, or workload synthesis — this library is for you.

---

## 📦 Installation

You can install this library directly from GitHub using:

```bash
pip install git+https://github.com/RakeshHG/workload_vae.git
```

---

## 🧰 Features

| Feature                       | Description |
|------------------------------|-------------|
| 🧹 Data Cleaning              | Impute missing values, drop invalid fields, apply transformations |
| 📊 Visual EDA                | Histograms, count plots, correlation heatmaps |
| 🧠 VAE Architecture           | Encoder, decoder, latent space, customizable layers |
| 🏋️‍♀️ Training & Validation   | Train loop with β-VAE regularization, loss plots |
| 🧬 Sample Generation          | Draw from latent space to synthesize new job entries |
| 🧪 Distribution Comparison    | KDE-based plots of feature distributions |

---

## 🧪 Example Usage

```python
from workload_vae import (
    load_data, preprocess_data, split_data,
    perform_eda, build_vae_model,
    train_vae, validate_vae,
    generate_samples, postprocess_generated_data,
    compare_distributions
)

# Step 1: Load & preprocess
df = load_data("workload.csv")
df_clean, tensor_data, scaler = preprocess_data(df, log_transform_cols=["Run Time", "Requested Time"])

# Step 2: EDA
perform_eda(df_clean, log_transformed_cols=["Run Time", "Requested Time"])

# Step 3: Train/val split
train_loader, val_loader = split_data(tensor_data)

# Step 4: Build model
model = build_vae_model(input_dim=tensor_data.shape[1], latent_dim=16)

# Step 5: Training loop (simplified)
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=1e-3)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(20):
    train_loss = train_vae(model, train_loader, optimizer, device, epoch, beta=1.0)
    val_loss = validate_vae(model, val_loader, device)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.2f}, Val Loss = {val_loss:.2f}")

# Step 6: Generate new samples
synthetic_data = generate_samples(model, num_samples=1000, latent_dim=16, device=device)
df_synth = postprocess_generated_data(synthetic_data, scaler, columns=df.columns)

# Step 7: Compare distributions
compare_distributions(df_clean, df_synth)
```

---

## 🗂 Project Structure

```
workload_vae/
├── workload_vae/
│   ├── __init__.py
│   ├── utils.py                # Dependency checks
│   ├── data.py                 # Load, clean, scale
│   ├── eda.py                  # Plot histograms, heatmaps
│   ├── model.py                # VAE model architecture
│   ├── training.py             # Training loop, validation, plots
│   ├── generate.py             # Sampling, inverse scaling
│   ├── compare.py              # KDE comparisons
├── setup.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🧑‍💻 Development

To contribute:

```bash
# Clone and install in editable mode
git clone https://github.com/RakeshHG/workload_vae.git
cd workload_vae
pip install -e .
```

---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgements

This project is inspired by real-world HPC workloads and research into workload modeling with deep generative models. Kudos to the community for SWF datasets and PyTorch ❤️

---

