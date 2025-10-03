import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# --- Setup ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# GPU/CPU Handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data saving structure
experiment_data = {
    "synthetic_solar_cell": {
        "metrics": {"train_mae": [], "val_mae": [], "train_r2": [], "val_r2": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    },
}


# --- 1. Data Generation ---
def generate_synthetic_data(num_samples=2000):
    """Generates synthetic data for solar cell efficiency based on geometry."""
    np.random.seed(42)
    # Features: geometry_type (0: flat, 1: cylindrical, 2: spherical), radius, thickness
    geometry_type = np.random.randint(0, 3, num_samples)
    radius = np.random.uniform(10, 100, num_samples)  # mm
    thickness = np.random.uniform(0.1, 1.0, num_samples)  # mm

    # Target: Power Conversion Efficiency (PCE)
    # Model PCE based on hypothesis
    base_pce = 10.0
    noise = np.random.normal(0, 0.25, num_samples)

    # Flat cells (type 0): base efficiency
    pce = np.where(geometry_type == 0, base_pce, 0)

    # Cylindrical cells (type 1): higher efficiency, dependent on radius and thickness
    pce_cyl = (
        base_pce + 2.5 * (1 - np.exp(-thickness / 0.5)) - 0.002 * (radius - 55) ** 2
    )
    pce = np.where(geometry_type == 1, pce_cyl, pce)

    # Spherical cells (type 2): highest efficiency, different optimal parameters
    pce_sph = (
        base_pce + 4.0 * (1 - np.exp(-thickness / 0.4)) - 0.004 * (radius - 45) ** 2
    )
    pce = np.where(geometry_type == 2, pce_sph, pce)

    pce += noise
    pce = np.clip(pce, 5, 20)  # Clip PCE to a realistic range

    # Set radius to 0 for flat cells to make it a non-feature
    radius[geometry_type == 0] = 0

    features = np.stack([geometry_type, radius, thickness], axis=1)
    return features, pce.reshape(-1, 1)


X, y = generate_synthetic_data()

# --- 2. Data Preparation ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.FloatTensor(y_val)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# --- 3. Model Definition ---
class SolarCellMLP(nn.Module):
    def __init__(self, input_size):
        super(SolarCellMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


# --- 4. Training and Evaluation ---
input_features = X_train.shape[1]
model = SolarCellMLP(input_features).to(device)
criterion = nn.MSELoss()
mae_loss_fn = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

for epoch in range(num_epochs):
    # Training loop
    model.train()
    train_loss_sum = 0
    train_mae_sum = 0
    all_train_preds = []
    all_train_targets = []
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * features.size(0)
        train_mae_sum += mae_loss_fn(outputs, targets).item() * features.size(0)
        all_train_preds.append(outputs.detach().cpu().numpy())
        all_train_targets.append(targets.cpu().numpy())

    avg_train_loss = train_loss_sum / len(train_loader.dataset)
    avg_train_mae = train_mae_sum / len(train_loader.dataset)
    train_r2 = r2_score(
        np.concatenate(all_train_targets), np.concatenate(all_train_preds)
    )

    # Validation loop
    model.eval()
    val_loss_sum = 0
    val_mae_sum = 0
    all_val_preds = []
    all_val_targets = []
    with torch.no_grad():
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)

            val_loss_sum += criterion(outputs, targets).item() * features.size(0)
            val_mae_sum += mae_loss_fn(outputs, targets).item() * features.size(0)
            all_val_preds.append(outputs.cpu().numpy())
            all_val_targets.append(targets.cpu().numpy())

    avg_val_loss = val_loss_sum / len(val_loader.dataset)
    avg_val_mae = val_mae_sum / len(val_loader.dataset)
    final_val_preds = np.concatenate(all_val_preds)
    final_val_targets = np.concatenate(all_val_targets)
    val_r2 = r2_score(final_val_targets, final_val_preds)

    print(
        f"Epoch {epoch+1}/{num_epochs}: validation_loss = {avg_val_loss:.4f}, val_mae = {avg_val_mae:.4f}, val_r2 = {val_r2:.4f}"
    )

    # Store data
    experiment_data["synthetic_solar_cell"]["losses"]["train"].append(avg_train_loss)
    experiment_data["synthetic_solar_cell"]["losses"]["val"].append(avg_val_loss)
    experiment_data["synthetic_solar_cell"]["metrics"]["train_mae"].append(
        avg_train_mae
    )
    experiment_data["synthetic_solar_cell"]["metrics"]["val_mae"].append(avg_val_mae)
    experiment_data["synthetic_solar_cell"]["metrics"]["train_r2"].append(train_r2)
    experiment_data["synthetic_solar_cell"]["metrics"]["val_r2"].append(val_r2)
    experiment_data["synthetic_solar_cell"]["epochs"].append(epoch + 1)

# Store final predictions and ground truth for validation set
experiment_data["synthetic_solar_cell"]["predictions"] = final_val_preds
experiment_data["synthetic_solar_cell"]["ground_truth"] = final_val_targets


# --- 5. Final Evaluation and Visualization ---
final_val_r2 = experiment_data["synthetic_solar_cell"]["metrics"]["val_r2"][-1]
print("\n--- Final Evaluation ---")
print(
    f"Final R-squared Score on Power Conversion Efficiency (Validation): {final_val_r2:.4f}"
)

# Plotting Predicted vs True values
plt.figure(figsize=(8, 8))
plt.scatter(final_val_targets, final_val_preds, alpha=0.5)
plt.plot(
    [min(y_val.min(), final_val_preds.min()), max(y_val.max(), final_val_preds.max())],
    [min(y_val.min(), final_val_preds.min()), max(y_val.max(), final_val_preds.max())],
    "--",
    color="red",
    label="Ideal Fit",
)
plt.title("Predicted vs. True PCE on Validation Set")
plt.xlabel("True Power Conversion Efficiency (%)")
plt.ylabel("Predicted Power Conversion Efficiency (%)")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.tight_layout()
plot_path = os.path.join(working_dir, "pce_prediction_scatter.png")
plt.savefig(plot_path)
print(f"Saved prediction plot to {plot_path}")

# Plotting training history
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (MSE)", color="tab:blue")
ax1.plot(
    experiment_data["synthetic_solar_cell"]["epochs"],
    experiment_data["synthetic_solar_cell"]["losses"]["train"],
    label="Train Loss (MSE)",
    color="tab:blue",
)
ax1.plot(
    experiment_data["synthetic_solar_cell"]["epochs"],
    experiment_data["synthetic_solar_cell"]["losses"]["val"],
    label="Validation Loss (MSE)",
    color="tab:cyan",
    linestyle="--",
)
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.set_ylabel("R-squared Score", color="tab:red")
ax2.plot(
    experiment_data["synthetic_solar_cell"]["epochs"],
    experiment_data["synthetic_solar_cell"]["metrics"]["val_r2"],
    label="Validation R-squared Score",
    color="tab:red",
)
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylim(0, 1)
ax2.legend(loc="upper right")

plt.title("Model Training History")
fig.tight_layout()
history_plot_path = os.path.join(working_dir, "training_history.png")
plt.savefig(history_plot_path)
print(f"Saved training history plot to {history_plot_path}")

# --- 6. Save Data ---
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
