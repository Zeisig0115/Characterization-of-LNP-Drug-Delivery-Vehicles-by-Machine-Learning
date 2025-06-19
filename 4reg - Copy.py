import re
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# -----------------------------
# 1. Data Loading & Augmentation
# -----------------------------

def load_separately_with_names(file_type):
    """
    Load SAXS data from a single HDF5 file, excluding 'q_fixed'.
    Extract numeric identifiers from keys (e.g., 'd20_qIq') and return:
      - sorted_names: sorted list of diameters
      - sorted_data: corresponding data (stacked vertically)
    """
    names = []
    data_list = []
    with h5py.File(file_type, 'r') as f:
        for key in f.keys():
            if key == 'q_fixed':
                continue
            m = re.search(r'd(\d+)_qIq', key)
            if m:
                num = int(m.group(1))
                names.append(num)
                data_list.append(f[key][:])
    sorted_indices = np.argsort(names)
    sorted_names = [names[i] for i in sorted_indices]
    sorted_data = np.vstack([data_list[i] for i in sorted_indices])
    return sorted_names, sorted_data


def load_all_3_types(file_type1, file_type2, file_type3):
    """
    Load SAXS datasets from three HDF5 files.
    Returns (names, data) tuples for each file.
    """
    names1, data1 = load_separately_with_names(file_type1)
    names2, data2 = load_separately_with_names(file_type2)
    names3, data3 = load_separately_with_names(file_type3)
    return (names1, data1), (names2, data2), (names3, data3)


def augment_type_data(names, X, new_sample_count=2000):
    """
    Augment data by generating synthetic SAXS curves:
    - Sample log-normal weight distributions using random (mean, sigma),
    - Weight and combine real curves to generate synthetic ones.
    """
    new_samples = []
    diameters = np.array(names, dtype=np.float32)
    for _ in range(new_sample_count):
        mean_real = np.random.uniform(25, 95)
        sigma_real = np.random.uniform(1, 4)
        mu = np.log(mean_real**2 / np.sqrt(mean_real**2 + sigma_real**2))
        sigma_log = np.sqrt(np.log(1 + (sigma_real**2 / mean_real**2)))
        weights = (1.0 / (diameters * sigma_log * np.sqrt(2 * np.pi))) * \
                  np.exp(- (np.log(diameters) - mu)**2 / (2 * sigma_log**2))
        weights /= np.sum(weights)
        new_curve = np.sum(weights[:, np.newaxis] * X, axis=0)
        new_samples.append(new_curve)
    return np.array(new_samples), None


def add_noise_to_curve(curve):
    """
    Add multiplicative log-normal noise to a single SAXS curve.
    """
    curve_safe = np.maximum(curve, 1e-12)
    log_alpha = np.random.uniform(np.log(1e4), np.log(10**7.5))
    alpha = np.exp(log_alpha)
    sigma2 = np.log(1 + alpha / curve_safe)
    epsilon = np.random.randn(len(curve))
    noise_factor = np.exp(np.sqrt(sigma2) * epsilon - sigma2 / 2)
    return curve * noise_factor


def add_noise_to_data(data):
    """
    Apply noise to each curve in the dataset.
    """
    return np.array([add_noise_to_curve(x) for x in data])


def generate_weighted_mixture_data_3(X1, X2, X3, sample_count=1000, add_noise=False, do_log10=False):
    """
    Generate mixture data from three types:
    - Randomly sample one curve from each,
    - Generate Dirichlet weights,
    - Mix the curves using the weights.
    Optionally adds noise and log10 transform.
    Returns (X_mix, alphas).
    """
    n1, n2, n3 = X1.shape[0], X2.shape[0], X3.shape[0]
    feature_dim = X1.shape[1]
    X_mix = np.zeros((sample_count, feature_dim), dtype=np.float32)
    alphas = np.zeros((sample_count, 3), dtype=np.float32)

    for i in range(sample_count):
        idx1, idx2, idx3 = np.random.randint(0, n1), np.random.randint(0, n2), np.random.randint(0, n3)
        alpha = np.random.dirichlet([1, 1, 1])
        alphas[i] = alpha
        curve = alpha[0] * X1[idx1] + alpha[1] * X2[idx2] + alpha[2] * X3[idx3]
        if add_noise:
            curve = add_noise_to_curve(curve)
        if do_log10:
            curve = np.log10(np.maximum(curve, 1e-12))
        X_mix[i] = curve
    return X_mix, alphas


# -----------------------------
# 2. CNN Regression Model
# -----------------------------

class WeightsRegressionCNN(nn.Module):
    """
    CNN model to regress 3 mixing weights (alphas) from SAXS curves.
    Final output is soft-normalized to sum to 1.
    """
    def __init__(self, input_length):
        super(WeightsRegressionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * input_length, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, feature_dim)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        sig_out = torch.sigmoid(logits)
        norm_out = sig_out / (sig_out.sum(dim=1, keepdim=True) + 1e-8)
        return norm_out


# -----------------------------
# 3. Training Pipeline
# -----------------------------

def train_weight_regression(file_type1, file_type2, file_type3,
                            each_type_count=500,
                            mix_count=4000,
                            add_noise=True,
                            do_log10=True):
    """
    Train CNN to predict mixing weights of 3 SAXS types.
    """
    # Load & augment data
    (names1, data1), (names2, data2), (names3, data3) = load_all_3_types(file_type1, file_type2, file_type3)
    X1, _ = augment_type_data(names1, data1, each_type_count)
    X2, _ = augment_type_data(names2, data2, each_type_count)
    X3, _ = augment_type_data(names3, data3, each_type_count)

    # Generate synthetic mixtures
    X_mix, alphas = generate_weighted_mixture_data_3(X1, X2, X3, mix_count, add_noise, do_log10)

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_mix, alphas, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Create DataLoaders
    def to_loader(X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=512, shuffle=True)

    train_loader = to_loader(X_train, y_train)
    val_loader = to_loader(X_val, y_val)
    test_loader = to_loader(X_test, y_test)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeightsRegressionCNN(input_length=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    # Training loop with early stopping
    epochs = 800
    patience = 80
    best_val_loss = float('inf')
    best_state_dict = None
    no_improve_count = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = sum(
            criterion(model(X.to(device)), y.to(device)).item()
            for X, y in train_loader
        ) / len(train_loader)
        train_losses.append(total_train_loss)

        model.eval()
        with torch.no_grad():
            total_val_loss = sum(
                criterion(model(X.to(device)), y.to(device)).item()
                for X, y in val_loader
            ) / len(val_loader)
        val_losses.append(total_val_loss)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {total_train_loss:.6f}, Val Loss = {total_val_loss:.6f}")

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_state_dict = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print("Early stopping triggered.")
                break

    if best_state_dict:
        model.load_state_dict(best_state_dict)

    # Plot loss curve
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    # Final evaluation
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X, y in test_loader:
            y_pred.append(model(X.to(device)).cpu().numpy())
            y_true.append(y.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    print("Test MSE:", mse)
    print("Test MAE:", mae)

    print("MSE per alpha:", np.mean((y_pred - y_true) ** 2, axis=0))
    print("MAE per alpha:", np.mean(np.abs(y_pred - y_true), axis=0))

    r2 = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(3)]
    print("R² per alpha:", r2)
    print("Overall R²:", r2_score(y_true.flatten(), y_pred.flatten()))

    # Show some examples
    n_show = 8
    idx = np.random.choice(len(y_true), n_show, replace=False)
    print(f"\nShowing {n_show} random test samples:")
    for i in idx:
        print(f"True: {y_true[i]}, Pred: {y_pred[i]}")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    file_type1 = "./output/clean1.h5"
    file_type2 = "./output/clean2.h5"
    file_type3 = "./output/clean3.h5"

    train_weight_regression(
        file_type1, file_type2, file_type3,
        each_type_count=100000,
        mix_count=100000,
        add_noise=True,
        do_log10=False
    )
