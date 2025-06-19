import re
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from model import Resnet, CNN, simpleCNN  # You can switch architectures here

def load_separately_with_names(file_type):
    """
    Load all datasets from an HDF5 file except 'q_fixed', extract numbers from dataset names (e.g., 'd20_qIq'),
    and return sorted (names, data) arrays.
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

def load_all_types(file_type3, file_type4):
    """
    Load Type 3 and Type 4 data separately, ensuring sorted order.
    """
    names3, data3 = load_separately_with_names(file_type3)
    names4, data4 = load_separately_with_names(file_type4)
    return (names3, data3), (names4, data4)

def augment_type_data(names, X, new_sample_count=2000):
    """
    Augment data using lognormal-weighted mixing from the original 81 SAXS curves.
    Each generated sample gets a synthetic label (mean diameter).
    """
    new_samples = []
    labels = []
    diameters = np.array(names, dtype=np.float32)
    for _ in range(new_sample_count):
        mean_real = np.random.uniform(25, 95)
        sigma_real = np.random.uniform(0.1, 0.4)
        mu = np.log(mean_real ** 2 / np.sqrt(mean_real ** 2 + sigma_real ** 2))
        sigma_log = np.sqrt(np.log(1 + (sigma_real ** 2 / mean_real ** 2)))
        weights = (1.0 / (diameters * sigma_log * np.sqrt(2 * np.pi))) * \
                  np.exp(- (np.log(diameters) - mu) ** 2 / (2 * sigma_log ** 2))
        weights /= np.sum(weights)
        new_curve = np.sum(weights[:, np.newaxis] * X, axis=0)
        new_samples.append(new_curve)
        labels.append(mu)
    return np.array(new_samples), np.array(labels)

def add_noise_to_curve(curve):
    """
    Add lognormal (Poisson-like) noise to a SAXS curve.
    """
    curve_safe = np.maximum(curve, 1e-12)
    log_alpha = np.random.uniform(np.log(1e4), np.log(10 ** 7.5))
    alpha = np.exp(log_alpha)
    sigma2 = np.log(1 + alpha / curve_safe)
    epsilon = np.random.randn(curve.shape[0])
    noise_factor = np.exp(np.sqrt(sigma2) * epsilon - sigma2 / 2)
    return curve * noise_factor

def add_noise_to_data(data):
    """
    Add lognormal noise to each sample (row) in the dataset.
    """
    return np.array([add_noise_to_curve(sample) for sample in data])

def generate_heter_data(X0, labels0, X1, labels1, desired_count=1000):
    """
    Generate heterogeneous SAXS curves by mixing Type 3 and Type 4 data.
    For each mixed sample, return:
      - the mixed curve,
      - its alpha (mixing ratio),
      - and its weighted mean diameter (mu).
    """
    X_heter_list = []
    alpha_list = []
    mu_heter_list = []
    for _ in range(desired_count):
        idx0 = np.random.randint(len(X0))
        idx1 = np.random.randint(len(X1))
        alpha = np.random.rand()
        mixed_curve = alpha * X0[idx0] + (1 - alpha) * X1[idx1]
        mu_val = alpha * labels0[idx0] + (1 - alpha) * labels1[idx1]
        X_heter_list.append(mixed_curve)
        alpha_list.append(alpha)
        mu_heter_list.append(mu_val)
    return np.array(X_heter_list), np.array(alpha_list, dtype=np.float32), np.array(mu_heter_list, dtype=np.float32)

def load_hdf5_data_for_regression(file_type3, file_type4, desired_count=2000, heter_count=1000, target="alpha"):
    """
    Main data preparation for regression:
    - Augment Type 3 and Type 4 separately.
    - Mix them to create heterogeneous samples.
    - Add noise.
    - Return X and the selected target ("alpha" or "mu").
    """
    (names3, data3), (names4, data4) = load_all_types(file_type3, file_type4)
    X0, labels0 = augment_type_data(names3, data3, new_sample_count=desired_count)
    X1, labels1 = augment_type_data(names4, data4, new_sample_count=desired_count)
    X_heter, alphas, mu_heter = generate_heter_data(X0, labels0, X1, labels1, desired_count=heter_count)
    X_heter_noisy = add_noise_to_data(X_heter)
    if target == "alpha":
        labels = alphas
    elif target == "mu":
        labels = mu_heter
    else:
        raise ValueError("target must be 'alpha' or 'mu'")
    return X_heter_noisy, labels

def train_the_model(file_type3, file_type4, desired_count, heter_count):
    X, y = load_hdf5_data_for_regression(file_type3, file_type4, desired_count, heter_count, target="alpha")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Target stats: min = {:.3f}, max = {:.3f}, mean = {:.3f}".format(y.min(), y.max(), y.mean()))

    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)

    from sklearn.model_selection import train_test_split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(input_length=X.shape[1]).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 120
    patience = 20
    best_val_loss = float('inf')
    best_model_weights = None
    counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_losses.append(total_train_loss / len(train_loader))

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_losses[-1]:.6f}, Val Loss = {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Plot loss
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

    # Evaluate on test set
    model.eval()
    y_preds, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            y_preds.extend(outputs.cpu().numpy())
            y_true.extend(y_batch.numpy())

    y_preds = np.array(y_preds)
    y_true = np.array(y_true)
    print(f"Test MSE: {mean_squared_error(y_true, y_preds):.6f}, R²: {r2_score(y_true, y_preds):.6f}")

    plt.figure()
    plt.scatter(y_true, y_preds, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title("Regression: Prediction vs True")
    plt.grid(True)
    plt.show()

# Main entry point
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    file_type3 = "./output/clean3.h5"
    file_type4 = "./output/clean4.h5"
    desired_count = 3000
    heter_count = 3000
    train_the_model(file_type3, file_type4, desired_count, heter_count)
