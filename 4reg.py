import re
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sympy.physics.paulialgebra import epsilon
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import Resnet

def load_separately_with_names(file_type):
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

def load_all_4_types(file_type1, file_type2, file_type3, file_type4):
    names1, data1 = load_separately_with_names(file_type1)
    names2, data2 = load_separately_with_names(file_type2)
    names3, data3 = load_separately_with_names(file_type3)
    names4, data4 = load_separately_with_names(file_type4)
    return (names1, data1), (names2, data2), (names3, data3), (names4, data4)

def augment_type_data(names, X, new_sample_count=2000):
    new_samples = []
    diameters = np.array(names, dtype=np.float32)
    for _ in range(new_sample_count):
        mean_real = np.random.uniform(25, 95)
        sigma_real = np.random.uniform(1, 4)
        mu = np.log(mean_real**2 / np.sqrt(mean_real**2 + sigma_real**2))
        sigma_log = np.sqrt(np.log(1 + (sigma_real**2 / mean_real**2)))
        weights = (1.0 / (diameters * sigma_log * np.sqrt(2*np.pi))) * \
                  np.exp(- (np.log(diameters)-mu)**2 / (2*sigma_log**2))
        weights /= np.sum(weights)
        new_curve = np.sum(weights[:, np.newaxis] * X, axis=0)
        new_samples.append(new_curve)
    return np.array(new_samples), None

def add_noise_to_curve(curve):
    curve_safe = np.maximum(curve, 1e-12)
    log_alpha = np.random.uniform(np.log(1e4), np.log(10**7.5))
    alpha = np.exp(log_alpha)
    sigma2 = np.log(1 + alpha / curve_safe)
    epsilon = np.random.randn(len(curve))
    noise_factor = np.exp(np.sqrt(sigma2)*epsilon - sigma2/2)
    return curve * noise_factor

def add_noise_to_data(data):
    return np.array([add_noise_to_curve(x) for x in data])

def generate_weighted_mixture_data_4(X1, X2, X3, X4, sample_count=1000, add_noise=False, do_log10=False):
    n1, n2, n3, n4 = X1.shape[0], X2.shape[0], X3.shape[0], X4.shape[0]
    feature_dim = X1.shape[1]
    X_mix = np.zeros((sample_count, feature_dim), dtype=np.float32)
    alphas = np.zeros((sample_count, 4), dtype=np.float32)
    for i in range(sample_count):
        idx1 = np.random.randint(0, n1)
        idx2 = np.random.randint(0, n2)
        idx3 = np.random.randint(0, n3)
        idx4 = np.random.randint(0, n4)

        alpha = np.random.dirichlet([5, 5, 5, 1])

        alphas[i] = alpha
        curve = alpha[0]*X1[idx1] + alpha[1]*X2[idx2] + alpha[2]*X3[idx3] + alpha[3]*X4[idx4]
        if add_noise:
            curve = add_noise_to_curve(curve)
        if do_log10:
            curve = np.log10(np.maximum(curve, 1e-12))
        X_mix[i] = curve
    return X_mix, alphas

class WeightsRegressionCNN(nn.Module):
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
            nn.Linear(64*input_length, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
      x = x.unsqueeze(1)
      x = self.layer1(x)
      x = self.layer2(x)
      x = x.view(x.size(0), -1)
      logits = self.fc(x)
      sig_out = F.sigmoid(logits)
      norm_out = sig_out / (sig_out.sum(dim=1, keepdim=True) + 1e-8)
      return norm_out


def train_weight_regression(file_type1, file_type2, file_type3, file_type4,
                            each_type_count=500,
                            mix_count=4000,
                            add_noise=True,
                            do_log10=True):
    (names1, data1), (names2, data2), (names3, data3), (names4, data4) = \
        load_all_4_types(file_type1, file_type2, file_type3, file_type4)

    X1, _ = augment_type_data(names1, data1, new_sample_count=each_type_count)
    X2, _ = augment_type_data(names2, data2, new_sample_count=each_type_count)
    X3, _ = augment_type_data(names3, data3, new_sample_count=each_type_count)
    X4, _ = augment_type_data(names4, data4, new_sample_count=each_type_count)

    X_mix, alphas = generate_weighted_mixture_data_4(
        X1, X2, X3, X4,
        sample_count=mix_count,
        add_noise=add_noise,
        do_log10=do_log10
    )

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_mix, alphas, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_length = X_train.shape[1]
    model = WeightsRegressionCNN(input_length).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    epochs = 400
    patience = 40

    best_val_loss = float('inf')
    best_state_dict = None
    no_improve_count = 0

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print("Early stopping!")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    model.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred_alpha = model(X_batch)
            y_pred_list.append(pred_alpha.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())

    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)

    mse = np.mean((y_pred - y_true)**2)
    print("Test MSE:", mse)
    mae = np.mean(np.abs(y_pred - y_true))
    print("Test MAE:", mae)

    mse_per_dim = np.mean((y_pred - y_true)**2, axis=0)
    print("MSE per alpha-dim:", mse_per_dim)
    mae_per_dim = np.mean(np.abs(y_pred - y_true), axis=0)
    print("MAE per alpha-dim:", mae_per_dim)

    from sklearn.metrics import r2_score

    r2s = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(4)]
    r2_overall = r2_score(y_true.flatten(), y_pred.flatten())
    print("R² per alpha-dim:", r2s)
    print("Overall R²:", r2_overall)

    n_show = 8
    idx_list = np.random.choice(len(y_true), n_show, replace=False)
    print("\nShow random {} test samples:".format(n_show))
    for i in idx_list:
        print(f"True: {y_true[i]}, Pred: {y_pred[i]}")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    file_type1 = "./output/clean1.h5"
    file_type2 = "./output/clean2.h5"
    file_type3 = "./output/clean3.h5"
    file_type4 = "./output/clean4.h5"

    train_weight_regression(
        file_type1, file_type2, file_type3, file_type4,
        each_type_count=20000,
        mix_count=20000,
        add_noise=True,
        do_log10=True
    )