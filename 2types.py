import re
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_separately_with_names(file_path):
    """
        Read all datasets in an HDF5 file except 'q_fixed', extract numbers from dataset names,
        and return a list of names (numbers) and a corresponding array of data.
        Assumes dataset names follow the pattern "d20_qIq", "d100_qIq", etc.
        """
    names = []
    data_list = []
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            if key == 'q_fixed':
                continue
            match = re.search(r'd(\d+)_qIq', key)
            if match:
                num = int(match.group(1))
                names.append(num)
                data_list.append(f[key][:])
    sorted_indices = np.argsort(names)
    sorted_names = [names[i] for i in sorted_indices]
    sorted_data = np.vstack([data_list[i] for i in sorted_indices])
    return sorted_names, sorted_data

def load_all_types(file_type3, file_type4):
    """
        Load datasets for Type 3 and Type 4 separately and return two tuples: (names, data)
    """
    names3, data3 = load_separately_with_names(file_type3)
    names4, data4 = load_separately_with_names(file_type4)
    return (names3, data3), (names4, data4)

def augment_type_data(names, X, new_sample_count=2000):
    """
        Augment data for one type.
        For each sample, generate a log-normal weighted sum of the original data using a random mean and sigma.
        Labels are not needed here since the type label will be assigned externally.
    """
    new_samples = []
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
    return np.array(new_samples), None

# Apply lognormal noise
def add_noise_to_curve(curve):
    """
        Add lognormal noise to a single scattering curve using Poisson-like scaling.
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
        Apply add_noise_to_curve to each curve in the dataset.
    """
    return np.array([add_noise_to_curve(data[i]) for i in range(data.shape[0])])

# Mix samples from two types
def generate_heter_data(X0, _, X1, __, desired_count=1000):
    """
        Generate heterogeneous samples by linearly mixing random pairs of Type 3 and Type 4 samples.
        Labels for mixture ratios are ignored; all are treated as a new class in classification.
    """
    n0, n1 = X0.shape[0], X1.shape[0]
    X_heter_list = []
    for _ in range(desired_count):
        idx0 = np.random.randint(n0)
        idx1 = np.random.randint(n1)
        alpha = np.random.rand()
        mixed_curve = alpha * X0[idx0] + (1 - alpha) * X1[idx1]
        X_heter_list.append(mixed_curve)
    return np.array(X_heter_list), None, None

# Generate and prepare dataset for classification
def load_hdf5_data_for_classification(file_type3, file_type4, desired_count=2000, heter_count=1000):
    """
        Load, augment, add noise, and prepare training data for classification of 3 classes:
        Type 3 (label 0), Type 4 (label 1), and heterogeneous mixtures (label 2).
        Data is also transformed to log10 scale after noise injection.
    """
    (names3, data3), (names4, data4) = load_all_types(file_type3, file_type4)
    X_type3, _ = augment_type_data(names3, data3, desired_count)
    X_type4, _ = augment_type_data(names4, data4, desired_count)
    X_heter, _, _ = generate_heter_data(X_type3, None, X_type4, None, heter_count)
    y_type3 = np.zeros(X_type3.shape[0], dtype=np.int64)
    y_type4 = np.ones(X_type4.shape[0], dtype=np.int64)
    y_heter = np.full(X_heter.shape[0], 2, dtype=np.int64)
    X_type3 = np.log10(np.maximum(add_noise_to_data(X_type3), 1e-12))
    X_type4 = np.log10(np.maximum(add_noise_to_data(X_type4), 1e-12))
    X_heter = np.log10(np.maximum(add_noise_to_data(X_heter), 1e-12))
    X_all = np.concatenate([X_type3, X_type4, X_heter], axis=0)
    y_all = np.concatenate([y_type3, y_type4, y_heter], axis=0)
    return X_all, y_all

# CNN Model Definition
class ClassificationCNN(nn.Module):
    """
    CNN model definition for classifying SAXS curves into 3 classes.
    """
    def __init__(self, input_length, num_classes=3):
        super(ClassificationCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * input_length, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_the_model_classification(file_type3, file_type4, desired_count):
    """
    End-to-end training routine:
    - Load data and augment
    - Add noise and transform
    - Normalize and split into train/val/test
    - Train CNN with early stopping
    - Plot loss curves
    - Evaluate and print classification metrics
    """

    X, y = load_hdf5_data_for_classification(file_type3, file_type4, desired_count, desired_count)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Class counts:", {cls: int(np.sum(y == cls)) for cls in np.unique(y)})

    # Normalize input features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train/val/test
    from sklearn.model_selection import train_test_split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassificationCNN(input_length=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
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
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
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
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best Val Loss: {best_val_loss:.6f}")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Plot training vs validation loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.show()

    # Evaluate on test set
    model.eval()
    y_preds, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predicted = torch.argmax(outputs, dim=1)
            y_preds.extend(predicted.cpu().numpy())
            y_true.extend(y_batch.numpy())

    print("Test Accuracy:", accuracy_score(y_true, y_preds))
    print("Classification Report:")
    print(classification_report(y_true, y_preds))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_preds)
    print(cm)

    # Confusion matrix heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    classes = ['Type 3', 'Type 4', 'Heter']
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

# Entry point
if __name__ == "__main__":
    np.random.seed(31)
    torch.manual_seed(31)
    file_type3 = "./output/clean3.h5"
    file_type4 = "./output/clean4.h5"
    desired_count = 2000
    train_the_model_classification(file_type3, file_type4, desired_count)
