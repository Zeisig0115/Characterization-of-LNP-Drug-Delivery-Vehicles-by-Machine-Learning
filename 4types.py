import re
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#######################################
# 1. Data Loading and Preprocessing
#######################################

def load_separately_with_names(file_type):
    """
    Load SAXS data from a single HDF5 file, excluding 'q_fixed'.
    Extract numeric identifiers from keys and return:
      - sorted_names: list of numeric identifiers (sorted),
      - sorted_data: corresponding stacked data in sorted order.
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


def load_all_4_types(file_type1, file_type2, file_type3, file_type4):
    """
    Load SAXS data from four HDF5 files corresponding to clean1 to clean4.
    Returns a tuple of (names, data) for each type.
    """
    names1, data1 = load_separately_with_names(file_type1)
    names2, data2 = load_separately_with_names(file_type2)
    names3, data3 = load_separately_with_names(file_type3)
    names4, data4 = load_separately_with_names(file_type4)
    return (names1, data1), (names2, data2), (names3, data3), (names4, data4)


def augment_type_data(names, X, new_sample_count=2000):
    """
    Augment SAXS data for a single type by:
      - Randomly sampling log-normal weights based on diameters;
      - Using the weights to combine original curves into new synthetic samples.
    Returns (new_samples, None).
    """
    new_samples = []
    diameters = np.array(names, dtype=np.float32)

    for _ in range(new_sample_count):
        mean_real = np.random.uniform(25, 95)
        sigma_real = np.random.uniform(0.1, 0.4)
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
    Add realistic multiplicative noise to a single SAXS curve using log-normal noise.
    Returns the noisy curve.
    """
    curve_safe = np.maximum(curve, 1e-12)
    log_alpha = np.random.uniform(np.log(1e4), np.log(10**7.5))
    alpha = np.exp(log_alpha)
    sigma2 = np.log(1 + alpha / curve_safe)
    epsilon = np.random.randn(curve.shape[0])
    noise_factor = np.exp(np.sqrt(sigma2)*epsilon - sigma2/2)
    curve_noisy = curve * noise_factor
    return curve_noisy


def add_noise_to_data(data):
    """
    Add noise to every SAXS curve in the dataset.
    """
    noisy_data = np.array([add_noise_to_curve(data[i]) for i in range(data.shape[0])])
    return noisy_data


def generate_heter_data_4(X1, X2, X3, X4, desired_count=1000):
    """
    Generate synthetic heterogeneous samples by mixing one curve from each of the four types.
    Weighted sums are created with random normalized coefficients.
    Returns (X_heter, None).
    """
    n1, n2, n3, n4 = X1.shape[0], X2.shape[0], X3.shape[0], X4.shape[0]
    X_heter_list = []

    for _ in range(desired_count):
        idx1 = np.random.randint(n1)
        idx2 = np.random.randint(n2)
        idx3 = np.random.randint(n3)
        idx4 = np.random.randint(n4)
        alphas = np.random.rand(4)
        alphas /= np.sum(alphas)
        mixed_curve = (alphas[0] * X1[idx1]
                       + alphas[1] * X2[idx2]
                       + alphas[2] * X3[idx3]
                       + alphas[3] * X4[idx4])
        X_heter_list.append(mixed_curve)

    return np.array(X_heter_list), None


def load_hdf5_data_for_classification_5types(file_type1, file_type2, file_type3, file_type4,
                                            desired_count=2000, heter_count=1000):
    """
    Complete pipeline to prepare 5-class classification data:
      1) Load clean SAXS data from four HDF5 files;
      2) Augment each type to desired_count samples;
      3) Generate heterogeneous mixtures;
      4) Label the samples: Types 1-4 -> 0, Heter -> 1;
      5) Add noise, apply log10, and return all samples and labels.
    """
    (names1, data1), (names2, data2), (names3, data3), (names4, data4) = \
        load_all_4_types(file_type1, file_type2, file_type3, file_type4)

    X_type1, _ = augment_type_data(names1, data1, new_sample_count=desired_count)
    X_type2, _ = augment_type_data(names2, data2, new_sample_count=desired_count)
    X_type3, _ = augment_type_data(names3, data3, new_sample_count=desired_count)
    X_type4, _ = augment_type_data(names4, data4, new_sample_count=desired_count)

    X_heter, _ = generate_heter_data_4(X_type1, X_type2, X_type3, X_type4, desired_count=heter_count)

    # Adjust labels for binary classification (clean -> 0, heter -> 1)
    y_type1 = np.full(X_type1.shape[0], 0, dtype=np.int64)
    y_type2 = np.full(X_type2.shape[0], 0, dtype=np.int64)
    y_type3 = np.full(X_type3.shape[0], 0, dtype=np.int64)
    y_type4 = np.full(X_type4.shape[0], 0, dtype=np.int64)
    y_heter = np.full(X_heter.shape[0], 1, dtype=np.int64)

    # Add noise and apply log10 transform
    X_type1_noisy = np.log10(np.maximum(add_noise_to_data(X_type1), 1e-12))
    X_type2_noisy = np.log10(np.maximum(add_noise_to_data(X_type2), 1e-12))
    X_type3_noisy = np.log10(np.maximum(add_noise_to_data(X_type3), 1e-12))
    X_type4_noisy = np.log10(np.maximum(add_noise_to_data(X_type4), 1e-12))
    X_heter_noisy = np.log10(np.maximum(add_noise_to_data(X_heter), 1e-12))

    X_all = np.concatenate([X_type1_noisy, X_type2_noisy,
                            X_type3_noisy, X_type4_noisy,
                            X_heter_noisy], axis=0)
    y_all = np.concatenate([y_type1, y_type2, y_type3, y_type4, y_heter], axis=0)

    return X_all, y_all


#######################################
# 2. Define Classification CNN
#######################################

class ClassificationCNN(nn.Module):
    """
    A simple 1D CNN for SAXS-based classification with two convolutional layers
    followed by a fully connected classifier.
    """
    def __init__(self, input_length, num_classes=5):
        super(ClassificationCNN, self).__init__()
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
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # Input x shape: (batch_size, feature_dim)
        x = x.unsqueeze(1)  # -> (batch_size, 1, feature_dim)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x


#######################################
# 3. Training and Evaluation
#######################################

def train_the_model_classification_5(file_type1, file_type2, file_type3, file_type4, desired_count):
    """
    Train and evaluate the classification model on 5-class SAXS data:
    clean types (1-4) vs. heterogeneously mixed (5th class).
    Includes training loop, early stopping, and evaluation metrics.
    """
    X, y = load_hdf5_data_for_classification_5types(
        file_type1, file_type2, file_type3, file_type4,
        desired_count=desired_count, heter_count=desired_count
    )
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Class distribution:", {cls: int(np.sum(y == cls)) for cls in np.unique(y)})

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split dataset
    from sklearn.model_selection import train_test_split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Create DataLoaders
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Model, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassificationCNN(input_length=X.shape[1], num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop with early stopping
    epochs = 200
    patience = 40
    best_val_loss = float('inf')
    best_model_weights = None
    counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
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
        total_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
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

    accuracy = accuracy_score(y_true, y_preds)
    print("Test Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_true, y_preds))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_preds)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    classes = ["Double-ellipsoid", "Spherical", "x", "x", "x"]  # Replace 'x' with actual labels if needed
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


#######################################
# 4. Main Entry Point
#######################################
if __name__ == "__main__":
    np.random.seed(31)
    torch.manual_seed(31)

    # Modify paths as needed
    file_type1 = "./output/clean1.h5"
    file_type2 = "./output/clean2.h5"
    file_type3 = "./output/clean3.h5"
    file_type4 = "./output/clean4.h5"

    desired_count = 4000  # Augmented sample count per clean type

    train_the_model_classification_5(file_type1, file_type2, file_type3, file_type4, desired_count)
