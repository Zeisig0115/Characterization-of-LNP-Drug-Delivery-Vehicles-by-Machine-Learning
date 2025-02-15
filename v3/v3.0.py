import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, input_length):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加 channel 维度
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load dataset from HDF5 files
def load_hdf5_data(file_type3, file_type4):
    with h5py.File(file_type3, 'r') as f3, h5py.File(file_type4, 'r') as f4:
        dataset_keys = [key for key in f3.keys() if key != 'q_fixed']
        data_list = []
        labels = []

        for key in dataset_keys:
            data_list.append(f3[key][:])
            labels.append(3)  # Type3
            data_list.append(f4[key][:])
            labels.append(4)  # Type4

    data_array = np.vstack(data_list)
    labels = np.array(labels)
    return data_array, labels

# File paths
file_type3 = "./output/interpolated_type3.h5"
file_type4 = "./output/interpolated_type4.h5"
X, y = load_hdf5_data(file_type3, file_type4)

# Convert labels to binary: Type3 -> 0, Type4 -> 1
y = np.where(y == 3, 0, 1)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 再从训练集中划分出 validation set (例如，20%的训练数据作为验证集)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val_tensor   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
y_test_tensor  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create DataLoader for training, validation and test
train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
val_dataset   = data.TensorDataset(X_val_tensor, y_val_tensor)
test_dataset  = data.TensorDataset(X_test_tensor, y_test_tensor)
train_loader  = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader    = data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader   = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(input_length=X.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 记录每个 epoch 的训练损失和验证损失
train_losses = []
val_losses = []

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    train_epoch_loss = total_train_loss / len(train_loader)
    train_losses.append(train_epoch_loss)

    # 在验证集上评估模型
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_val_loss += loss.item()

    val_epoch_loss = total_val_loss / len(val_loader)
    val_losses.append(val_epoch_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

# 绘制每个 epoch 的训练损失和验证损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
plt.plot(range(1, epochs+1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# === 在测试集上评估模型 ===
model.eval()
y_preds = []
y_true = []
with torch.no_grad():
    for batch in test_loader:
        X_batch, y_batch = batch
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.sigmoid(outputs).cpu().numpy().flatten()
        y_preds.extend(preds)
        y_true.extend(y_batch.numpy().flatten())

# Convert probabilities to binary predictions
y_preds_binary = [1 if p > 0.5 else 0 for p in y_preds]

# Compute accuracy
accuracy = np.mean(np.array(y_preds_binary) == np.array(y_true))
print(f"Test Accuracy: {accuracy:.4f}")

# ==== 绘制混淆矩阵 ====
cm = confusion_matrix(y_true, y_preds_binary)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
