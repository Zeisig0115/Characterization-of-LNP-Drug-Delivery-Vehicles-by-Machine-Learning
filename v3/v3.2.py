import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

##############################################
# 数据加载与扩增相关函数
##############################################

def load_separately(file_type3, file_type4):
    """
    分别读取 Type3 与 Type4 中除 'q_fixed' 外的所有数据，并垂直堆叠。
    返回:
       X_type3: shape=(N3, feature_dim)
       X_type4: shape=(N4, feature_dim)
    """
    with h5py.File(file_type3, 'r') as f3, h5py.File(file_type4, 'r') as f4:
        dataset_keys = [key for key in f3.keys() if key != 'q_fixed']
        X_type3_list = []
        X_type4_list = []
        for key in dataset_keys:
            if key in f4:
                data3 = f3[key][:]
                data4 = f4[key][:]
                X_type3_list.append(data3)
                X_type4_list.append(data4)
        X_type3 = np.vstack(X_type3_list) if X_type3_list else np.array([])
        X_type4 = np.vstack(X_type4_list) if X_type4_list else np.array([])
        return X_type3, X_type4


def sample_or_augment_class(X, desired_count=1000):
    """
    对同类数据进行下采样或扩增，使样本数达到 desired_count。
    - 若 len(X) >= desired_count: 随机下采样
    - 若 len(X) < desired_count: 利用线性组合生成新样本
    """
    n = X.shape[0]
    if n == 0:
        raise ValueError("X 无数据。")
    if n == desired_count:
        return X
    elif n > desired_count:
        indices = np.random.choice(n, size=desired_count, replace=False)
        return X[indices]
    else:
        num_new = desired_count - n
        new_samples = []
        for _ in range(num_new):
            idx1 = np.random.randint(n)
            idx2 = np.random.randint(n)
            alpha = np.random.rand()
            new_samples.append(alpha * X[idx1] + (1 - alpha) * X[idx2])
        new_samples = np.array(new_samples, dtype=X.dtype)
        return np.vstack([X, new_samples])


def generate_heter_data(X0, X1, desired_count=1000):
    """
    生成 heterogeneous 数据：从 X0 与 X1 中各随机采样一行，
    用 alpha * row0 + (1-alpha) * row1 混合生成新样本，同时返回混合权重 alpha。
    """
    n0, n1 = X0.shape[0], X1.shape[0]
    X_heter_list = []
    alpha_list = []
    for _ in range(desired_count):
        idx0 = np.random.randint(n0)
        idx1 = np.random.randint(n1)

        alpha = np.random.rand()

        X_heter_list.append(alpha * X0[idx0] + (1 - alpha) * X1[idx1])
        alpha_list.append(alpha)
    return np.array(X_heter_list, dtype=X0.dtype), np.array(alpha_list, dtype=np.float32)


def load_hdf5_data_for_regression(file_type3, file_type4, desired_count=1000):
    """
    - 分别加载 Type3 和 Type4 数据；
    - 对每个类别进行扩增或下采样至 desired_count；
    - 生成 heterogeneous 数据，其混合比例 alpha 作为回归目标；
    - 仅返回通过 alpha 混合生成的数据 X2 及其目标 alphas。
    """
    X_type3, X_type4 = load_separately(file_type3, file_type4)
    if X_type3.size == 0 or X_type4.size == 0:
        raise ValueError("未找到有效的 Type3 或 Type4 数据。")

    # 扩增或下采样
    X0 = sample_or_augment_class(X_type3, desired_count)
    X1 = sample_or_augment_class(X_type4, desired_count)

    # 生成 heterogeneous 数据，混合比例 alpha 作为回归目标
    X2, alphas = generate_heter_data(X0, X1, desired_count)

    return X2, alphas


##############################################
# 定义更复杂的 1D CNN 回归模型
##############################################
class ComplexCNNRegression(nn.Module):
    def __init__(self, input_length):
        """
        input_length: 特征维度长度
        """
        super(ComplexCNNRegression, self).__init__()
        # 第一层卷积块：Conv1d -> BatchNorm -> ReLU -> MaxPool
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # 第二层卷积块
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # 第三层卷积块
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # 全局自适应池化，将时序长度降为 1，避免对输入长度有严格要求
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # 全连接层部分，含有 Dropout 层以减缓过拟合
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 保证输出在 [0,1] 范围内
        )

    def forward(self, x):
        # x 的初始形状: (batch_size, input_length)
        x = x.unsqueeze(1)  # -> (batch_size, 1, input_length)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x).squeeze(-1)  # 输出形状: (batch_size, 128)
        x = self.fc(x)  # 输出形状: (batch_size, 1)
        return x.squeeze(1)  # 调整为 (batch_size,)


##############################################
# 主程序：加载数据、训练与评估
##############################################
if __name__ == "__main__":
    # 固定随机种子，确保结果可复现
    np.random.seed(42)
    torch.manual_seed(42)

    # 数据文件路径（请根据实际情况修改）
    file_type3 = "./output/test3.h5"
    file_type4 = "./output/test4.h5"

    desired_count = 20000
    X, y = load_hdf5_data_for_regression(file_type3, file_type4, desired_count)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Target stats: min = {:.3f}, max = {:.3f}, mean = {:.3f}".format(y.min(), y.max(), y.mean()))

    # 数据预处理：标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分数据集
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # 转换为 PyTorch 张量（回归任务中 y 使用 float 类型）
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 创建 Dataset 和 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 定义模型、损失函数与优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComplexCNNRegression(input_length=X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    patience = 20  # 当连续 10 个 epoch 验证集损失没有改善时触发 Early Stopping
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
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        # Early Stopping 逻辑：
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            counter = 0  # 重置计数器
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs. Best Val Loss: {best_val_loss:.6f}")
                break

    # 训练结束后，加载验证集上表现最好的模型参数
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # 绘制损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    # 在测试集上评估模型
    model.eval()
    y_preds, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            y_preds.extend(outputs.cpu().numpy())
            y_true.extend(y_batch.numpy())
    y_preds = np.array(y_preds)
    y_true = np.array(y_true)
    test_mse = mean_squared_error(y_true, y_preds)
    test_r2 = r2_score(y_true, y_preds)
    print(f"Test MSE: {test_mse:.6f}, R^2: {test_r2:.6f}")

    # 绘制预测值与真实值的散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_preds, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("True Mixing Weight")
    plt.ylabel("Predicted Mixing Weight")
    plt.title("Regression: True vs Predicted")
    plt.show()
