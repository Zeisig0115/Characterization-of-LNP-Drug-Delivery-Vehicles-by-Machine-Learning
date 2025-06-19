import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

##############################################
# 数据加载及扩增相关函数
##############################################

def load_separately(file_type3, file_type4):
    """
    分别读取 Type3、Type4 所有 key (排除 'q_fixed' )，将它们各自的所有数据垂直堆叠起来。
    返回:
       X_type3: shape=(N3, feature_dim)
       X_type4: shape=(N4, feature_dim)
    """
    with h5py.File(file_type3, 'r') as f3, h5py.File(file_type4, 'r') as f4:
        # 收集 f3 中除 'q_fixed' 以外的所有 key
        dataset_keys = [key for key in f3.keys() if key != 'q_fixed']

        X_type3_list = []
        X_type4_list = []
        for key in dataset_keys:
            if key in f4:
                data3 = f3[key][:]
                data4 = f4[key][:]
                X_type3_list.append(data3)
                X_type4_list.append(data4)
            else:
                # 若 f4 中没有对应的 key，可以选择跳过
                pass

        X_type3 = np.vstack(X_type3_list) if X_type3_list else np.array([])
        X_type4 = np.vstack(X_type4_list) if X_type4_list else np.array([])
        return X_type3, X_type4


def sample_or_augment_class(X, desired_count=1000):
    """
    在同一类内部做数据扩增或下采样，使最终数量 = desired_count。

    - 若 len(X) >= desired_count: 随机下采样到 desired_count 行
    - 若 len(X) < desired_count: 通过在 X 内部随机挑选 (row_i, row_j) 并做
      alpha * row_i + (1-alpha) * row_j 的方式生成新的样本，直到总数达标。
    """
    n = X.shape[0]
    if n == 0:
        raise ValueError("X has no data. Can't augment or sample.")

    if n == desired_count:
        return X
    elif n > desired_count:
        # 下采样
        indices = np.random.choice(n, size=desired_count, replace=False)
        return X[indices]
    else:
        # 扩增：生成 (desired_count - n) 个新样本
        num_new = desired_count - n
        new_samples = []
        for _ in range(num_new):
            idx1 = np.random.randint(n)
            idx2 = np.random.randint(n)
            alpha = np.random.rand()
            row = alpha * X[idx1] + (1 - alpha) * X[idx2]
            new_samples.append(row)
        new_samples = np.array(new_samples, dtype=X.dtype)
        return np.vstack([X, new_samples])


def generate_heter_data(X0, X1, desired_count=1000):
    """
    生成 heterogeneous 数据: 随机从 X0, X1 中各取一行，
    用 alpha * row0 + (1-alpha) * row1 线性组合得到新样本。
    返回 shape=(desired_count, feature_dim)。
    """
    n0 = X0.shape[0]
    n1 = X1.shape[0]
    X_heter_list = []
    for _ in range(desired_count):
        idx0 = np.random.randint(n0)
        idx1 = np.random.randint(n1)
        alpha = np.random.rand()
        row_mix = alpha * X0[idx0] + (1 - alpha) * X1[idx1]
        X_heter_list.append(row_mix)
    return np.array(X_heter_list, dtype=X0.dtype)


def add_noise_to_curve(curve):
    """
    对单个 I(q) 曲线添加噪声，并归一化散射强度。
    """
    # 防止 curve 中存在极小值导致除 0 问题
    curve_safe = np.maximum(curve, 1e-12)

    # 采样满足对数均匀分布的 α，范围 [10^5, 10^(8.5)]
    log_alpha = np.random.uniform(np.log(1e2), np.log(10 ** 5.5))
    alpha = np.exp(log_alpha)

    # 对每个 q 计算 sigma^2: sigma2 = ln(1 + alpha / I(q))
    sigma2 = np.log(1 + alpha / curve_safe)

    # 为每个 q 生成正态分布变量 epsilon
    epsilon = np.random.randn(curve.shape[0])

    # 计算噪声系数：N(q) = exp(sqrt(sigma2)*epsilon - sigma2/2)
    noise_factor = np.exp(np.sqrt(sigma2) * epsilon - sigma2 / 2)

    # 添加噪声
    curve_noisy = curve * noise_factor

    # 归一化散射强度（这里采用梯形积分，假设 q 轴步长均匀）
    total_intensity = np.trapz(curve_noisy)
    return curve_noisy / total_intensity


def add_noise_to_data(data):
    """
    对数据集中的每个样本（每行）添加噪声。
    data: numpy 数组，形状 (样本数, 特征维度)
    """
    noisy_data = np.array([add_noise_to_curve(data[i]) for i in range(data.shape[0])])
    return noisy_data


def load_hdf5_data_with_expansion(file_type3, file_type4, desired_count=1000):
    """
    - 分别加载 Type3 (class=0) 与 Type4 (class=1) 的所有数据
    - 在同类内部做扩增或下采样，使各自达到 desired_count
    - 跨类（Type3 vs Type4）随机混合生成 desired_count 条 heterogeneous 数据 (class=2)
    - 合并返回 X_all, y_all

    最终:
      X_all.shape = (3 * desired_count, feature_dim)
      y_all.shape = (3 * desired_count,)
      其中 y_all 的取值为: 0 (Type3), 1 (Type4), 2 (Heterogeneous)
    """
    # 1) 分别加载 Type3 和 Type4 的数据
    X_type3_full, X_type4_full = load_separately(file_type3, file_type4)
    if X_type3_full.size == 0 or X_type4_full.size == 0:
        raise ValueError("No valid Type3 or Type4 data found.")

    # 2) 同类内部下采样或扩增到 desired_count
    X0 = sample_or_augment_class(X_type3_full, desired_count=desired_count)  # label=0
    X1 = sample_or_augment_class(X_type4_full, desired_count=desired_count)  # label=1

    # 3) 生成 heterogeneous 数据 (label=2)
    X2 = generate_heter_data(X0, X1, desired_count=desired_count)

    X0 = add_noise_to_data(X0)
    X1 = add_noise_to_data(X1)
    X2 = add_noise_to_data(X2)

    # 4) 合并数据及标签
    y0 = np.zeros(desired_count, dtype=np.int32)
    y1 = np.ones(desired_count, dtype=np.int32)
    y2 = np.full(desired_count, 2, dtype=np.int32)

    X_all = np.vstack([X0, X1, X2])
    y_all = np.concatenate([y0, y1, y2])

    return X_all, y_all

##############################################
# 定义 1D CNN 模型
##############################################
class SimpleCNN(nn.Module):
    def __init__(self, input_length, num_classes=3):
        """
        input_length: 原始特征维度的长度
        """
        super(SimpleCNN, self).__init__()
        # 输入数据经过 unsqueeze 后形状为 (batch_size, 1, input_length)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 经过两层卷积和池化后，序列长度变为 input_length/4（假设 input_length 可以被 4 整除）
        fc_input_dim = 32 * (input_length // 4)
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x 的初始形状: (batch_size, input_length)
        # 增加 channel 维度: (batch_size, 1, input_length)
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # 展平
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

##############################################
# 主程序：加载数据、预处理、训练、评估
##############################################
if __name__ == "__main__":
    # 固定随机种子，确保结果可复现
    np.random.seed(42)
    torch.manual_seed(42)

    # 文件路径（请根据实际情况修改路径）
    file_type3 = "./output/clean3.h5"
    file_type4 = "./output/clean4.h5"

    # 加载数据，每个类别 desired_count 条（总共 3*desired_count 条数据）
    desired_count = 2000
    X, y = load_hdf5_data_with_expansion(
        file_type3=file_type3,
        file_type4=file_type4,
        desired_count=desired_count
    )

    print("Final X shape:", X.shape)  # 例如 (3000, feature_dim)
    print("Final y shape:", y.shape)  # 例如 (3000,)
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))
    # 期望输出: {0:1000, 1:1000, 2:1000}

    # -------------------------------
    # 数据预处理：标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分数据集：先划分出测试集（20%），再从训练集中划分出验证集（例如训练集的 20%）
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )

    # 转换为 PyTorch 张量
    # 注意：CrossEntropyLoss 要求标签为 Long 类型，并且标签 shape 为 (N,)（而非 (N,1)）
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 创建 Dataset 和 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # -------------------------------
    # 定义模型、损失函数与优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用 CNN 模型，注意这里的 input_length 等于特征维度
    model = SimpleCNN(input_length=X.shape[1], num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()  # 适用于三分类问题
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # 记录每个 epoch 的训练损失和验证损失
    train_losses = []
    val_losses = []

    epochs = 80
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)  # 输出 shape: (batch_size, num_classes)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_epoch_loss = total_train_loss / len(train_loader)
        train_losses.append(train_epoch_loss)

        # 在验证集上评估
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()

        val_epoch_loss = total_val_loss / len(val_loader)
        val_losses.append(val_epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # -------------------------------
    # 在测试集上评估模型
    model.eval()
    y_preds = []
    y_true = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)  # shape: (batch_size, num_classes)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # 取预测的类别
            y_preds.extend(preds)
            y_true.extend(y_batch.numpy())

    accuracy = np.mean(np.array(y_preds) == np.array(y_true))
    print(f"Test Accuracy: {accuracy:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show(block=True)
