import re
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 如果 model 中有其他模型，这里暂时不使用，直接定义我们需要的分类网络
# from model import Resnet, CNN, simpleCNN

#######################################
# 数据处理与生成函数（与原代码大致一致）
#######################################

def load_separately_with_names(file_type):
    """
    读取 HDF5 文件中除 'q_fixed' 外的数据，提取键名中的数字，
    返回两个列表：names（数字）和数据（对应的数据数组）。
    假设键格式类似 "d20_qIq", "d100_qIq"。
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
    # 按照 names 排序，保证顺序从小到大
    sorted_indices = np.argsort(names)
    sorted_names = [names[i] for i in sorted_indices]
    sorted_data = np.vstack([data_list[i] for i in sorted_indices])
    return sorted_names, sorted_data


def load_all_types(file_type3, file_type4):
    """
    分别加载 Type3 和 Type4 的数据，并返回两个元组：(names, data)。
    """
    names3, data3 = load_separately_with_names(file_type3)
    names4, data4 = load_separately_with_names(file_type4)
    return (names3, data3), (names4, data4)


def augment_type_data(names, X, new_sample_count=2000):
    """
    对单个类型的原始数据 X（形状：(81, feature_dim)）进行扩增，
    每次扩增时随机采样 mean_real 和 sigma_real，并利用对数正态分布对原始数据加权求和，
    返回扩增后的数据（忽略原来的 label，因为分类任务中直接赋予类别标签）。
    """
    new_samples = []
    # 将 names 转换为 numpy 数组作为直径数组
    diameters = np.array(names, dtype=np.float32)
    for i in range(new_sample_count):
        mean_real = np.random.uniform(25, 95)
        sigma_real = np.random.uniform(0.1, 0.4)
        # 计算 mu 和 sigma_log
        mu = np.log(mean_real ** 2 / np.sqrt(mean_real ** 2 + sigma_real ** 2))
        sigma_log = np.sqrt(np.log(1 + (sigma_real ** 2 / mean_real ** 2)))
        # 计算权重（对数正态分布）
        weights = (1.0 / (diameters * sigma_log * np.sqrt(2 * np.pi))) * \
                  np.exp(- (np.log(diameters) - mu) ** 2 / (2 * sigma_log ** 2))
        weights /= np.sum(weights)
        new_curve = np.sum(weights[:, np.newaxis] * X, axis=0)
        new_samples.append(new_curve)
    return np.array(new_samples), None  # 第二项占位，不再使用


def add_noise_to_curve(curve):
    # 防止除零问题
    curve_safe = np.maximum(curve, 1e-12)
    # 从对数均匀分布中采样噪声参数 α
    log_alpha = np.random.uniform(np.log(1e4), np.log(10 ** 7.5))
    alpha = np.exp(log_alpha)
    sigma2 = np.log(1 + alpha / curve_safe)
    epsilon = np.random.randn(curve.shape[0])
    noise_factor = np.exp(np.sqrt(sigma2) * epsilon - sigma2 / 2)
    curve_noisy = curve * noise_factor
    return curve_noisy


def add_noise_to_data(data):
    """
    对数据集中的每个样本添加噪声。
    """
    noisy_data = np.array([add_noise_to_curve(data[i]) for i in range(data.shape[0])])
    return noisy_data


def generate_heter_data(X0, dummy_labels0, X1, dummy_labels1, desired_count=1000):
    """
    生成 heterogeneous 数据：
      - 从 X0 与 X1 中各随机采样一行，用 alpha * row0 + (1 - alpha) * row1 混合生成新样本。
      - 此处忽略混合比例对应的标签信息，因为分类任务中 heterogeneous 统一归为一类。
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


def load_hdf5_data_for_classification(file_type3, file_type4, desired_count=2000, heter_count=1000):
    """
    1. 分别加载 Type3 与 Type4 数据；
    2. 对每个类别利用基于对数正态分布加权扩增方法生成 desired_count 条新样本；
    3. 生成 heterogeneous 数据（desired_count 条）；
    4. 对三类数据均添加噪声并取对数变换；
    5. 最后合并数据，并为每类数据赋予类别标签：
         Type3 -> 0, Type4 -> 1, Heterogeneous -> 2。
    """
    (names3, data3), (names4, data4) = load_all_types(file_type3, file_type4)

    X_type3, _ = augment_type_data(names3, data3, new_sample_count=desired_count)
    X_type4, _ = augment_type_data(names4, data4, new_sample_count=desired_count)
    X_heter, _, _ = generate_heter_data(X_type3, None, X_type4, None, desired_count=heter_count)

    # 为三类数据赋予标签
    y_type3 = np.zeros(X_type3.shape[0], dtype=np.int64)
    y_type4 = np.ones(X_type4.shape[0], dtype=np.int64)
    y_heter = np.full(X_heter.shape[0], 2, dtype=np.int64)

    # 对每一类数据添加噪声，并取 log10 变换（防止负数加噪声则取最大值 1e-12）
    X_type3_noisy = np.log10(np.maximum(add_noise_to_data(X_type3), 1e-12))
    X_type4_noisy = np.log10(np.maximum(add_noise_to_data(X_type4), 1e-12))
    X_heter_noisy = np.log10(np.maximum(add_noise_to_data(X_heter), 1e-12))

    # 合并数据和标签
    X_all = np.concatenate([X_type3_noisy, X_type4_noisy, X_heter_noisy], axis=0)
    y_all = np.concatenate([y_type3, y_type4, y_heter], axis=0)
    return X_all, y_all


#######################################
# 定义一个简单的 1D 卷积分类网络
#######################################

class ClassificationCNN(nn.Module):
    def __init__(self, input_length, num_classes=3):
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
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


#######################################
# 训练与评估分类模型
#######################################

def train_the_model_classification(file_type3, file_type4, desired_count):
    # 加载分类任务数据
    X, y = load_hdf5_data_for_classification(file_type3, file_type4, desired_count, desired_count)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("类别分布:", {cls: int(np.sum(y == cls)) for cls in np.unique(y)})

    # 数据标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分数据集
    from sklearn.model_selection import train_test_split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # 转换为 PyTorch 张量
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassificationCNN(input_length=X.shape[1], num_classes=3).to(device)
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
            outputs = model(X_batch)  # 输出 shape: (batch_size, num_classes)
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs. Best Val Loss: {best_val_loss:.6f}")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # 测试集评估
    model.eval()
    y_preds = []
    y_true = []
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

    # 计算混淆矩阵并打印
    cm = confusion_matrix(y_true, y_preds)
    print("Confusion Matrix:")
    print(cm)

    # 绘制混淆矩阵热力图
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    # plt.colorbar()
    classes = ['Type3', 'Type4', 'Heter']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 在热力图上添加数值标签
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



if __name__ == "__main__":
    np.random.seed(31)
    torch.manual_seed(31)

    # 数据文件路径（请根据实际情况修改）
    file_type3 = "./output/clean3.h5"
    file_type4 = "./output/clean4.h5"

    # desired_count: 每个类型扩增后的样本数
    desired_count = 2000

    train_the_model_classification(file_type3, file_type4, desired_count)
