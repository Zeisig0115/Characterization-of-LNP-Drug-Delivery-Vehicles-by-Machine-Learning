import re
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from model import Resnet, CNN, simpleCNN


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
    # 按照 names 排序，保证顺序从小到大，与直径对应（如20, 21, ..., 100）
    sorted_indices = np.argsort(names)
    sorted_names = [names[i] for i in sorted_indices]
    sorted_data = np.vstack([data_list[i] for i in sorted_indices])
    return sorted_names, sorted_data


def load_all_types(file_type3, file_type4):
    """
    分别加载 Type3 和 Type4 的数据，并返回两个元组：(names, data)。
    保证数据与名称一一对应，且顺序为从小到大。
    """
    names3, data3 = load_separately_with_names(file_type3)
    names4, data4 = load_separately_with_names(file_type4)
    return (names3, data3), (names4, data4)


def augment_type_data(names, X, new_sample_count=2000):
    """
    对单个类型的原始数据 X（形状：(81, feature_dim)）进行扩增，
    每次扩增时随机采样 mean_real 和 sigma_real，并利用对数正态分布对原始 81 条数据加权求和，
    返回扩增后的数据及其 label（随机生成的 mean_real）。
    """
    new_samples = []
    labels = []
    # 将 names 转换为 numpy 数组作为直径数组
    diameters = np.array(names, dtype=np.float32)
    for i in range(new_sample_count):
        mean_real = np.random.uniform(25, 95)
        sigma_real = np.random.uniform(0.1, 0.4)
        # 计算 mu 和 sigma_log
        mu = np.log(mean_real ** 2 / np.sqrt(mean_real ** 2 + sigma_real ** 2))
        sigma_log = np.sqrt(np.log(1 + (sigma_real ** 2 / mean_real ** 2)))
        # 计算权重（对数正态分布），在 diameters 数组上计算 pdf
        weights = (1.0 / (diameters * sigma_log * np.sqrt(2 * np.pi))) * \
                  np.exp(- (np.log(diameters) - mu) ** 2 / (2 * sigma_log ** 2))
        weights /= np.sum(weights)
        new_curve = np.sum(weights[:, np.newaxis] * X, axis=0)
        new_samples.append(new_curve)
        labels.append(mu)
    return np.array(new_samples), np.array(labels)


def add_noise_to_curve(curve):
    # 防止除零问题
    curve_safe = np.maximum(curve, 1e-12)
    # 从对数均匀分布中采样噪声参数 α，范围 [1e4, 10^(7.5)]
    log_alpha = np.random.uniform(np.log(1e4), np.log(10 ** 7.5))
    alpha = np.exp(log_alpha)
    # 计算 sigma²(q) = ln(1 + α / I(q))
    sigma2 = np.log(1 + alpha / curve_safe)
    # 为每个 q 值生成标准正态分布随机数
    epsilon = np.random.randn(curve.shape[0])
    # 计算噪声因子 N(q)
    noise_factor = np.exp(np.sqrt(sigma2) * epsilon - sigma2 / 2)
    # 得到添加噪声后的曲线
    curve_noisy = curve * noise_factor
    return curve_noisy


def add_noise_to_data(data):
    """
    对数据集中的每个样本（每行）添加噪声。
    data: numpy 数组，形状 (样本数, 特征维度)
    """
    noisy_data = np.array([add_noise_to_curve(data[i]) for i in range(data.shape[0])])
    return noisy_data


def generate_heter_data(X0, labels0, X1, labels1, desired_count=1000):
    """
    生成 heterogeneous 数据：
      - 从 X0 与 X1 中各随机采样一行，用 alpha * row0 + (1 - alpha) * row1 混合生成新样本，
      - 同时返回混合比例 alpha 以及混合后的均值 mu，
        其中 mu = alpha * labels0 + (1 - alpha) * labels1.
    """
    n0, n1 = X0.shape[0], X1.shape[0]
    X_heter_list = []
    alpha_list = []
    mu_heter_list = []
    for _ in range(desired_count):
        idx0 = np.random.randint(n0)
        idx1 = np.random.randint(n1)
        alpha = np.random.rand()
        mixed_curve = alpha * X0[idx0] + (1 - alpha) * X1[idx1]
        X_heter_list.append(mixed_curve)
        alpha_list.append(alpha)
        mu_val = alpha * labels0[idx0] + (1 - alpha) * labels1[idx1]
        mu_heter_list.append(mu_val)
    return np.array(X_heter_list), np.array(alpha_list, dtype=np.float32), np.array(mu_heter_list, dtype=np.float32)


def load_hdf5_data_for_regression(file_type3, file_type4, desired_count=2000, heter_count=1000, target="alpha"):
    """
    1. 分别加载 Type3 与 Type4 数据，保留名称信息（如 [20,21,...,100]）和数据；
    2. 对每个类别利用基于对数正态分布加权扩增方法生成 desired_count 条新样本，
       每条新样本的 label 为随机生成的 mean_real；
    3. 利用扩增后的数据生成 heterogeneous 数据：从扩增后的 Type3 与 Type4 数据中随机采样生成混合样本，
       同时生成混合比例 alpha 和混合后的均值 mu；
    4. 最后对生成的 heterogeneous 数据再加一次噪声。

    参数 target 用于选择返回哪个目标：
         "alpha" - 返回混合比例作为目标；
         "mu"    - 返回混合后的均值作为目标。
    """
    (names3, data3), (names4, data4) = load_all_types(file_type3, file_type4)

    X0, labels0 = augment_type_data(names3, data3, new_sample_count=desired_count)
    X1, labels1 = augment_type_data(names4, data4, new_sample_count=desired_count)

    # 生成 heterogeneous 数据，同时生成两个标签
    X_heter, alphas, mu_heter = generate_heter_data(X0, labels0, X1, labels1, desired_count=heter_count)

    # 对 heterogeneous 数据再加一次噪声
    X_heter_noisy = add_noise_to_data(X_heter)
    # X_loglog = np.log10(np.maximum(X_heter_noisy, 1e-12))

    # 根据 target 选择返回的标签
    if target == "alpha":
        labels = alphas
    elif target == "mu":
        labels = mu_heter
    else:
        raise ValueError("target 参数必须为 'alpha' 或 'mu'")

    return X_heter_noisy, labels

def train_the_model(file_type3, file_type4, desired_count,heter_count):
    # target 参数可以选择 "alpha" 或 "mu"
    X, y = load_hdf5_data_for_regression(file_type3, file_type4, desired_count, heter_count, target="alpha")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Target stats: min = {:.3f}, max = {:.3f}, mean = {:.3f}".format(y.min(), y.max(), y.mean()))

    # 数据预处理：标准化
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
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

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
            outputs = model(X_batch)  # 输出 shape 为 [batch_size, 1]
            loss = criterion(outputs.squeeze(), y_batch)
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
                loss = criterion(outputs.squeeze(), y_batch)
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

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 Loss")
    plt.legend()
    plt.show()

    model.eval()
    y_preds, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            y_preds.extend(outputs.squeeze().cpu().numpy())
            y_true.extend(y_batch.numpy())
    y_preds = np.array(y_preds)
    y_true = np.array(y_true)
    test_mse = mean_squared_error(y_true, y_preds)
    test_r2 = r2_score(y_true, y_preds)
    print(f"Test MSE: {test_mse:.6f}, R^2: {test_r2:.6f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_preds, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Target")
    plt.ylabel("Predicted Target")
    plt.title("Regression: True vs Predicted")
    plt.show()


##############################################
# 主程序：加载数据、训练与评估
##############################################

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # 数据文件路径（请根据实际情况修改）
    file_type3 = "./output/clean3.h5"
    file_type4 = "./output/clean4.h5"

    # desired_count: 每个类型扩增后的样本数，heter_count: heterogeneous 数据样本数
    desired_count = 3000
    heter_count = 3000

    # (names3, data3), (names4, data4) = load_all_types(file_type3, file_type4)
    #
    # X0, labels0 = augment_type_data(names3, data3, new_sample_count=desired_count)
    # X1, labels1 = augment_type_data(names4, data4, new_sample_count=desired_count)
    #
    # # 生成 heterogeneous 数据，同时生成两个标签
    # X_heter, alphas, mu_heter = generate_heter_data(X0, labels0, X1, labels1, desired_count=heter_count)
    #
    # # 对 heterogeneous 数据再加一次噪声
    # X_heter_noisy = add_noise_to_data(X_heter)
    # X_loglog = np.log10(np.maximum(X_heter_noisy, 1e-12))

    train_the_model(file_type3, file_type4, desired_count, heter_count)


