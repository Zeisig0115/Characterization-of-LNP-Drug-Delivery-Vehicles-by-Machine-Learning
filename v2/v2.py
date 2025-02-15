import re
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import copy
import matplotlib.pyplot as plt
from model import DeepResNet1D, simpleCNN

# ------------- GPU device -------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_type1_data(hdf5_file: str = "./output/interpolated_type1.h5"):
    """从 HDF5 文件加载 q 值、diameters 和 I(q) 数据"""
    with h5py.File(hdf5_file, "r") as f:
        q_values = f["q_fixed"][:]  # 读取 q 值
        iq_datasets = [key for key in f.keys() if key.startswith("d") and key.endswith("_qIq")]

        diameters = []
        iq_list = []

        for dataset_name in iq_datasets:
            diameter_match = re.search(r"\d+", dataset_name)
            if diameter_match:
                diameter = int(diameter_match.group())
                diameters.append(diameter)
                iq_data = f[dataset_name][:]  # 获取 SAXS 信号
                iq_list.append(iq_data)
            else:
                print(f"Error: Could not parse diameter from dataset name: {dataset_name}")

        diameters = np.array(diameters)
        Iq_array = np.array(iq_list)

        print(f"Loaded {len(diameters)} diameter SAXS curves. Shape of Iq_array: {Iq_array.shape}")

    return q_values, diameters, Iq_array


def generate_synthesized_curve(mean: float, sigma: float, diameters: np.ndarray, iq_array: np.ndarray) -> np.ndarray:
    """
    基于对数正态分布计算合成的 SAXS 数据

    参数:
    - mean: 颗粒尺寸分布的均值 (真实空间)
    - sigma: 颗粒尺寸分布的标准差 (真实空间)
    - diameters: 训练数据集中可用的粒径值
    - iq_array: 每个粒径对应的 SAXS 信号

    返回:
    - 计算出的合成 SAXS 信号
    """
    # 转换为对数正态分布的 mu 和 sigma_log
    mu = np.log(mean ** 2 / np.sqrt(mean ** 2 + sigma ** 2))
    sigma_log = np.sqrt(np.log(1 + (sigma ** 2 / mean ** 2)))

    # 计算权重（对数正态分布）
    weights = (1.0 / (diameters * sigma_log * np.sqrt(2 * np.pi))) * \
              np.exp(- (np.log(diameters) - mu) ** 2 / (2 * sigma_log ** 2))

    # 归一化权重，确保总和为 1
    weights /= np.sum(weights)

    # 计算加权 SAXS 曲线
    iq_weighted_sum = np.dot(weights, iq_array)

    return iq_weighted_sum


def create_dataset(num_samples: int = 100000, hdf5_file: str = "./output/interpolated_type1.h5"):
    """基于 Lognormal 采样 (μ, σ) 生成合成 SAXS 数据"""
    # 读取已有 SAXS 数据
    q_values, diameters, iq_array = load_type1_data(hdf5_file)

    means = np.random.uniform(low=30, high=90, size=num_samples)
    sigmas = np.random.uniform(low=1, high=5, size=num_samples)

    mean_sigma_array = np.zeros((num_samples, 2))
    synthesized_curves = np.zeros((num_samples, len(q_values)))

    for i in range(num_samples):
        mean_val = means[i]
        sigma_val = sigmas[i]
        synthesized_curve = generate_synthesized_curve(mean_val, sigma_val, diameters, iq_array)

        synthesized_curves[i] = synthesized_curve
        mean_sigma_array[i, 0] = mean_val
        mean_sigma_array[i, 1] = sigma_val

    with h5py.File("synthesized_dataset.h5", "w") as f_out:
        f_out.create_dataset("q", data=q_values)
        f_out.create_dataset("curves", data=synthesized_curves)
        f_out.create_dataset("mean_sigma", data=mean_sigma_array)

    print(f"Dataset saved as synthesized_dataset.h5 with {num_samples} samples")



class SynthesizedCurveDataset(Dataset):
    """用于加载合成 SAXS 数据的 PyTorch 数据集"""
    def __init__(self, hdf5_file: str):
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, "r") as f:
            self.curves = f["curves"][:]
            self.mean_sigma = f["mean_sigma"][:]
            self.q_values = f["q"][:]
        self.num_samples = self.mean_sigma.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        curve = torch.tensor(self.curves[idx], dtype=torch.float32)
        mean_sigma = torch.tensor(self.mean_sigma[idx], dtype=torch.float32)
        return curve, mean_sigma


def train_one_epoch(model, dataloader, criterion, optimizer, device, target_index):
    """
    单个 epoch 的训练过程，返回本 epoch 的平均训练损失
    """
    model.train()
    running_loss = 0.0
    n_total = 0

    for curves, mean_sigma in dataloader:
        curves = curves.to(device)
        mean_sigma = mean_sigma.to(device)

        optimizer.zero_grad()
        outputs = model(curves)

        # 只取需要的 target: 0 表 mean, 1 表 sigma
        targets = mean_sigma[:, target_index].unsqueeze(1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size_current = curves.size(0)
        running_loss += loss.item() * batch_size_current
        n_total += batch_size_current

    return running_loss / n_total


def evaluate_one_epoch(model, dataloader, criterion, device, target_index):
    """
    在验证集/测试集上评估一个 epoch，返回平均损失
    """
    model.eval()
    running_loss = 0.0
    n_total = 0

    with torch.no_grad():
        for curves, mean_sigma in dataloader:
            curves = curves.to(device)
            mean_sigma = mean_sigma.to(device)

            outputs = model(curves)
            targets = mean_sigma[:, target_index].unsqueeze(1)

            loss = criterion(outputs, targets)

            batch_size_current = curves.size(0)
            running_loss += loss.item() * batch_size_current
            n_total += batch_size_current

    return running_loss / n_total


def get_predictions(model, dataloader, device, target_index):
    """
    使用当前模型在给定 dataloader 上做预测，返回两个 numpy 数组:
        - true_vals: 真实的标签 (mean 或 sigma)
        - pred_vals: 模型预测值
    """
    model.eval()
    all_true_vals = []
    all_pred_vals = []

    with torch.no_grad():
        for curves, mean_sigma in dataloader:
            curves = curves.to(device)
            mean_sigma = mean_sigma.to(device)

            # 模型输出 (batch_size, 1)
            outputs = model(curves).squeeze(1)  # shape: (batch_size,)
            targets = mean_sigma[:, target_index]  # shape: (batch_size,)

            all_true_vals.append(targets.cpu().numpy())
            all_pred_vals.append(outputs.cpu().numpy())

    # 拼接成单个数组
    all_true_vals = np.concatenate(all_true_vals, axis=0)
    all_pred_vals = np.concatenate(all_pred_vals, axis=0)
    return all_true_vals, all_pred_vals


def train_and_evaluate_kfold(dataset_path: str, target_index: int, target_name: str, num_epochs: int = 40):
    """K-fold 交叉验证训练 ML 模型，并绘制回归散点图"""
    dataset = SynthesizedCurveDataset(dataset_path)
    input_length = dataset.curves.shape[1]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_losses = []
    all_true_vals = []
    all_pred_vals = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n----- Fold {fold+1}/5 -----")

        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=512, shuffle=True)
        val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=512, shuffle=False)

        model = DeepResNet1D(input_length).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.1)

        best_val_loss = float('inf')
        best_model_weights = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, target_index)
            val_loss = evaluate_one_epoch(model, val_loader, criterion, device, target_index)

            print(f"Epoch [{epoch+1}/{num_epochs}] - Train {target_name} Loss: {train_loss:.4f} | Val {target_name} Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= 10:
                print(f"Early stopping at epoch {epoch+1}.")
                break

        model.load_state_dict(best_model_weights)
        final_val_loss = evaluate_one_epoch(model, val_loader, criterion, device, target_index)
        fold_losses.append(final_val_loss)

        # 获取预测值
        fold_true, fold_pred = get_predictions(model, val_loader, device, target_index)
        all_true_vals.append(fold_true)
        all_pred_vals.append(fold_pred)

    print(f"\nAverage Validation Loss: {np.mean(fold_losses):.4f}")

    # === 统一绘制所有 fold 的回归散点图 ===
    all_true_vals = np.concatenate(all_true_vals, axis=0)
    all_pred_vals = np.concatenate(all_pred_vals, axis=0)

    plt.figure(figsize=(8, 6))
    plt.scatter(all_true_vals, all_pred_vals, alpha=0.5, edgecolors='k')

    # 绘制 y=x 理想回归线
    min_val, max_val = all_true_vals.min(), all_true_vals.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    plt.xlabel(f"True {target_name} Values")
    plt.ylabel(f"Predicted {target_name} Values")
    plt.title(f"Regression Plot: {target_name}")
    plt.show()



if __name__ == "__main__":
    # create_dataset()
    # train_and_evaluate_kfold("synthesized_dataset.h5", target_index=0, target_name="Mean")
    train_and_evaluate_kfold("synthesized_dataset.h5", target_index=1, target_name="Sigma")
