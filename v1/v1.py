import re
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import copy

from model import DeepResNet1D, simpleCNN

# ------------- GPU device -------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_type1_data(hdf5_file: str = "./dict/type1.h5"):
    """
    从指定的 HDF5 文件中加载 q 值、直径列表和 I(q) 数组。
    """
    with h5py.File(hdf5_file, "r") as f:
        q_values = f["/q"][:]
        q_values = np.squeeze(q_values)
        iq_datasets = [key for key in f.keys() if key != "q"]

        diameters = []
        iq_list = []

        for dataset_name in iq_datasets:
            diameter_match = re.search(r"\d+", dataset_name)
            if diameter_match:
                diameter = int(diameter_match.group())
                diameters.append(diameter)
                iq_data = f[dataset_name][:]
                iq_data = np.squeeze(iq_data)
                iq_list.append(iq_data)
            else:
                print("Error: Could not parse diameter from dataset name.")
                continue

        diameters = np.array(diameters)
        Iq_array = np.array(iq_list)
        print("Iq_array shape:", Iq_array.shape)

    return q_values, diameters, Iq_array


def generate_synthesized_curve(
        mean: float,
        sigma: float,
        diameters: np.ndarray,
        iq_array: np.ndarray
) -> np.ndarray:
    # 将对数正态参数转换为实际空间的 mu 和 sigma
    mu = np.log(mean ** 2 / np.sqrt(mean ** 2 + sigma ** 2))
    sigma_log = np.sqrt(np.log(1 + (sigma ** 2 / mean ** 2)))

    # 计算对数正态权重
    weights = (1 / (diameters * sigma_log * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(diameters) - mu) ** 2) / (2 * sigma_log ** 2)
    )
    weights /= np.sum(weights)
    iq_weighted_sum = np.dot(weights, iq_array)
    return iq_weighted_sum


def create_dataset(num_samples: int = 100000) -> None:
    """
    生成合成的数据集并保存到 synthesized_dataset.h5 文件中。
    """
    # 随机生成 mean 和 sigma
    means = np.random.uniform(low=30, high=90, size=num_samples)
    sigmas = np.random.uniform(low=1, high=5, size=num_samples)

    # 加载已有数据文件并得到 q_values, diameters, Iq_array
    q_values, diameters, iq_array = load_type1_data()

    # 用于存储最终的 (mean, sigma) 以及合成的曲线
    mean_sigma_array = np.zeros((num_samples, 2))
    synthesized_curves = np.zeros((num_samples, len(q_values)))

    for i in range(num_samples):
        mean_val = means[i]
        sigma_val = sigmas[i]
        synthesized_curve = generate_synthesized_curve(mean_val, sigma_val, diameters, iq_array)

        synthesized_curves[i] = synthesized_curve
        mean_sigma_array[i, 0] = mean_val
        mean_sigma_array[i, 1] = sigma_val

    # 将结果写入 HDF5 文件
    with h5py.File("synthesized_dataset.h5", "w") as f_out:
        f_out.create_dataset("q", data=q_values)
        f_out.create_dataset("curves", data=synthesized_curves)
        f_out.create_dataset("mean_sigma", data=mean_sigma_array)


class SynthesizedCurveDataset(Dataset):
    """
    自定义数据集类，用于从生成的 HDF5 文件中加载合成后的曲线及对应的 (mean, sigma)。
    """

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
        curve = self.curves[idx]
        mean_sigma = self.mean_sigma[idx]

        curve = torch.tensor(curve, dtype=torch.float32)
        mean_sigma = torch.tensor(mean_sigma, dtype=torch.float32)
        return curve, mean_sigma

    def get_q_values(self) -> np.ndarray:
        return self.q_values


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


def train_and_evaluate_kfold(
    dataset_path: str = "synthesized_dataset.h5",
    target_index: int = 0,
    target_name: str = "Mean",
    num_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 0.001,
    k_splits: int = 5,
    early_stopping_patience: int = 5
):
    """
    使用 K-fold 交叉验证对指定目标 (Mean 或 Sigma) 进行训练和评估。
    在每个 epoch 都会打印训练集损失和验证集损失，并支持 early stopping。
    训练完成后，会在验证集中做预测，并汇总每个 fold 的预测值和真实值用于可视化。
    """
    # 1) 加载数据集（完整数据）
    dataset = SynthesizedCurveDataset(dataset_path)
    input_length = dataset.curves.shape[1]

    # 2) 定义 KFold
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=42)

    # 用于统计 K 次验证的平均损失
    fold_losses = []
    # 记录所有 fold 的预测和真实值，方便最后统一可视化
    all_true_vals = []
    all_pred_vals = []

    # 3) 逐个 Fold 进行训练和验证
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n----- Fold {fold+1}/{k_splits} -----")

        # 3.1) 构建当折的训练集、验证集 Subset
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # 3.2) 定义模型、损失、优化器
        model = DeepResNet1D(input_length).to(device)  # 你也可以换成 DeepResNet1D(...)等
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)

        # 3.3) Early Stopping 的辅助变量
        best_val_loss = float('inf')
        best_model_weights = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0

        # 3.4) 开始训练
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, target_index)
            val_loss = evaluate_one_epoch(model, val_loader, criterion, device, target_index)

            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train {target_name} Loss: {train_loss:.4f} | "
                  f"Val {target_name} Loss: {val_loss:.4f}")

            # Early Stopping 检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1} (no improvement in {early_stopping_patience} epochs).")
                break

        # 3.5) 训练结束后（或 early stopping 提前终止后），恢复 best 权重
        model.load_state_dict(best_model_weights)

        # 3.6) 当前 Fold 完成后，在验证集上的最终 Loss
        final_val_loss = evaluate_one_epoch(model, val_loader, criterion, device, target_index)
        fold_losses.append(final_val_loss)
        print(f"Fold {fold+1} Final Val {target_name} Loss: {final_val_loss:.4f}")

        # ========== (新) 在该fold上获取预测结果 ==========
        fold_true, fold_pred = get_predictions(model, val_loader, device, target_index)
        all_true_vals.append(fold_true)
        all_pred_vals.append(fold_pred)

    # 4) 所有 fold 完成后的平均验证损失
    avg_loss = np.mean(fold_losses)
    print(f"\n===== K-Fold Cross Validation Result ({target_name}) =====")
    print(f"Average Val Loss across {k_splits} folds: {avg_loss:.4f}")

    # ========== (新) 统一绘制所有 fold 的预测 vs. 真实散点图 ==========
    # 将每个 fold 的预测/真实值拼接在一起
    all_true_vals = np.concatenate(all_true_vals, axis=0)
    all_pred_vals = np.concatenate(all_pred_vals, axis=0)

    plt.figure(figsize=(8, 6))
    plt.scatter(all_true_vals, all_pred_vals, alpha=0.5)
    # 画对角线
    min_val, max_val = all_true_vals.min(), all_true_vals.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel(f"True {target_name} Values")
    plt.ylabel(f"Predicted {target_name} Values")
    plt.title(f"Predicted vs. True {target_name} Values (All Folds)")
    plt.show()


if __name__ == "__main__":
    # 如果还没生成 h5 数据，可先执行 create_dataset():
    # create_dataset()

    # # 使用 K-fold + Early Stopping 训练并预测 Mean
    # train_and_evaluate_kfold(
    #     dataset_path="synthesized_dataset.h5",
    #     target_index=0,        # 0 表示 mean, 1 表示 sigma
    #     target_name="Mean",    # 用于打印时的标签
    #     num_epochs=10,         # 你可以自行设置一个足够大的 epoch, 让 early stopping 去判断
    #     batch_size=512,
    #     lr=0.001,
    #     k_splits=4,
    #     early_stopping_patience=4
    # )

    # 如果需要对 Sigma 进行预测，也可以调用一次
    train_and_evaluate_kfold(
        dataset_path="synthesized_dataset.h5",
        target_index=1,
        target_name="Sigma",
        num_epochs=10,
        batch_size=512,
        lr=0.001,
        k_splits=4,
        early_stopping_patience=4
    )
