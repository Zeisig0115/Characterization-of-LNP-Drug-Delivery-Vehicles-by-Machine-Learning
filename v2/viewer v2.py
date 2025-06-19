import h5py
import numpy as np
import matplotlib.pyplot as plt
import re
import random

# 固定随机种子
seed_value = 1
random.seed(seed_value)
np.random.seed(seed_value)

def add_noise_to_curve(curve):
    # 防止除零问题
    curve_safe = np.maximum(curve, 1e-12)
    log_alpha = np.random.uniform(np.log(1e5), np.log(10 ** 8.5))
    alpha = np.exp(log_alpha)
    # 计算 sigma²(q) = ln(1 + α / I(q))
    sigma2 = np.log(1 + alpha / curve_safe)
    # 为每个 q 值生成标准正态分布随机数
    epsilon = np.random.randn(curve.shape[0])
    # 计算噪声因子 N(q)
    noise_factor = np.exp(np.sqrt(sigma2) * epsilon - sigma2 / 2)
    # 得到添加噪声后的曲线
    curve_noisy = curve * noise_factor
    # normalization = np.trapz(curve_noisy)
    return curve_noisy / 1

def add_noise_to_data(data):
    """
    对数据集中的每个样本（每行）添加噪声。
    data: numpy 数组，形状 (样本数, 特征维度)
    """
    noisy_data = np.array([add_noise_to_curve(data[i]) for i in range(data.shape[0])])
    return noisy_data

# 读取 HDF5 文件
hdf5_file = "../output/clean3.h5"  # 请修改为实际路径
with h5py.File(hdf5_file, 'r') as f:
    q = f['q_fixed'][:]  # 读取标准 q 轴

    # 获取所有 Iq 数据集名称（排除 q_fixed）
    iq_datasets = [key for key in f.keys() if key != 'q_fixed']
    print("Available Iq datasets:", iq_datasets)

    # 提取直径信息
    diameters = []
    for dataset_name in iq_datasets:
        diameter_match = re.search(r'\d+', dataset_name)
        if diameter_match:
            diameter = int(diameter_match.group())
            diameters.append(diameter)
        else:
            print(f"Error: Cannot parse diameter from dataset name: {dataset_name}")
            continue
    diameters = np.array(diameters)

    # 创建一个大图，并添加两个子图，一个用于权重分布，一个用于合成曲线
    fig, (ax_weights, ax_curves) = plt.subplots(2, 1, figsize=(10, 12))

    num_new_datasets = 2  # 例如生成 3 个新数据集
    for i in range(num_new_datasets):
        # 每次生成新的 mean 和 σ
        mean_real = random.uniform(25, 95)  # mean 在 25 到 95 之间随机
        sigma_real = random.uniform(3, 6)   # σ 在 1 到 5 之间随机
        print(f"Dataset {i+1}: mean_real = {mean_real:.3f}, sigma_real = {sigma_real:.3f}")

        # 根据新的参数计算对数正态分布的 mu 和 sigma_log
        mu = np.log(mean_real ** 2 / np.sqrt(mean_real ** 2 + sigma_real ** 2))
        sigma_log = np.sqrt(np.log(1 + (sigma_real ** 2 / mean_real ** 2)))

        # 计算权重（对数正态分布）
        weights = (1.0 / (diameters * sigma_log * np.sqrt(2 * np.pi))) * \
                  np.exp(- (np.log(diameters) - mu) ** 2 / (2 * sigma_log ** 2))
        weights /= np.sum(weights)

        # 在权重图中绘制当前权重曲线，并标注最大值点
        ax_weights.plot(weights, label=f'Type 3 Sample {i+1}: ({int(mean_real)},{int(sigma_real)})', lw=3)
        max_idx = np.argmax(weights)
        max_val = weights[max_idx]
        ax_weights.plot(max_idx, max_val, 'o')
        ax_weights.annotate(f'Max: {max_val:.3e}',
                            xy=(max_idx, max_val),
                            xytext=(max_idx + 1, max_val * 0.9),
                            arrowprops=dict(arrowstyle='->', color='red'),
                            fontsize=12)

        # 计算加权合成 SAXS 曲线
        Iq_weighted_sum = np.zeros_like(f[iq_datasets[0]][:])
        for j, dataset_name in enumerate(iq_datasets):
            Iq = f[dataset_name][:]
            Iq_weighted_sum += weights[j] * Iq

        # 对合成曲线添加噪声（转换成一维数组进行处理）
        flat_Iq = Iq_weighted_sum.flatten()
        noisy_Iq = add_noise_to_curve(flat_Iq)

        ax_curves.scatter(q.flatten(), noisy_Iq, s=8, label=f'Type 3 Sample {i+1} Noisy')

        # 标注原始曲线的最大值点
        max_idx_curve = np.argmax(flat_Iq)
        max_Iq = flat_Iq[max_idx_curve]
        max_q = q.flatten()[max_idx_curve]
        ax_curves.plot(max_q, max_Iq, 'o')
        ax_curves.annotate(f'Max: {max_Iq:.3e}',
                           xy=(max_q, max_Iq),
                           xytext=(max_q * 1.1, max_Iq * 0.9),
                           arrowprops=dict(arrowstyle='->', color='blue'),
                           fontsize=12)

    # 配置权重图
    ax_weights.set_xlabel('Index')
    ax_weights.set_ylabel('Weight')
    ax_weights.set_title('Weight Distributions for 3 Samples')
    ax_weights.legend(fontsize=12)
    ax_weights.grid(True)

    # 配置合成曲线图
    ax_curves.set_xlabel('q (Å^{-1})')
    ax_curves.set_ylabel('I(q) (a.u.)')
    ax_curves.set_title('Synthesized Curves (Pure Type 3)')
    ax_curves.legend(fontsize=12)
    ax_curves.grid(True, which='both', ls='--', lw=0.5)
    ax_curves.set_xscale('log')
    ax_curves.set_yscale('log')

    plt.tight_layout()
    plt.show()
