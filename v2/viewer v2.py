import h5py
import numpy as np
import matplotlib.pyplot as plt
import re
import random

#######################
# MAIN SCRIPT
#######################

# 读取插值后的 HDF5 文件
hdf5_file = "./output/interpolated_type3.h5"  # 请修改为实际路径

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

    # 设置对数正态分布参数（示例: 均值 60，标准差 2）
    mean_real = 60  # 真实空间均值
    sigma_real = 0.1  # 真实空间标准差

    # 转换为对数正态分布的 mu 和 sigma_log
    mu = np.log(mean_real ** 2 / np.sqrt(mean_real ** 2 + sigma_real ** 2))
    sigma_log = np.sqrt(np.log(1 + (sigma_real ** 2 / mean_real ** 2)))

    # 计算权重（对数正态分布）
    weights = (1.0 / (diameters * sigma_log * np.sqrt(2 * np.pi))) * \
              np.exp(- (np.log(diameters) - mu) ** 2 / (2 * sigma_log ** 2))

    # 归一化权重
    weights /= np.sum(weights)

    # 打印每个数据集的权重
    for dataset_name, diameter, weight in zip(iq_datasets, diameters, weights):
        print(f"Dataset {dataset_name} (Diameter {diameter}) assigned weight {weight:.6f}")

    # 计算加权合成 SAXS 曲线
    Iq_weighted_sum = np.zeros_like(f[iq_datasets[0]][:])
    for i, dataset_name in enumerate(iq_datasets):
        Iq = f[dataset_name][:]
        Iq_weighted_sum += weights[i] * Iq

    # 绘制权重分布 vs. 直径
    plt.figure(figsize=(8, 6))
    plt.plot(diameters, weights, 'o-', label='Weight Distribution')
    plt.xlabel('Diameter (nm)')
    plt.ylabel('Weight')
    plt.title('Log-Normal Weight Distribution')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    # 随机选择 5 条 I(q) 曲线，并绘制加权合成曲线
    selected_datasets = random.sample(iq_datasets, 5)
    plt.figure()
    # for dataset_name in selected_datasets:
    #     Iq_temp = f[dataset_name][:]
    #     plt.loglog(q.flatten(), Iq_temp.flatten(), label=dataset_name, linewidth=0.5)
    plt.loglog(q.flatten(), Iq_weighted_sum.flatten(), 'r-', linewidth=2,
               label='Weighted Synthesized Curve')

    plt.xlabel('q (Å^{-1})')
    plt.ylabel('I(q) (a.u.)')
    plt.title('I(q) vs q with Weighted Synthesized Curve')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.show(block=True)
