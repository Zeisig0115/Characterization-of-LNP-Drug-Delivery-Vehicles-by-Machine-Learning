import h5py
import numpy as np
import matplotlib.pyplot as plt
import re
import random


#######################
# MAIN SCRIPT
#######################

hdf5_file = './dict/type1.h5'

with h5py.File(hdf5_file, 'r') as f:
    q = f['/q'][:]

    iq_datasets = [key for key in f.keys() if key != 'q']
    print("Available Iq datasets:", iq_datasets)

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

    # Example: 真实空间均值为 60，标准差为 2 的对数正态分布
    mean_real = 20  # 真实空间均值
    sigma_real = 2  # 真实空间标准差

    # 转换为对数正态分布的参数 mu 和 sigma_log
    mu = np.log(mean_real ** 2 / np.sqrt(mean_real ** 2 + sigma_real ** 2))
    sigma_log = np.sqrt(np.log(1 + (sigma_real ** 2 / mean_real ** 2)))

    # 确保直径为正（对数正态分布要求）
    if np.any(diameters <= 0):
        raise ValueError("All diameters must be positive for log-normal distribution.")

    # 生成对数正态分布权重
    weights = (1.0 / (diameters * sigma_log * np.sqrt(2 * np.pi))) * \
              np.exp(- (np.log(diameters) - mu) ** 2 / (2 * sigma_log ** 2))

    # 归一化权重
    weights /= np.sum(weights)

    # Print out the weights for each dataset
    for dataset_name, diameter, weight in zip(iq_datasets, diameters, weights):
        print(f"Dataset {dataset_name} (Diameter {diameter}) assigned weight {weight}")

    # Weighted sum of I(q)
    Iq_weighted_sum = np.zeros_like(f[iq_datasets[0]][:])
    for i, dataset_name in enumerate(iq_datasets):
        Iq = f[dataset_name][:]
        Iq_weighted_sum += weights[i] * Iq

    # Plot the weight distribution vs. diameter
    plt.figure(figsize=(8, 6))
    plt.plot(diameters, weights, 'o-', label='Weight Distribution')
    plt.xlabel('Diameter')
    plt.ylabel('Weight')
    plt.title('Log-Normal Weight Distribution')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    # Plot a few random dataset curves plus the weighted sum
    selected_datasets = random.sample(iq_datasets, 5)
    plt.figure()
    for dataset_name in selected_datasets:
        Iq_temp = f[dataset_name][:]
        plt.loglog(q.flatten(), Iq_temp.flatten(), label=dataset_name, linewidth=0.5)
    plt.loglog(q.flatten(), Iq_weighted_sum.flatten(), 'r-', linewidth=2,
               label='Weighted Synthesized Curve')

    plt.xlabel('q (Å^{-1})')
    plt.ylabel('I(q) (a.u.)')
    plt.title('I(q) vs q with Weighted Synthesized Curve')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.show(block=True)


