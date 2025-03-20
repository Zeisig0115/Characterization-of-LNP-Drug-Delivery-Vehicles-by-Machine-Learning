import h5py
import numpy as np
import matplotlib.pyplot as plt
import random

def load_saxs_data(filename):
    """
    从 HDF5 文件中加载 SAXS 数据。
    假设文件中包含一个 'q_fixed' 数据集作为标准 q 轴，
    其他数据集对应不同样本的 I(q) 曲线。
    """
    with h5py.File(filename, 'r') as f:
        # 读取标准 q 轴
        q = f['q_fixed'][:]  # 修改：从文件对象 f 中读取 "q_fixed" 数据
        # 获取除 q_fixed 外的所有数据集名称
        dataset_names = [key for key in f.keys() if key != 'q_fixed']
        # 将所有 I(q) 曲线存入列表
        curves = [f[name][:] for name in dataset_names]
    return q, curves, dataset_names


def add_noise_to_curve(curve, q=None):
    """
    给定理论 SAXS 曲线 I(q)，添加符合实验设计的噪声，并进行归一化处理。

    噪声模型：
      - 使用 lognormal 噪声模型，即对每个 q 值：
            I_noisy(q) = I(q) * N(q)
        其中 N(q) = exp( sqrt(sigma²(q)) * epsilon - sigma²(q)/2 )
        且 sigma²(q) = ln(1 + α / I(q))，使得 E[I_noisy(q)] = I(q)
        和 Var[I_noisy(q)] = α * I(q).

      - 噪声参数 α 从对数均匀分布中采样，范围 [1e2, 10^(5.5)]。

    归一化：
      - 若提供 q 值，则用 4πq² 权重计算积分（模拟 Porod 不变量），否则采用简单梯形积分。

    Parameters:
        curve: 1D numpy 数组，表示理论散射强度 I(q)。
        q: 1D numpy 数组，与 curve 对应的 q 值（可选）。

    Returns:
        添加噪声并归一化后的 I(q) 曲线（numpy 数组）。
    """
    # 防止除零问题
    curve_safe = np.maximum(curve, 1e-12)

    # 从对数均匀分布中采样噪声参数 α，范围 [1e4, 10^(7.5)]
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

    normalization = np.trapz(curve_noisy)

    return curve_noisy / 1


def add_noise_to_data(data):
    """
    对数据集中的每个样本（每行）添加噪声。
    data: numpy 数组，形状 (样本数, 特征维度)
    """
    noisy_data = np.array([add_noise_to_curve(data[i]) for i in range(data.shape[0])])
    return noisy_data


if __name__ == '__main__':
    # 设置 type3 与 type4 的 HDF5 文件路径（请根据实际情况修改）
    type3_file = "./output/clean3.h5"
    type4_file = "./output/clean4.h5"

    # 分别加载 type3 与 type4 的数据
    q3, curves3, names3 = load_saxs_data(type3_file)
    q4, curves4, names4 = load_saxs_data(type4_file)

    # X_type3 = np.vstack(curves3) if curves3 else np.array([])
    # X_type4 = np.vstack(curves4) if curves4 else np.array([])
    #
    # curves3 = add_noise_to_data(X_type3)
    # curves4 = add_noise_to_data(X_type4)

    # 假设两个文件的 q 轴是一致的
    if not np.allclose(q3, q4):
        print("Warning: type3 与 type4 的 q 轴不同！")
    q = q3

    # 设定要生成的 α 合成曲线数量
    num_examples = 2

    plt.figure(figsize=(10, 6))
    for i in range(num_examples):
        # 随机选取一条 type3 曲线和一条 type4 曲线
        idx3 = random.randint(0, len(curves3) - 1)
        idx4 = random.randint(0, len(curves4) - 1)
        curve3 = curves3[idx3]
        curve4 = curves4[idx4]

        # 随机选择混合系数 α（取值范围 [0, 1]）
        alpha = random.random()
        # α 合成：新曲线 = α * type3_curve + (1 - α) * type4_curve
        synthesized_curve = alpha * curve3 + (1 - alpha) * curve4
        # synthesized_curve = curve3

        synthesized_curve = add_noise_to_curve(synthesized_curve)

        # # 绘制合成的曲线（对数坐标）
        # plt.loglog(q.flatten(), synthesized_curve.flatten(),
        #            lw=2, label=f"Example {i+1} (α = {alpha:.2f})")

        plt.scatter(q.flatten(), synthesized_curve.flatten(),
                    label=f"Mixture Sample {i + 1} (α = {alpha:.2f})", s=15)

        # plt.scatter(q.flatten(), curve3.flatten(), s=15)

    # 设置对数坐标
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("q (nm$^{-1}$)")
    plt.ylabel("I(q) (a.u.)")
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()
