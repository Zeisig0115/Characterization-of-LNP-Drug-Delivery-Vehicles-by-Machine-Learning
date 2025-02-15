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
        q = f['q_fixed'][:]
        # 获取除 q_fixed 外的所有数据集名称
        dataset_names = [key for key in f.keys() if key != 'q_fixed']
        # 将所有 I(q) 曲线存入列表
        curves = [f[name][:] for name in dataset_names]
    return q, curves, dataset_names

if __name__ == '__main__':
    # 设置 type3 与 type4 的 HDF5 文件路径（请根据实际情况修改）
    type3_file = "./output/test3.h5"
    type4_file = "./output/test4.h5"

    # 分别加载 type3 与 type4 的数据
    q3, curves3, names3 = load_saxs_data(type3_file)
    q4, curves4, names4 = load_saxs_data(type4_file)

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
        # synthesized_curve = alpha * curve3 + (1 - alpha) * curve4
        synthesized_curve = curve3

        # # 绘制合成的曲线（对数坐标）
        # plt.loglog(q.flatten(), synthesized_curve.flatten(),
        #            lw=2, label=f"Example {i+1} (α = {alpha:.2f})")

        plt.scatter(q.flatten(), synthesized_curve.flatten(),
                    label=f"Example {i + 1} (α = {alpha:.2f})", s=15)

    # 设置对数坐标
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("q (Å$^{-1}$)")
    plt.ylabel("I(q) (a.u.)")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()
