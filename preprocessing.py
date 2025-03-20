import h5py
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt



# 读取 HDF5 文件
input_file_path = "./output/raw4.h5"  # 修改为实际路径
output_file_path = "./output/clean4.h5"  # 输出文件路径

with h5py.File(input_file_path, "r") as h5_file:
    dataset_names = list(h5_file.keys())

    # 提取所有数据集的 q_min，并找到最大的 q_min
    q_mins = [h5_file[name][0, 0] for name in dataset_names]
    q_min = min(q_mins)  # 选择所有数据集中最大的 q_min
    q_max = 0.5  # 固定的 q_max

    # 生成统一 q 轴
    N_fixed = 500
    q_fixed = np.linspace(q_min, q_max, N_fixed)

    # 创建新的 HDF5 文件
    with h5py.File(output_file_path, "w") as h5_out:
        # 存储标准 q 轴
        h5_out.create_dataset("q_fixed", data=q_fixed)

        # 遍历所有数据集进行插值和添加噪声
        for dataset in dataset_names:
            q_orig, Iq_orig = h5_file[dataset][:]

            # 三次样条插值
            spline_func = CubicSpline(q_orig, Iq_orig, extrapolate=True)
            Iq_spline = spline_func(q_fixed)  # 此时 Iq_spline 就是无噪声的 I(q)

            h5_out.create_dataset(dataset, data=Iq_spline)

print(f"所有数据已插值 {output_file_path}")
