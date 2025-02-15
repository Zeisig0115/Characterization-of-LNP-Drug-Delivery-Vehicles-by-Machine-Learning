import h5py
import numpy as np
from scipy.interpolate import CubicSpline

# 读取 HDF5 文件
input_file_path = "./output/type4.h5"  # 修改为实际路径
output_file_path = "./output/test4.h5"  # 输出文件路径

with h5py.File(input_file_path, "r") as h5_file:
    dataset_names = list(h5_file.keys())

    # 提取所有数据集的 q_min，并找到最大的 q_min
    q_mins = [h5_file[name][0, 0] for name in dataset_names]
    q_min = max(q_mins)  # 选择所有数据集中最大的 q_min
    q_max = 0.5  # 固定的 q_max

    # 生成统一 q 轴
    N_fixed = 300
    q_fixed = np.linspace(q_min, q_max, N_fixed)

    # 创建新的 HDF5 文件
    with h5py.File(output_file_path, "w") as h5_out:
        # 存储标准 q 轴
        h5_out.create_dataset("q_fixed", data=q_fixed)

        # 遍历所有数据集进行插值和添加噪声
        for dataset in dataset_names:
            q_orig, Iq_orig = h5_file[dataset][:]

            Iq_orig /= (4 * np.pi)

            # 三次样条插值
            spline_func = CubicSpline(q_orig, Iq_orig, extrapolate=True)
            Iq_spline = spline_func(q_fixed)  # 此时 Iq_spline 就是无噪声的 I(q)


            # Prevent some extreme value in Iq_spline causing /0 problem
            Iq_safe = np.maximum(Iq_spline, 1e-12)

            # Sample α obeying log-uniform distribution ，range [10^2, 10^(5.5)]
            log_alpha = np.random.uniform(np.log(1e5), np.log(10 ** 8.5))
            alpha = np.exp(log_alpha)

            # Calculate sigma^2 for each of q: sigma2 = ln(1 + alpha / I(q))
            sigma2 = np.log(1 + alpha / Iq_safe)

            # Generate normal variable epsilon for each q
            epsilon = np.random.randn(len(Iq_safe))

            # Calculate noise coefficient：N(q) = exp(sqrt(sigma2)*epsilon - sigma2/2)
            noise_factor = np.exp(np.sqrt(sigma2) * epsilon - sigma2 / 2)

            # Calculate the noisified I(q)
            Iq_noisy = Iq_spline * noise_factor

            # Normalize the scattering intensity
            total_intensity = np.trapz(Iq_noisy, q_fixed)
            Iq_noisy_normalized = Iq_noisy / total_intensity


            h5_out.create_dataset(dataset, data=Iq_noisy_normalized)

print(f"所有数据已插值、添加噪声并保存到 {output_file_path}")
