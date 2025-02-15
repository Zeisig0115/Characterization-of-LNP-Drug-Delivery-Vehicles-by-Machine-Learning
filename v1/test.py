import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import time
from scipy.spatial.transform import Rotation as R  # 导入旋转模块

def is_point_inside_ellipsoid(points, ellipsoid):
    a, b, c, x0, y0, z0 = ellipsoid
    return (((points[..., 0] - x0) / a) ** 2 +
            ((points[..., 1] - y0) / b) ** 2 +
            ((points[..., 2] - z0) / c) ** 2) <= 1

def is_point_inside_sphere(points, radius, center):
    return np.sum((points - center) ** 2, axis=-1) <= radius ** 2

def lnp(cube_size, sld_core, sld_shell, sld_fill, sld_solvent, alpha, type_option, rotation=False):
    # 创建3D空间
    x, y, z = np.meshgrid(np.arange(cube_size), np.arange(cube_size), np.arange(cube_size), indexing='ij')
    points = np.stack([x, y, z], axis=-1)
    cube = np.zeros((cube_size, cube_size, cube_size))

    # 如果启用旋转，则对点进行旋转
    if rotation:
        cube_center = np.array([cube_size / 2, cube_size / 2, cube_size / 2])
        points_centered = points - cube_center
        rot = R.random()
        rot_matrix = rot.as_matrix()
        points_flat = points_centered.reshape(-1, 3)
        rotated_points_flat = np.dot(points_flat, rot_matrix.T)
        points = rotated_points_flat.reshape(cube_size, cube_size, cube_size, 3) + cube_center

    r1 = 28.57 * alpha
    r2 = 35.71 * alpha
    shell = 4
    l = 100

    outer1 = np.array([r1, r1, r1, 50, 50, (1 + alpha) * l / 2 - r1])
    outer2 = np.array([r1, r1, r2, 50, 50, (1 - alpha) * l / 2 + r2])
    inner1 = np.array([r1 - shell, r1 - shell, r1 - shell, 50, 50, (1 + alpha) * l / 2 - r1])
    inner2 = np.array([r1 - shell, r1 - shell, r2 - shell, 50, 50, (1 - alpha) * l / 2 + r2])

    # 计算平面位置
    z0_inner1 = inner1[5]
    z0_inner2 = inner2[5]
    plane0 = (z0_inner1 + z0_inner2) / 2
    plane_up = plane0 + 2
    plane_down = plane0 - 2

    inside_inner1 = is_point_inside_ellipsoid(points, inner1)
    inside_inner2 = is_point_inside_ellipsoid(points, inner2)
    inside_outer1 = is_point_inside_ellipsoid(points, outer1)
    inside_outer2 = is_point_inside_ellipsoid(points, outer2)

    if type_option == 1:
        core_mask = inside_inner1 | inside_inner2
        shell_mask = inside_outer1 | inside_outer2
        cube[core_mask] = sld_core
        cube[shell_mask & ~core_mask] = sld_shell
        cube[~(core_mask | shell_mask)] = sld_solvent
    elif type_option == 2:
        core_mask = (inside_inner1 & (points[..., 2] >= plane_up))
        fill_mask = (inside_inner2 & (points[..., 2] <= plane_down))
        shell_mask = (inside_outer1 | inside_outer2) & ~core_mask & ~fill_mask
        cube[core_mask] = sld_core
        cube[fill_mask] = sld_fill
        cube[shell_mask] = sld_shell
        cube[~(core_mask | fill_mask | shell_mask)] = sld_solvent
    elif type_option == 3:
        core_mask = (inside_inner1 & (points[..., 2] >= plane_up)) | \
                    (inside_inner2 & (points[..., 2] <= plane_down))
        shell_mask = inside_outer1 | inside_outer2
        cube[core_mask] = sld_core
        cube[shell_mask & ~core_mask] = sld_shell
        cube[~(core_mask | shell_mask)] = sld_solvent
    elif type_option == 4:
        # 选项4：创建一个具有定义半径和电子密度的球体
        radius = 5
        center = np.array([cube_size // 2, cube_size // 2, cube_size // 2])
        sphere_mask = np.sum((points - center) ** 2, axis=-1) <= radius ** 2
        cube[sphere_mask] = 1  # 将球体内的电子密度设置为`sld_core`（即1）
    else:
        raise ValueError("Invalid type_option. Choose between 1, 2, 3 or 4.")

    return cube

def visualize_lnp(cube, sld_core, sld_shell, sld_fill, sld_solvent):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    core_indices = np.argwhere(cube == sld_core)
    shell_indices = np.argwhere(cube == sld_shell)
    fill_indices = np.argwhere(cube == sld_fill)
    solvent_indices = np.argwhere(cube == sld_solvent)

    ax.scatter(core_indices[:, 0], core_indices[:, 1], core_indices[:, 2], color='red', s=1, label='Core')
    ax.scatter(shell_indices[:, 0], shell_indices[:, 1], shell_indices[:, 2], color='green', s=1, label='Shell')
    # ax.scatter(fill_indices[:, 0], fill_indices[:, 1], fill_indices[:, 2], color='blue', s=1, label='Fill')
    # ax.scatter(solvent_indices[:, 0], solvent_indices[:, 1], solvent_indices[:, 2], color='cyan', s=1, label='Solvent')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show(block=True)

def visualize_slice(cube, slice_index, alpha):
    slice_data = cube[:, slice_index, :]
    plt.figure()
    plt.imshow(slice_data, cmap='viridis', origin='lower')
    plt.colorbar(label='SLD')
    plt.title(f'Slice at y={slice_index}, diameter={1000 * alpha:.2f}')
    plt.xlabel('Z')
    plt.ylabel('X')
    plt.show(block=True)

def save_lnp_data(alpha, cube, X, Y, Z, index=1):
    directory = 'type1_set'
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f'd{int(alpha*1000)}_{index}.mat')
    scipy.io.savemat(filename, {'rhoS': cube, 'X': X, 'Y': Y, 'Z': Z})

def main():
    # 参数设置
    cube_size = 100
    sld_core = 16.0
    sld_shell = 9.0
    sld_fill = sld_shell
    sld_solvent = 0.0

    # 生成网格坐标
    X = np.linspace(-cube_size / 2, cube_size / 2, cube_size)
    Y = np.linspace(-cube_size / 2, cube_size / 2, cube_size)
    Z = np.linspace(-cube_size / 2, cube_size / 2, cube_size)

    alphas = np.linspace(0.2, 0.8, 100)

    for alpha in alphas:
        for i in range(1, 4):
            cube = lnp(cube_size, sld_core, sld_shell, sld_fill, sld_solvent,
                       alpha, type_option=1, rotation=True)
            save_lnp_data(alpha, cube, X, Y, Z, index=i)
            # 如果需要可视化，可以取消下面的注释
            # visualize_slice(cube, 50, alpha)
            # visualize_lnp(cube, sld_core, sld_shell, sld_fill, sld_solvent)

if __name__ == "__main__":
    st = time.time()
    main()
    et = time.time()
    print(f"总耗时：{et - st:.2f} 秒")
