import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import time

def is_point_inside_ellipsoid(points, ellipsoid):
    a, b, c, x0, y0, z0 = ellipsoid
    return (((points[..., 0] - x0) / a) ** 2 +
            ((points[..., 1] - y0) / b) ** 2 +
            ((points[..., 2] - z0) / c) ** 2) <= 1

def is_point_inside_sphere(points, radius, center):
    return np.sum((points - center) ** 2, axis=-1) <= radius ** 2

def is_point_inside_cylinder(points, radius, height, center):
    x, y, z = points[..., 0] - center[0], points[..., 1] - center[1], points[..., 2] - center[2]
    return (x**2 + y**2 <= radius**2) & (np.abs(z) <= height / 2)

def lnp(cube_size, sld_core, sld_shell, sld_fill, sld_solvent, alpha, type_option):
    # Create 3D coordinate grid
    x, y, z = np.meshgrid(np.arange(cube_size), np.arange(cube_size), np.arange(cube_size), indexing='ij')
    points = np.stack([x, y, z], axis=-1)
    cube = np.zeros((cube_size, cube_size, cube_size))

    # Define geometric parameters
    r1 = 28.57 * alpha
    r2 = 35.71 * alpha
    shell = 4
    l = 100

    outer1 = np.array([r1, r1, r1, 50, 50, (1 + alpha) * l / 2 - r1])
    outer2 = np.array([r1, r1, r2, 50, 50, (1 - alpha) * l / 2 + r2])
    inner1 = np.array([r1 - shell, r1 - shell, r1 - shell, 50, 50, (1 + alpha) * l / 2 - r1])
    inner2 = np.array([r1 - shell, r1 - shell, r2 - shell, 50, 50, (1 - alpha) * l / 2 + r2])

    # Calculate the z-plane for cutoff
    z0_inner1 = inner1[5]
    z0_inner2 = inner2[5]
    plane0 = (z0_inner1 + z0_inner2) / 2
    plane_up = plane0 + 2
    plane_down = plane0 - 2

    # Check point membership for each region
    inside_inner1 = is_point_inside_ellipsoid(points, inner1)
    inside_inner2 = is_point_inside_ellipsoid(points, inner2)
    inside_outer1 = is_point_inside_ellipsoid(points, outer1)
    inside_outer2 = is_point_inside_ellipsoid(points, outer2)

    if type_option == 1:
        # Type 1: Standard core–shell structure
        core_mask = inside_inner1 | inside_inner2
        shell_mask = inside_outer1 | inside_outer2
        cube[core_mask] = sld_core
        cube[shell_mask & ~core_mask] = sld_shell
        cube[~(core_mask | shell_mask)] = sld_solvent

    elif type_option == 2:
        # Type 2: Core only in upper half, fill in lower half
        core_mask = (inside_inner1 & (points[..., 2] >= plane_up))
        fill_mask = (inside_inner2 & (points[..., 2] <= plane_down))
        shell_mask = (inside_outer1 | inside_outer2) & ~core_mask & ~fill_mask
        cube[core_mask] = sld_core
        cube[fill_mask] = sld_fill
        cube[shell_mask] = sld_shell
        cube[~(core_mask | fill_mask | shell_mask)] = sld_solvent

    elif type_option == 3:
        # Type 3: Core in both ends, hollow middle
        core_mask = (inside_inner1 & (points[..., 2] >= plane_up)) | \
                    (inside_inner2 & (points[..., 2] <= plane_down))
        shell_mask = inside_outer1 | inside_outer2
        cube[core_mask] = sld_core
        cube[shell_mask & ~core_mask] = sld_shell
        cube[~(core_mask | shell_mask)] = sld_solvent

    elif type_option == 4:
        # Type 4: Sphere-only model
        radius = 50 * alpha
        center = np.array([cube_size // 2, cube_size // 2, cube_size // 2])
        sphere_mask = np.sum((points - center) ** 2, axis=-1) <= radius ** 2
        cube[sphere_mask] = sld_shell  # or sld_core if needed

    elif type_option == 5:
        # Type 5: Cylinder model
        cylinder_radius = 10
        cylinder_height = 50
        cylinder_center = np.array([cube_size // 2, cube_size // 2, cube_size // 2])
        cylinder_mask = is_point_inside_cylinder(points, cylinder_radius, cylinder_height, cylinder_center)
        cube[cylinder_mask] = 1  # Set to sld_core or 1

    else:
        raise ValueError("Invalid type_option. Choose between 1, 2, 3, 4 or 5.")

    return cube

def visualize_lnp(cube, sld_core, sld_shell, sld_fill, sld_solvent):
    # Define a boolean mask for all visible voxels
    voxels = (cube == sld_core) | (cube == sld_shell) | (cube == sld_fill) | (cube == sld_solvent)

    # Assign colors based on material types
    colors = np.empty(cube.shape, dtype=object)
    colors[cube == sld_core] = 'red'
    colors[cube == sld_shell] = 'green'
    colors[cube == sld_fill] = 'blue'
    colors[cube == sld_solvent] = 'cyan'

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def visualize_slice(cube, slice_index, alpha):
    # Visualize a 2D slice of the 3D model at a specified Y index
    slice_data = cube[:, slice_index, :]
    plt.figure()
    plt.imshow(slice_data, cmap='viridis', origin='lower')
    plt.colorbar(label='SLD')
    plt.title(f'Slice at y={slice_index}, diameter={100 * alpha:.2f}')
    plt.xlabel('Z')
    plt.ylabel('X')
    plt.show(block=True)

def save_lnp_data(alpha, cube, X, Y, Z):
    # Save the 3D LNP model and coordinate axes to a .mat file
    directory = 'type_4_model'
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f'd{int(round(alpha*100))}.mat')
    scipy.io.savemat(filename, {'rhoS': cube, 'X': X, 'Y': Y, 'Z': Z})

def main():
    # Model parameters
    cube_size = 100
    sld_core = 16.0
    sld_shell = 9.0
    sld_fill = sld_shell
    sld_solvent = 0.0

    # Generate coordinate grids
    X = np.linspace(-cube_size / 2, cube_size / 2, cube_size)
    Y = np.linspace(-cube_size / 2, cube_size / 2, cube_size)
    Z = np.linspace(-cube_size / 2, cube_size / 2, cube_size)

    # Alpha controls overall particle size
    alphas = np.linspace(0.2, 1, 81)

    for alpha in alphas:
        cube = lnp(cube_size, sld_core, sld_shell, sld_fill, sld_solvent, alpha, type_option=4)
        save_lnp_data(alpha, cube, X, Y, Z)
        # visualize_slice(cube, 50, alpha)
        # visualize_lnp(cube, sld_core, sld_shell, sld_fill, sld_solvent)

if __name__ == "__main__":
    st = time.time()
    main()
    et = time.time()
    print(et - st)
