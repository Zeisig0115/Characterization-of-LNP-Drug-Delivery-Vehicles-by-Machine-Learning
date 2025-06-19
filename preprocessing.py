import h5py
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Read the input HDF5 file
input_file_path = "./output/raw4.h5"       # Modify this to the actual path
output_file_path = "./output/clean4.h5"    # Path for the output file

with h5py.File(input_file_path, "r") as h5_file:
    dataset_names = list(h5_file.keys())

    # Extract the q_min from each dataset and find the minimum q_min across all
    q_mins = [h5_file[name][0, 0] for name in dataset_names]
    q_min = min(q_mins)
    q_max = 0.5  # Fixed q_max

    # Generate a unified q-axis
    N_fixed = 500
    q_fixed = np.linspace(q_min, q_max, N_fixed)

    # Create a new HDF5 file for output
    with h5py.File(output_file_path, "w") as h5_out:
        # Store the standardized q-axis
        h5_out.create_dataset("q_fixed", data=q_fixed)

        # Interpolate each dataset and write to new HDF5
        for dataset in dataset_names:
            q_orig, Iq_orig = h5_file[dataset][:]

            # Apply cubic spline interpolation
            spline_func = CubicSpline(q_orig, Iq_orig, extrapolate=True)
            Iq_spline = spline_func(q_fixed)  # Interpolated I(q) without noise

            h5_out.create_dataset(dataset, data=Iq_spline)

print(f"All datasets have been interpolated and saved to {output_file_path}")
