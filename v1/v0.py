import re
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from model import simpleCNN, DeepResNet1D

# ------------- GPU device -------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_type1_data(hdf5_file='./dict/type1.h5'):
    with h5py.File(hdf5_file, 'r') as f:
        q = f['/q'][:]
        q = np.squeeze(q)
        iq_datasets = [key for key in f.keys() if key != 'q']

        diameters = []
        Iq_list = []
        for dataset_name in iq_datasets:
            diameter_match = re.search(r'\d+', dataset_name)
            if diameter_match:
                diameter = int(diameter_match.group())
                diameters.append(diameter)
                Iq = f[dataset_name][:]
                Iq = np.squeeze(Iq)
                Iq_list.append(Iq)
            else:
                print(f"Error")
                continue
        diameters = np.array(diameters)
        Iq_array = np.array(Iq_list)
        print("Iq_array shape:", Iq_array.shape)
    return q, diameters, Iq_array


def generate_synthesized_curve(mean, sigma, diameters, Iq_array):
    weights = np.exp(-((diameters - mean)**2) / (2 * sigma**2))
    weights /= np.sum(weights)
    Iq_weighted_sum = np.dot(weights, Iq_array)
    return Iq_weighted_sum


def create_dataset():
    num_samples = 10000
    means = np.random.uniform(low=30, high=90, size=num_samples)
    sigmas = np.random.uniform(low=1, high=5, size=num_samples)

    q_values, diameters, Iq_array = load_type1_data()

    mean_sigma_array = np.zeros((num_samples, 2))
    synthesized_curves = np.zeros((num_samples, len(q_values)))

    for i in range(num_samples):
        mean = means[i]
        sigma = sigmas[i]
        synthesized_curve = generate_synthesized_curve(mean, sigma, diameters, Iq_array)
        synthesized_curves[i] = synthesized_curve
        mean_sigma_array[i, 0] = mean
        mean_sigma_array[i, 1] = sigma

    with h5py.File('synthesized_dataset.h5', 'w') as f_out:
        f_out.create_dataset('q', data=q_values)
        f_out.create_dataset('curves', data=synthesized_curves)
        f_out.create_dataset('mean_sigma', data=mean_sigma_array)


class SynthesizedCurveDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, 'r') as f:
            self.curves = f['curves'][:]
            self.mean_sigma = f['mean_sigma'][:]
            self.q_values = f['q'][:]
        self.num_samples = self.mean_sigma.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        curve = self.curves[idx]
        mean_sigma = self.mean_sigma[idx]
        curve = torch.tensor(curve, dtype=torch.float32)
        mean_sigma = torch.tensor(mean_sigma, dtype=torch.float32)
        return curve, mean_sigma

    def get_q_values(self):
        return self.q_values


def mean_pred():
    dataset = SynthesizedCurveDataset('synthesized_dataset.h5')
    input_length = dataset.curves.shape[1]
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = DeepResNet1D(input_length).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for curves, mean_sigma in train_loader:
            # Move data to GPU
            curves = curves.to(device)
            mean_sigma = mean_sigma.to(device)

            optimizer.zero_grad()
            outputs = model(curves)
            mean_targets = mean_sigma[:, 0].unsqueeze(1)
            loss = criterion(outputs, mean_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * curves.size(0)

        epoch_loss = running_loss / train_size
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        true_means = []
        predicted_means = []
        for curves, mean_sigma in test_loader:
            curves = curves.to(device)
            mean_sigma = mean_sigma.to(device)

            outputs = model(curves)
            mean_targets = mean_sigma[:, 0].unsqueeze(1)
            loss = criterion(outputs, mean_targets)
            total_loss += loss.item() * curves.size(0)
            true_means.extend(mean_targets.squeeze().cpu().tolist())
            predicted_means.extend(outputs.squeeze().cpu().tolist())

        test_loss = total_loss / test_size
        print(f'Test Loss: {test_loss:.4f}')

    # Plot training loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epochs')
    plt.legend()
    plt.show()

    # Scatter plot of predictions vs. true
    true_means = np.array(true_means)
    predicted_means = np.array(predicted_means)
    plt.figure(figsize=(8, 6))
    plt.scatter(true_means, predicted_means, alpha=0.5)
    plt.plot([true_means.min(), true_means.max()], [true_means.min(), true_means.max()], 'r--')
    plt.xlabel('True Mean Values')
    plt.ylabel('Predicted Mean Values')
    plt.title('Predicted vs. True Mean Values on Test Set')
    plt.show(block=True)


def sigma_pred():
    dataset = SynthesizedCurveDataset('synthesized_dataset.h5')
    input_length = dataset.curves.shape[1]
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 更深网络
    model = DeepResNet1D(input_length).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for curves, mean_sigma in train_loader:
            # Move data to GPU
            curves = curves.to(device)
            mean_sigma = mean_sigma.to(device)

            optimizer.zero_grad()
            outputs = model(curves)
            sigma_targets = mean_sigma[:, 1].unsqueeze(1)  # sigma
            loss = criterion(outputs, sigma_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * curves.size(0)

        epoch_loss = running_loss / train_size
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        true_sigmas = []
        predicted_sigmas = []
        for curves, mean_sigma in test_loader:
            curves = curves.to(device)
            mean_sigma = mean_sigma.to(device)

            outputs = model(curves)
            sigma_targets = mean_sigma[:, 1].unsqueeze(1)
            loss = criterion(outputs, sigma_targets)
            total_loss += loss.item() * curves.size(0)
            true_sigmas.extend(sigma_targets.squeeze().cpu().tolist())
            predicted_sigmas.extend(outputs.squeeze().cpu().tolist())

        test_loss = total_loss / test_size
        print(f'Test Loss: {test_loss:.4f}')

    # Plot training loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epochs')
    plt.legend()
    plt.show()

    # Plot predicted vs. true sigma values
    true_sigmas = np.array(true_sigmas)
    predicted_sigmas = np.array(predicted_sigmas)
    plt.figure(figsize=(8, 6))
    plt.scatter(true_sigmas, predicted_sigmas, alpha=0.5)
    plt.plot([true_sigmas.min(), true_sigmas.max()], [true_sigmas.min(), true_sigmas.max()], 'r--')
    plt.xlabel('True Sigma Values')
    plt.ylabel('Predicted Sigma Values')
    plt.title('Predicted vs. True Sigma Values on Test Set')
    plt.show(block=True)


if __name__ == '__main__':
    # create_dataset()  # 如果还没生成 h5 数据，可先执行
    mean_pred()
    # sigma_pred()
    
