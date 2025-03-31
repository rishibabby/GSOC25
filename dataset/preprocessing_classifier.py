import os
import torch
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split

class CustomDataset(Dataset):
    def __init__(self, root_dir, max_samples=10000, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []
        self.labels = []
        self.sub_dirs = {"axion": 0, "cdm": 1, "no_sub": 2}  # Assign labels to subdirectories

        # Evenly distribute samples across folders
        samples_per_folder = max_samples // len(self.sub_dirs)

        for sub_dir, label in self.sub_dirs.items():
            sub_path = os.path.join(root_dir, sub_dir)
            if os.path.exists(sub_path):
                files = [os.path.join(sub_path, f) for f in os.listdir(sub_path) if f.endswith('.npy')]
                random.shuffle(files)  # Shuffle before sampling
                selected_files = files[:samples_per_folder]

                self.file_list.extend(selected_files)
                self.labels.extend([label] * len(selected_files))  # Assign label to each sample

        # Shuffle again to mix data from all folders
        combined = list(zip(self.file_list, self.labels))
        random.shuffle(combined)
        self.file_list, self.labels = zip(*combined)
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]
        data = np.load(file_path, allow_pickle=True)

        if "axion" in file_path:
            data = data[0]  # Extract first element for axion category

        # Normalize data
        data = (data - np.min(data)) / (np.max(data) - np.min(data))  # Avoid division by zero

        # Convert to torch tensor and add channel dimension
        data = torch.from_numpy(data).float().unsqueeze(0)
        
        if self.transform:
            data = self.transform(data)
        
        return data, torch.tensor(label, dtype=torch.long)  # Return both data and label

def get_dataloaders(config):

    dataset = CustomDataset(config.data.folder, max_samples=10000)

    # Compute split sizes
    train_size = int(config.data.train_split * len(dataset))
    val_size = int(config.data.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size  # Ensure total sums up correctly

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader