
import os
import torch
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder



class CustomDataset(Dataset):
    def __init__(self, root_dir, max_samples=100000, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []
        self.sub_dirs = ["axion", "cdm", "no_sub"]

        # Evenly distribute samples across folders
        samples_per_folder = max_samples // len(self.sub_dirs)

        for sub_dir in self.sub_dirs:
            sub_path = os.path.join(root_dir, sub_dir)
            if os.path.exists(sub_path):
                files = [os.path.join(sub_path, f) for f in os.listdir(sub_path) if f.endswith('.npy')]
                random.shuffle(files)  # Shuffle before sampling
                self.file_list.extend(files[:samples_per_folder])

        random.shuffle(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path, allow_pickle=True)

        if "axion" in file_path:
            data = data[0]
            
        # Normalize data
        data = (data - np.min(data)) / (np.max(data) - np.min(data))  # Avoid division by zero

        # Convert to torch tensor
        data = torch.from_numpy(data).float()
        data = data.unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            data = self.transform(data)

        return data 


def load_data(config):

    # data set
    dataset = CustomDataset(root_dir=config.data.folder)


    data_loader = DataLoader(dataset=dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle)
    
    return data_loader

if __name__ == "__main__":

    # Read config file
    from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
    # Read the configuration file
    config_name = "default"
    pipe = ConfigPipeline(
        [
            YamlConfig(
                "./gl_ijepa.yaml", config_name='default', config_folder='cfg/'
            ),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(config_folder='cfg/')
        ]
    )
    config = pipe.read_conf()

    # Random seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.benchmark =False

    ## load dataloder
    dataloader = load_data(config)
    # Iterate through the dataloader
    for batch in dataloader:
        print(batch[2][1].shape)  # Example check
        exit()