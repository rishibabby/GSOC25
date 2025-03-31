import random
import torch
import numpy as np

# load data
from dataset.preprocessing_mae import load_data

# Read config file
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

# Load model
from models.model import MAE_ViT

# train
from train.train_mae import train_mae

# Read the configuration file
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./mae.yaml", config_name='default', config_folder='cfg/'
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

# Dataloader
dataloader = load_data(config)

# Model
model = MAE_ViT(config) 

# Train the model
model = train_mae(epochs = config.train.epochs, lr = config.train.lr, batch_size = config.data.batch_size, data_loader=dataloader, device=config.device,  model=model, config=config)

