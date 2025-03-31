import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

# load model
from models.model import MAE_Encoder, ViT_Classifier, MAE_ViT

# logging
import logging
import yaml
import ruamel.yaml.scalarfloat

# Read config file
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

# dataloader
from dataset.preprocessing_classifier import get_dataloaders

# Random seed
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.benchmark =False

# Read the configuration file
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./classifier.yaml", config_name='default', config_folder='cfg/'
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder='cfg/')
    ]
)
config = pipe.read_conf()

# Read the configuration file for the encoder
encoder_config_name = "default"
encoder_pipe = ConfigPipeline(
    [
        YamlConfig(
            "./mae.yaml", config_name='default', config_folder='cfg/'
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder='cfg/')
    ]
)
encoder_config = encoder_pipe.read_conf()


# Load data
train_loader, val_loader, test_loader = get_dataloaders(config)

# Load classifier model
encoder = MAE_Encoder(encoder_config)
mae_model = MAE_ViT(encoder_config) 

# # Construct the filename dynamically using config values
model_filename = (
    f"mae_enc_patch_size{encoder_config.enc.patch_size}_"
    f"img_size{encoder_config.enc.img_size}_in_chs{encoder_config.enc.in_chs}_"
    f"emb_dim{encoder_config.enc.emb_dim}_depth{encoder_config.enc.depth}_num_head{encoder_config.enc.num_head}_"
    f"mask_ratio{encoder_config.enc.mask_ratio}_"
    f"dec_emb_dim{encoder_config.dec.emb_dim}_dec_depth{encoder_config.dec.depth}_"
    f"dec_num_head{encoder_config.dec.num_head}_dec_mlp_ratio{encoder_config.dec.mlp_ratio}.pth"
)

# # Define the model path
model_path = f"saved_models/mae/50k_{model_filename}_final.pth"

# # load pretrained model
checkpoint = torch.load(model_path, map_location=config.device)
mae_model.load_state_dict(checkpoint, strict=False)  # Load all weights

# # Extract encoder weights
encoder_state_dict = mae_model.encoder.state_dict()

# Initialize only the encoder with extracted weights
encoder.load_state_dict(encoder_state_dict)

# Freeze encoder parameters
for param in encoder.parameters():
    param.requires_grad = False  # This prevents updates

model = ViT_Classifier(encoder=encoder).to(config.device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Configure logging
log_filename = "log/training_finetune.txt"
logging.basicConfig(
    filename=log_filename,  # Save logs to a file
    filemode="a",  
    format="%(message)s",
    level=logging.INFO,
)

logger = logging.getLogger()
logger.info("----------------------------------Classifer training using pretrained VIT started ----------------------------------------")

def clean_config(obj):
    """Recursively convert Bunch objects and ScalarFloats to standard Python types."""
    if isinstance(obj, dict):
        return {k: clean_config(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):  # Convert Bunch-like objects
        return {k: clean_config(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, ruamel.yaml.scalarfloat.ScalarFloat):  # Convert ScalarFloat to float
        return float(obj)
    else:
        return obj  # Return as is for non-dictionary values

# Convert config to a clean dictionary
config_dict = clean_config(config)

# Log the configuration in YAML format
logger.info("Configuration:\n" + yaml.dump(config_dict, default_flow_style=False))

# Log total trainable parameters
total_params = count_parameters(model)
print(f"Total Trainable Parameters: {total_params}")
logger.info(f"Total Trainable Parameters: {total_params:,}")

# training
num_epochs = config.train.epochs
best_val_acc = 0.0  # Track the best validation accuracy
best_model_path = "saved_models/finetune/finetune_best_model_for_model_3.pth"

for epoch in range(num_epochs):
    # Training Phase
    model.train()
    total_loss = 0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(config.device), labels.to(config.device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
    logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

    # Validation Phase
    model.eval()
    val_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            # Compute validation accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Acc: {val_acc:.4f}")
    logger.info(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        improvement = val_acc - best_val_acc
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        logger.info("🚀 New Best Model Found! 🚀")
        logger.info(f"Epoch {epoch+1}: Validation Acc Improved by {improvement:.4f}")
        logger.info(f"🔥 Best Model Saved with Validation Acc: {val_acc:.4f} 🔥")


# Load the best model for final testing
model.load_state_dict(torch.load(best_model_path))
model.to(config.device)
model.eval()

# Testing Phase
test_loss = 0
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(config.device), labels.to(config.device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()

        # Compute test accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
logger.info(f"Best model test Loss: {test_loss/len(test_loader):.4f}, Best model test Acc: {test_acc:.4f}")
print(f"Best model test Loss: {test_loss/len(test_loader):.4f}, Best model test Acc: {test_acc:.4f}")


logger.info("---------------------------Claasification training completed pretrained. Model saved.----------------------------------------------------")

