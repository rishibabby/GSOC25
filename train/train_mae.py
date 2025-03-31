import torch
import logging
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import csv
import yaml
import matplotlib.pyplot as plt
import ruamel.yaml.scalarfloat

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_mae(
             batch_size=64, 
             epochs=50, 
             lr=1.5e-3, 
             device='cuda' if torch.cuda.is_available() else 'cpu', data_loader=None, model=None, config=None):
    """
    Train the Masked Autoencoder
    
    Args:
        data_dir: Directory containing images
        img_size: Size of input images (default: 224)
        batch_size: Batch size for training (default: 16)
        epochs: Number of training epochs (default: 100)
        lr: Learning rate (default: 1.5e-4)
        mask_ratio: Ratio of patches to mask (default: 0.75)
        patch_size: Size of each image patch (default: 16)
        device: Device to train on (default: cuda if available, else cpu)
    """

    
    model = model.to(device)

    
    
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Configure logging
    log_filename = "log/training_mae.txt"
    logging.basicConfig(
        filename=log_filename,  # Save logs to a file
        filemode="a",  
        format="%(message)s",
        level=logging.INFO,
    )

    logger = logging.getLogger()
    logger.info("----------------------------------training mae started ----------------------------------------")

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

    # Construct the filename dynamically using config values
    model_filename = (
        f"mae_enc_patch_size{config.enc.patch_size}_"
        f"img_size{config.enc.img_size}_in_chs{config.enc.in_chs}_"
        f"emb_dim{config.enc.emb_dim}_depth{config.enc.depth}_num_head{config.enc.num_head}_"
        f"mask_ratio{config.enc.mask_ratio}_"
        f"dec_emb_dim{config.dec.emb_dim}_dec_depth{config.dec.depth}_"
        f"dec_num_head{config.dec.num_head}_dec_mlp_ratio{config.dec.mlp_ratio}.pth"
    )
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(data_loader)

        for i, imgs in enumerate(pbar): 
            
            imgs = imgs.to(device)
           
            # Forward pass
            predicted_img, mask = model(imgs)
           
            loss = torch.mean((predicted_img - imgs) ** 2 * mask / 0.75)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print progress
            if (batch_size + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
                
        # Print epoch results
        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        logger.info(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"saved_models/mae/100k_{model_filename}.pth")   
    
    # Save final model
    torch.save(model.state_dict(), f"saved_models/mae/100k_{model_filename}_final.pth") 

    logger.info("---------------------------Training complete. Model saved.----------------------------------------------------")
    os.makedirs("loss", exist_ok=True)  # Ensure the 'loss' directory exists
    filename_csv = ("loss/" + f"{model_filename}" + ".csv")
    with open(filename_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss"])  # Header
        for epoch, loss in enumerate(losses, start=1):
            writer.writerow([epoch, loss])
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("MAE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"plots/{model_filename}.png")
    plt.close()
    
    return model