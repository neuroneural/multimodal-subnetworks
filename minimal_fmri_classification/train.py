# train.py
import os
import sys
import yaml
import random
import time
import datetime
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --- Project Imports ---
# Assuming these files are in the same directory or accessible via PYTHONPATH
try:
    from data_utils import prepare_dataloaders
    from model import build_model
    from wandb_utils import WandbLogger
except ImportError as e:
    print(f"Error importing project components: {e}")
    print("Ensure data_utils.py, model.py, and wandb_utils.py are accessible.")
    sys.exit(1)

# --- Constants ---
CONFIG_PATH = 'config.yaml'
BEST_MODEL_FILENAME = 'model_best.pth'
CHECKPOINT_FILENAME_FORMAT = 'model_epoch_{epoch}.pth'

# --- Helper Function ---
def extract_id_from_filename(filename):
    """Extracts the leading numerical ID from filenames like '000300655084_fALFF.nii'."""
    match = re.match(r'^(\d+)_', filename)
    return match.group(1) if match else None

# --- Setup Function ---
def setup(config_path):
    """Loads config, sets seeds, determines device, creates output dir."""
    # Load Configuration
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
    except Exception as e:
        print(f"Error loading or parsing config file: {e}")
        sys.exit(1)

    # Setup Output Directory
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Set Random Seeds
    seed = config['dataloader'].get('random_seed', 42) # Use default if not specified
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")

    # Setup Device
    cfg_device = config.get('device', 'auto') # Use default 'auto'
    if cfg_device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg_device)
    print(f"Using device: {device}")

    return config, output_dir, device

# --- Data Loading Function ---
def load_data(config, wandb_logger):
    """Prepares and returns dataloaders."""
    print("\nPreparing dataloaders...")
    # We need the ID extractor, pass it if necessary or ensure prepare_dataloaders uses it
    # Assuming prepare_dataloaders handles the ID extraction based on config
    trainloader, testloader = prepare_dataloaders(config, wandb_logger=wandb_logger)

    if trainloader is None:
         print("Failed to create dataloaders. Exiting.")
         if wandb_logger and wandb_logger.is_active:
             wandb_logger.finish(exit_code=1)
         sys.exit(1)

    print("Dataloaders ready.")
    return trainloader, testloader

# --- Model and Training Components Function ---
def build_components(config, device):
    """Builds the model, criterion, and optimizer."""
    print("\nBuilding model and training components...")
    # Build Model
    model = build_model(config)
    model.to(device)

    # Define Loss Function
    # Assuming BCEWithLogitsLoss based on original code. Make configurable if needed.
    criterion = nn.BCEWithLogitsLoss()

    # Define Optimizer
    lr = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0) # Default to 0 if not specified
    # Assuming Adam based on original code. Make configurable if needed.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"Model: {config['model']['variant']}")
    print(f"Criterion: {type(criterion).__name__}")
    print(f"Optimizer: {type(optimizer).__name__} (LR={lr}, WD={weight_decay})")

    return model, criterion, optimizer

# --- Train Epoch Function ---
def train_epoch(model, dataloader, criterion, optimizer, device, wandb_logger, epoch_num, global_step, print_freq):
    """Runs a single training epoch."""
    model.train()
    epoch_loss_sum = 0.0
    interval_loss_sum = 0.0
    interval_batches = 0
    total_samples = 0
    num_batches = len(dataloader)

    epoch_start_time = time.time()

    for i, batch in enumerate(dataloader):
        # Basic check for empty batch from collate_fn
        if not batch or (isinstance(batch, (list, tuple)) and (not batch[0].nelement())):
            print(f"Warning: Skipping empty batch {i+1}/{num_batches}")
            continue

        images, labels = batch
        batch_size_actual = images.size(0)
        global_step += 1
        total_samples += batch_size_actual

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(1), labels.float()) # Ensure labels are float for BCE variants
        loss.backward()
        optimizer.step()

        loss_item = loss.item()
        epoch_loss_sum += loss_item * batch_size_actual # Accumulate weighted loss
        interval_loss_sum += loss_item
        interval_batches += 1

        # Log interval loss
        if (i + 1) % print_freq == 0 or (i + 1) == num_batches:
             avg_interval_loss = interval_loss_sum / interval_batches if interval_batches > 0 else 0
             print(f'  Epoch [{epoch_num}], Batch [{i+1}/{num_batches}], Train Loss Interval: {avg_interval_loss:.4f}')
             if wandb_logger and wandb_logger.is_active:
                  wandb_logger.log_batch({"train/loss_interval": avg_interval_loss}, step=global_step)
             # Reset interval counters
             interval_loss_sum = 0.0
             interval_batches = 0

    epoch_duration = time.time() - epoch_start_time
    epoch_avg_loss = epoch_loss_sum / total_samples if total_samples > 0 else 0

    return epoch_avg_loss, epoch_duration, global_step

# --- Evaluate Function ---
def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the given dataloader."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    num_batches = len(dataloader)

    print("Evaluating...")
    with torch.no_grad():
         for i, batch in enumerate(dataloader):
            if not batch or (isinstance(batch, (list, tuple)) and (not batch[0].nelement())):
                print(f"Warning: Skipping empty batch {i+1}/{num_batches} during evaluation")
                continue

            images, labels = batch
            batch_size_actual = images.size(0)
            total_samples += batch_size_actual

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs.squeeze(1), labels.float()) # Ensure labels are float

            total_loss += loss.item() * batch_size_actual # Accumulate weighted loss

            # Calculate accuracy (assuming binary classification with 0.5 threshold)
            predicted = (torch.sigmoid(outputs.squeeze(1)) > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = (100.0 * correct_predictions / total_samples) if total_samples > 0 else 0

    return avg_loss, accuracy

# --- Save Checkpoint Function ---
def save_checkpoint(model, optimizer, epoch, metrics, is_best, output_dir):
    """Saves model checkpoint."""
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **metrics # Add metrics like loss, accuracy etc.
    }
    # Save regular checkpoint
    checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILENAME_FORMAT.format(epoch=epoch))
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Save best model checkpoint if applicable
    if is_best:
        best_checkpoint_path = os.path.join(output_dir, BEST_MODEL_FILENAME)
        torch.save(checkpoint_data, best_checkpoint_path)
        print(f"Best model saved to {best_checkpoint_path}")


# --- Main Training Loop Function ---
def run_training_loop(config, model, trainloader, testloader, criterion, optimizer, device, output_dir, wandb_logger):
    """Runs the main training and evaluation loop."""
    num_epochs = config['training']['num_epochs']
    print_freq = config['logging'].get('print_freq', 50) # Default print frequency
    save_freq = config['saving'].get('save_freq', 1) # Default save frequency
    watch_log_type = config.get('wandb', {}).get('watch_log_type', 'all') # For wandb.watch

    if wandb_logger and wandb_logger.is_active:
        print("Watching model with WandB...")
        wandb_logger.watch(model, log_type=watch_log_type)

    print(f"\n--- Starting Training for {num_epochs} Epochs ---")
    start_training_time = time.time()
    best_test_accuracy = -1.0
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train one epoch
        avg_train_loss, epoch_duration, global_step = train_epoch(
            model, trainloader, criterion, optimizer, device, wandb_logger, epoch, global_step, print_freq
        )
        print(f"Epoch {epoch} Train: Avg Loss: {avg_train_loss:.4f}, Duration: {epoch_duration:.2f}s")

        # Evaluate on test set if available
        current_test_accuracy = -1.0
        avg_test_loss = -1.0
        is_best = False
        if testloader:
            avg_test_loss, current_test_accuracy = evaluate(model, testloader, criterion, device)
            print(f"Epoch {epoch} Test : Avg Loss: {avg_test_loss:.4f}, Accuracy: {current_test_accuracy:.2f}%")

            # Check if this is the best model so far
            if current_test_accuracy > best_test_accuracy:
                print(f"  New best test accuracy: {current_test_accuracy:.2f}% (Previous best: {best_test_accuracy:.2f}%)")
                best_test_accuracy = current_test_accuracy
                is_best = True
                if wandb_logger and wandb_logger.is_active:
                    wandb_logger.update_summary("best_test_accuracy", best_test_accuracy)
        else:
            print(f"Epoch {epoch} Test : Skipped (No test data).")

        # Log epoch metrics to WandB
        if wandb_logger and wandb_logger.is_active:
            log_dict_epoch = {
                "epoch": epoch,
                "train/loss_epoch": avg_train_loss,
                "train/epoch_duration_sec": epoch_duration,
                "train/learning_rate": optimizer.param_groups[0]['lr'], # Log current LR
                "test/loss_epoch": avg_test_loss if testloader else 0,
                "test/accuracy": current_test_accuracy if testloader else 0
            }
            wandb_logger.log_epoch(log_dict_epoch)

        # Save checkpoint periodically or if it's the best
        if (epoch % save_freq == 0) or (epoch == num_epochs) or is_best:
             metrics_to_save = {
                  'train_loss': avg_train_loss,
                  'test_loss': avg_test_loss,
                  'test_accuracy': current_test_accuracy
             }
             save_checkpoint(model, optimizer, epoch, metrics_to_save, is_best, output_dir)

    # End of Training
    total_training_time = time.time() - start_training_time
    print("\n--- Training Finished ---")
    print(f"Total Training Time: {datetime.timedelta(seconds=int(total_training_time))}")
    print(f"Best Test Accuracy achieved: {best_test_accuracy:.2f}%")


# --- Main Execution Guard ---
if __name__ == "__main__":
    print("Main training script started.")

    # Setup environment, load config, set seeds, device
    config, output_dir, device = setup(CONFIG_PATH)

    # Initialize WandB Logger
    wandb_logger = WandbLogger(config) # Handles wandb.init()

    # Load Data
    trainloader, testloader = load_data(config, wandb_logger)

    # Build Model, Criterion, Optimizer
    model, criterion, optimizer = build_components(config, device)

    # Run Training Loop
    try:
        run_training_loop(
            config=config,
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            output_dir=output_dir,
            wandb_logger=wandb_logger
        )
    except Exception as e:
        print(f"\n--- An error occurred during training ---")
        print(e)
        exit_code = 1
    else:
        exit_code = 0 # Success
    finally:
        # Finish WandB Run
        if wandb_logger and wandb_logger.is_active:
            print("Finishing WandB run...")
            wandb_logger.finish(exit_code=exit_code)
        print("Script finished.")
        sys.exit(exit_code)