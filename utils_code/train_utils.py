import os

import torch
import contextlib
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

from models.ddu_net import MultiGPU_UNet_with_comm
from .visualization import plot_results
from utils_code.losses import EnergyLoss, CombiLoss

def compute_validation_loss(model, loss_func, dataloader, device, data_type, half_precision, verbose=False):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        with (torch.autocast(device_type='cuda', dtype=data_type) if half_precision else contextlib.nullcontext()):
            for inputs, labels in tqdm(dataloader, disable=(not verbose)):

                labels = labels.to(device)
                predictions = model(inputs)
                loss = loss_with_energy_option(loss_func, inputs, labels, predictions)
                
                total_loss += loss.item()
                num_batches += 1
        
    average_loss = total_loss / num_batches
    return average_loss

# Train function
def train_parallel_model(model:MultiGPU_UNet_with_comm, dataloaders, settings, devices, save_path, scaler, data_type, half_precision, loss_func, val_loss_func, verbose=False, plot_freq:int=10, track_loss_functions=None):
    
    writer = SummaryWriter(save_path)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["training"]["lr"], weight_decay=settings["training"]["adam_weight_decay"])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.5)

    summary_losses = {}

    # Wrap your training loop with tqdm
    start_time = time.time()
    validation_losses = []
    best_val_loss = float('inf')

    # Iterate over the epochs
    epochs = tqdm(range(settings["training"]["num_epochs"]), desc="epochs", disable=False)
    early_stop_patience = 80 # TODO add this to training arguments
    epochs_since_improvement = 0
    for epoch in epochs:
        model.train()
        epoch_losses = 0.0  # Initialize losses for the epoch
        
        for inputs, labels in tqdm(dataloaders["train"], disable=(not verbose)):
            with (torch.autocast(device_type='cuda', dtype=data_type) if half_precision else contextlib.nullcontext()):
                
                labels = labels.to(devices[0])

                ## Forward propagation:
                predictions = model(inputs)
                loss_value = loss_with_energy_option(loss_func, inputs, labels, predictions)
                
            # Check for nan loss and break training if it occurs
            if torch.isnan(loss_value):
                print(f"NaN detected in loss at epoch {epoch}, stopping training")
                return model, summary_losses
            
            ## Backward propagation
            scaler.scale(loss_value).backward()
            epoch_losses += loss_value.item()  # Add loss to epoch losses

            # Weight upgrade of the encoders
            with torch.no_grad():
                for i in range(1, len(model.encoders)):
                    for param1, param2 in zip(model.encoders[0].parameters(), model.encoders[i].parameters()):
                        if param1.grad is not None:
                            param1.grad += param2.grad.to(devices[0])
                            param2.grad = None

            # Weight update of the decoders
            with torch.no_grad():
                for i in range(1, len(model.decoders)):
                    for param1, param2 in zip(model.decoders[0].parameters(), model.decoders[i].parameters()):
                        param1.grad += param2.grad.to(devices[0])
                        param2.grad = None
            
            # Set optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Syncrhonize weights to the encoder clones
            for i in range(1, len(model.encoders)):
                model.encoders[i].load_state_dict(model.encoders[0].state_dict())
                
            # Syncrhonize weights to the decoder clones
            for i in range(1, len(model.decoders)):
                model.decoders[i].load_state_dict(model.decoders[0].state_dict())
    
        # Compute and print validation loss
        val_loss = compute_validation_loss(model, val_loss_func, dataloaders["val"], devices[0], data_type=data_type, half_precision=half_precision, verbose=False)
        validation_losses.append(val_loss)
        # epoch_losses = epoch_losses.item()
        epochs.set_postfix_str(f"train loss: {(epoch_losses)/len(dataloaders['train']):.2e}, val loss: {val_loss:.2e}, lr: {optimizer.param_groups[0]['lr']:.1e}")
        
        # Check for improvement and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(save_path=os.path.join(save_path, "unet.pth"))
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}. No improvement for {early_stop_patience} epochs.")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                break

        writer.add_scalar("train_loss", epoch_losses/len(dataloaders["train"]), epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        for name, loss_func in track_loss_functions.items():
            try:
                tmp_val_loss = compute_validation_loss(model, loss_func, dataloaders["val"], devices[0], data_type=data_type, half_precision=half_precision, verbose=False)
                writer.add_scalar(f"val-{name}", tmp_val_loss, epoch)
                summary_losses[f"val-{name}"] = tmp_val_loss
            except: pass
            
        scheduler.step(val_loss)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if (epoch + 1) % plot_freq == 0:
            plot_results(model=model, savepath=save_path, epoch_number=epoch, dataloaders=dataloaders)
            
    summary_losses["val_losses"] = validation_losses
    print(f"Training the model {'with' if model.comm else 'without'} communication network took: {time.time() - start_time:.2f} seconds.", flush=True)
    
    # Load the best weights
    model.load_weights(load_path=os.path.join(save_path, "unet.pth"), device=devices[0])
    
    return model, summary_losses

def loss_with_energy_option(loss_func:torch.nn.Module, inputs, labels, predictions):
    if isinstance(loss_func, EnergyLoss):
        return loss_func(predictions, inputs)
    if isinstance(loss_func, CombiLoss):
        return loss_func(predictions, labels, inputs)
    else:
        return loss_func(predictions, labels)
