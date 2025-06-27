# Library imports
import json
import pathlib
import numpy as np
import yaml
import gc

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import optuna
from typing import Dict

from models import *
from utils_code.prepare_settings import prepare_settings, init_hyperparams_and_settings
from utils_code.visualization import plot_results
from dataprocessing import init_data
from utils_code.train_utils import *
from utils_code.visualization import *
from utils_code.losses import ThresholdedMAELoss, WeightedMAELoss, CombiRMSE_and_MAELoss, CombiLoss, EnergyLoss, matchLoss
from dataprocessing.dataloaders import DatasetMultipleSubdomains
import argparse

STUDY_DIR = "./results"

def evaluate(unet:MultiGPU_UNet_with_comm, losses:Dict[str,list], dataloaders:Dict[str,DataLoader], save_path: pathlib.Path):
    plot_results(model=unet, savepath=save_path, epoch_number="best", dataloaders=dataloaders)
    
    # Save to a JSON file
    with open(save_path / 'losses.json', 'w') as json_file:
        json.dump(losses, json_file)

def objective(trial):
    # Load and save the arguments from the arge parser and default settings
    hyperparams, settings = init_hyperparams_and_settings(path=pathlib.Path(STUDY_DIR))

    # OPTUNAT: OVERWRITE 
    save_path = pathlib.Path(f"{STUDY_DIR}/{trial.number}")
    save_path.mkdir(parents=True, exist_ok=False)


    # Check if we have half precision
    half_precision = torch.cuda.is_available()
    data_type = torch.float16 if half_precision else torch.float32
    # Half precision scaler
    scaler = GradScaler(enabled=half_precision)

    # Set devices
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    # devices = ["cuda:2"]
    print("Available GPUs:", devices, flush=True)

    # OPTUNAT: OVERWRITE 
    # data_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_5000dp inputs_pki outputs_t/"
    settings["model"]["UNet"]["num_channels"] = settings["data"]["n_inputs"]
    settings["model"]["UNet"]["num_outputs"] = settings["data"]["n_outputs"]

    if settings["data"]["include_pressure"] is False:
        settings["model"]["UNet"]["num_channels"] = settings["model"]["UNet"]["num_channels"] - 1

    try:
        track_loss_functions = {
            "mse": torch.nn.MSELoss(),
            "l1": torch.nn.L1Loss(),
            # "energy":EnergyLoss(data_dir=settings["data"]["dir"], device=devices[0]),
        }

        settings["data"]["batch_size_training"] = trial.suggest_categorical("batch_size", [int(item) for item in hyperparams["batch_size"]])
        print(settings["training"]["num_samples_overfitting"])
        max_dataset_size = (None if settings["training"]["max_dataset_size"] is None else int(settings["training"]["max_dataset_size"]))
        num_samples_overfitting = (None if settings["training"]["num_samples_overfitting"] is None else int(settings["training"]["num_samples_overfitting"]))
        dataloaders = init_data(settings["data"], data_dir=settings["data"]["dir"], num_samples_overfitting=num_samples_overfitting, max_dataset_size=max_dataset_size)

        # Generate trial suggestions
        settings["model"]["kernel_size"] = trial.suggest_categorical("kernel_size", [int(item) for item in hyperparams["kernel_size"]])
        settings["model"]["UNet"]["depth"] = trial.suggest_categorical("depth", [int(item) for item in hyperparams["depth"]])
        settings["model"]["UNet"]["complexity"] = trial.suggest_categorical("complexity", [int(item) for item in hyperparams["complexity"]])
        settings["model"]["UNet"]["num_convs"] = trial.suggest_categorical("num_convs", [int(item) for item in hyperparams["num_convs"]])
        settings["model"]["comm"]["comm"] = trial.suggest_categorical("comm", list(set(bool(x) for x in hyperparams["comm"])))
        settings["model"]["comm"]["exchange_fmaps"] = trial.suggest_categorical("exchange_fmaps", list(set(bool(item) for item in hyperparams["exchange_fmaps"])))
        settings["model"]["comm"]["num_comm_fmaps"] = trial.suggest_categorical("num_comm_fmaps", [int(item) for item in hyperparams["num_comm_fmaps"]])
        settings["training"]["lr"] = trial.suggest_float("lr", float(hyperparams["lr"]["min"]),  float(hyperparams["lr"]["max"]), log=hyperparams["lr"]["log"]) #suggest_categorical("lr", [1e-3, 2e-4, 1e-4, 5e-5, 1e-5])
        settings["training"]["adam_weight_decay"] = trial.suggest_categorical("weight_decay", [float(item) for item in hyperparams["weight_decay"]])
        settings["training"]["train_loss"] = trial.suggest_categorical("loss function", hyperparams["loss_functions"])
            
        # Print trial settings
        print("trial settings", settings)

        # dump settings to save_path
        with open(save_path / 'settings.yaml', 'w') as f:
            yaml.dump(settings, f)

        loss_func = matchLoss(settings["training"]["train_loss"])
        val_loss_func = matchLoss(settings["training"]["val_loss"])
        model = MultiGPU_UNet_with_comm(settings, devices=devices)
        model, data = train_parallel_model(model, dataloaders, settings, devices, save_path, scaler=scaler, data_type=data_type,  half_precision=half_precision, loss_func=loss_func, val_loss_func=val_loss_func, track_loss_functions=track_loss_functions, plot_freq=100) 
        

        loss = np.min(data["val_losses"])
        # Save and calculate losses
        evaluate(model, data, dataloaders, save_path)
                
    except Exception as e:
        print(f"Training failed with exception: {e}", flush=True)
        
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        raise optuna.TrialPruned()

    
    print("Finished!", flush=True)
    return loss

def run():
    # Load and save the arguments from the arge parser and default settings
    settings, save_path = prepare_settings()

    settings["model"]["UNet"]["num_channels"] = settings["data"]["n_inputs"]
    settings["model"]["UNet"]["num_outputs"] = settings["data"]["n_outputs"]
    
    # Check if we have half precision
    half_precision = torch.cuda.is_available()
    data_type = torch.float16 if half_precision else torch.float32
    # Half precision scaler
    scaler = GradScaler(enabled=half_precision)

    # Set devices
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    devices = ["cuda:2"]
    print("Available GPUs:", devices, flush=True)

    dataloaders = init_data(settings["data"], data_dir=settings["data"]["dir"])
    
    track_loss_functions = {
        "mse": torch.nn.MSELoss(),
        "l1": torch.nn.L1Loss(),
        # "energy":EnergyLoss(data_dir=settings["data"]["dir"], device=devices[0]),
    }

    print("final settings", settings)
    # dump settings to save_path
    with open(save_path / 'settings.yaml', 'w') as f:
        yaml.dump(settings, f)

    loss_func = matchLoss(settings["training"]["train_loss"], data_dir=settings["data"]["dir"], device=devices[0])
    val_loss_func = matchLoss(settings["training"]["val_loss"])

    # init model, load if pretrained
    model = MultiGPU_UNet_with_comm(settings, devices=devices)
    if settings["model"]["pretrained"]:
        print("Loading pretrained model from ", save_path/"unet.pth")
        model.load_weights(load_path=save_path/"unet.pth", device=devices[0])
    else:
        model.save_weights(save_path=save_path/"unet.pth")

    model, data = train_parallel_model(model, dataloaders, settings, devices, save_path, scaler=scaler, data_type=data_type,  half_precision=half_precision, loss_func=loss_func, val_loss_func=val_loss_func, track_loss_functions=track_loss_functions) 

    # Save and calculate losses
    evaluate(model, data, dataloaders, save_path)
        
    print("Finished!", flush=True)

if __name__ == "__main__":
    hyperparam_search = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study_dir", 
        type=str, 
        required=True,
        help="Path to the study directory"
    )
    parser.add_argument(
        "--train_single_model", 
        action="store_true", 
        help="Set this flag to train a single model instead of performing hyperparameter search"
    )
    
    args = parser.parse_args()
    STUDY_DIR = args.study_dir
    hyperparam_search = not args.train_single_model
    
    print(f"Running {'hyperparameter search' if hyperparam_search else 'single run'}", flush=True)
    print("Study directory:", STUDY_DIR)    

    study_dir = pathlib.Path(STUDY_DIR)
    study_dir.mkdir(parents=True, exist_ok=True)

    if hyperparam_search:
        study = optuna.create_study(direction="minimize", storage=f"sqlite:///{STUDY_DIR}/hyperparam_opti.db", study_name="search", load_if_exists=True)
        study.optimize(objective, n_trials=30)

        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))

        print("Best trial:")
        print("  Value: ", study.best_trial.value)
        print("  Params: ")
        for key, value in study.best_trial.params.items():
            print("    {}: {}".format(key, value))

        print("Complete trials:")
        for trial in complete_trials:
            print("  Trial {}: {}".format(trial.number, trial.value))

        print("Pruned trials:")
        for trial in pruned_trials:
            print("  Trial {}: {}".format(trial.number, trial.value))

    else:
        run()