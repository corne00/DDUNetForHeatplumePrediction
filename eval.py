import yaml
from pathlib import Path
import torch
from torch.nn import MSELoss, L1Loss, HuberLoss
from utils_code.losses import SSIMLoss, LinfLoss, IoULoss, PATLoss
from torchmetrics.regression import MeanAbsolutePercentageError as MAPE
import matplotlib.pyplot as plt
import argparse 

from models import MultiGPU_UNet_with_comm
from dataprocessing import init_data, init_data_single_dataloader
from dataprocessing.data_utils import NormalizeTransform


# Argument parser for experiment path
parser = argparse.ArgumentParser(description="Run heat plume prediction evaluation for given input directory (use --path).")
parser.add_argument("--path", type=str, required=True, help="Base path to the experiment directory")
args = parser.parse_args()

# Read the path from the command line argument
PATH = Path(args.path)

SCENARIOS_PATH = Path("./scenarios.yaml")

# Derive the paths for the settings and model weights and destination
settings_path = PATH / "settings.yaml"
ddunet_path = PATH / "unet.pth"
destination_path = PATH 

# Load the settings
settings = yaml.safe_load(open(settings_path))  
scenarios = yaml.safe_load(open(SCENARIOS_PATH))

# Load scenario from the settings file
SCENARIO = settings["data"]["scenario"]
scenario_settings = scenarios[SCENARIO]

# Set device
devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]

###################################################################################
####################     METRIC COLLECTION FUNCTION      ##########################
###################################################################################

def convert_tensors_to_python(obj):
    """
    Recursively convert PyTorch tensors to Python objects (int, float, list, dict).
    """
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_tensors_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_python(v) for v in obj]
    else:
        return obj

def collect_metrics(PATH_current_data, PATH_current_model, PATH_destination, 
                    dataloaders, model, norm, train_norm, info, output_channels, metric_file_name="", include_pressure=True):
    
    # Define a metrics collection dictionary
    collected_metrics = {
        "model": PATH_current_model.name, 
        "data": PATH_current_data.name, 
    }
    
    # Iterate over the dataloaders
    for case, dataloader in dataloaders.items():
        collected_metrics[case] = {}

        # Collect metrics to be calculated
        metrics:dict = {
            "MSE [phys. unit^2]": MSELoss(), 
            "MAE [phys. unit]": L1Loss(), 
            "Linf [phys. unit]": LinfLoss(), 
            "Huber [phys. unit]": HuberLoss(), 
            "SSIM": SSIMLoss()
        }
        
        # Select some metrics only for temperature predictions
        if output_channels == 1: 
            metrics["MoC [-]"], metrics["PAT0.1 [%]"], metrics["PAT1.0 [%]"] = None, PATLoss(pat_thresholds=[0.1]), PATLoss(pat_thresholds=[1])
        
        # Iterate over the metrics
        for idm, (metric_name, metric) in enumerate(metrics.items()):
            print(f"Calculating {metric_name} for {case}",end=" ", flush=True)
            metrics_values = []

            for idb, batch in enumerate(dataloader):
                assert batch[1].shape[0] == 1, "This code only consistently works for batch size 1"

                # Unpack the batch
                inputs, targets = batch
                
                inputs = model.concatenate_tensors(inputs)


                if include_pressure is False:
                    zero_tensor = torch.zeros((inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]), device=inputs.device, dtype=inputs.dtype)
                    inputs = torch.cat([zero_tensor, inputs], dim=1)

                for tmp_in in inputs:
                    norm.reverse(tmp_in, type="Inputs")
                for tmp_in in inputs:
                    train_norm(tmp_in, type="Inputs")

                inputs_model = inputs if include_pressure else inputs[:, 1:, :, :]
                inputs_model = model._split_concatenated_tensor(inputs_model)
                
                targets = targets.to(devices[0])
                with torch.no_grad():
                    outputs = model(inputs_model)#.detach()

                torch.cuda.empty_cache()

                # Concatenate the inputs and targets (inputs are a list due to the model architecture)
                if metric_name in ["IoU"]: #["SSIM", "IoU"]:
                    values = torch.Tensor([
                        metric(outputs[:,i], targets[:,i]) 
                        for i in range(outputs.shape[1])
                    ])
                else:
                    # Unnormalize inputs and targets
                    for tmp_in in inputs:
                        train_norm.reverse(tmp_in, type="Inputs")
                    for tmp_tar in targets:
                        norm.reverse(tmp_tar, type="Labels")
                    for tmp_out in outputs:
                        train_norm.reverse(tmp_out, type="Labels")

                    # Use same vmin and vmax for all metrics 
                    if idb == 0 and idm == 0:
                        vmin = torch.min(targets)
                        vmax = torch.max(targets)
                        
                        for channel in range(targets.shape[1]):
                            vmin = min(vmin, torch.min(outputs[:,channel]))
                            vmax = max(vmax, torch.max(outputs[:,channel]))

                            plt.imshow(outputs[0,channel].cpu().numpy(), cmap="RdBu", vmin=vmin, vmax=vmax)
                            plt.savefig(Path(PATH_destination) / (case + f"_{channel}" + ".png"))
                            plt.close()

                            plt.imshow(targets[0,channel].cpu().numpy(), cmap="RdBu", vmin=vmin, vmax=vmax) 
                            plt.savefig(Path(PATH_destination) / (case + f"_{channel}" + "_target.png"))
                            plt.close()

                    # Calc metrics per output channel
                    if "PAT" in metric_name:
                        values = torch.mean(torch.Tensor(metric(outputs, targets).squeeze()))
                        # TODO check for more than 1 batch
                    else:
                        values = torch.Tensor([metric(outputs[:,i], targets[:,i]) for i in range(outputs.shape[1])])

                torch.cuda.empty_cache()
                metrics_values.append(values)

            # assert len(dataloader) == 1, "I assumed I always have only one batch - otherwise please rethink this code"
            metrics[metric_name] = torch.mean(torch.stack(metrics_values), dim=0) # average over all batches
            print(f": average = {metrics[metric_name]}",flush=True)

            collected_metrics[case][metric_name] = metrics[metric_name]
    
    # Clean and save metrics
    collected_metrics = convert_tensors_to_python(collected_metrics)
    print("Current model name", PATH_current_model.name,flush=True)
    print("Current data name", PATH_current_data.name,flush=True)
    with open(PATH_destination / f"metrics_paper25_{metric_file_name}.yaml", "w") as file:
        yaml.dump(collected_metrics, file)

###################################################################################
########    TEST THE MODEL ON TRAIN / TEST / VAL DATASET   ########################
###################################################################################

# Check if the training dir correspond  s to the scenarios data dir
assert scenario_settings["dir"] == settings["data"]["dir"], "The training dir in the scenarios.yaml file does not correspond to the data dir in the settings.yaml file"

# Set dataset specific settings
settings["data"]["batch_size_training"] = 1
settings["data"]["batch_size_testing"] = 1
include_pressure = settings["data"].get("include_pressure", True)
info = yaml.safe_load(open(Path(settings["data"]["dir"]) / "info.yaml"))

# Set train norm and data norm (train norm needs to be used always for data normalization for the inputs and outputs of the model!)
train_norm = NormalizeTransform(info)
norm = NormalizeTransform(info)

# Initalize a model given the settings
model = MultiGPU_UNet_with_comm(settings, devices=devices)
print("Loading pretrained model from ", ddunet_path, flush=True)
model.load_weights(load_path=ddunet_path, device=devices[0])
model.eval()

# Load the dataloaders
dataloaders = init_data(
    settings["data"], 
    data_dir=settings["data"]["dir"], 
    num_samples_overfitting=None, 
    max_dataset_size=None
)

# Collect the metrics
collect_metrics(
    PATH_current_data=Path(settings["data"]["dir"]),
    PATH_current_model=ddunet_path,
    PATH_destination=destination_path,
    dataloaders=dataloaders,
    model = model,
    norm = norm,
    train_norm=train_norm,
    info = info,
    output_channels = settings["data"]["n_outputs"],
    metric_file_name="train_test_val",
    include_pressure=include_pressure
)

# Stop code if scenario is step 2
if SCENARIO == "step2":
    print("Stopping code because scenario is step 2 (no LGCNN and scaling data)",flush=True)
    exit()

###################################################################################
#############    TEST THE MODEL ON LGCNN TESTING DATASET   ########################
###################################################################################

# Set dataset specific settings
settings["data"]["dir"] = scenario_settings["test_data_lgcnn"]
settings["data"]["batch_size_training"] = 1
settings["data"]["batch_size_testing"] = 1
info = yaml.safe_load(open(Path(settings["data"]["dir"]) / "info.yaml"))
norm = NormalizeTransform(info)

# Change the model settings to the LGCNN testing dataset
train_size = settings["data"]["patch_size"]
eval_size = info["CellsNumber"][0]
subdom_dist_old = settings['data']["subdomains_dist"]
subdomains_dist_new = (subdom_dist_old[0] * (eval_size // train_size), subdom_dist_old[1] * (eval_size // train_size))
settings['data']["subdomains_dist"] = subdomains_dist_new
settings['data']["patch_size"] = eval_size

# Initalize a model given the settings
model = MultiGPU_UNet_with_comm(settings, devices=devices)
print("Loading pretrained model from ", ddunet_path)
model.load_weights(load_path=ddunet_path, device=devices[0])
model.eval()

# Load the dataloaders
dataloaders = init_data_single_dataloader(
    settings["data"], 
    data_dir=settings["data"]["dir"], 
    dataset_name="lgcnn_test",
)

# Collect the metrics
collect_metrics(
    PATH_current_data=Path(settings["data"]["dir"]),
    PATH_current_model=ddunet_path,
    PATH_destination=destination_path,
    dataloaders=dataloaders,
    model = model,
    norm = norm,
    train_norm=train_norm,
    info = info,
    output_channels = settings["data"]["n_outputs"],
    metric_file_name="lgcnn_test",
    include_pressure=include_pressure
)


###################################################################################
#################    TEST THE MODEL ON SCALING DATASET     ########################
###################################################################################

# Set dataset specific settings
settings["data"]["dir"] = scenario_settings["scaling_dir"]
settings["data"]["batch_size_training"] = 1
settings["data"]["batch_size_testing"] = 1
info = yaml.safe_load(open(Path(settings["data"]["dir"]) / "info.yaml"))
norm = NormalizeTransform(info)

# Change the model settings to the LGCNN testing dataset
train_size = settings["data"]["patch_size"]
eval_size = info["CellsNumber"][0]
if subdom_dist_old[0] == 1 and subdom_dist_old[1] == 1:
    subdomains_dist_new = (1, 1)
    settings['data']["subdomains_dist"] = subdomains_dist_new
else:    
    subdom_dist_old = settings['data']["subdomains_dist"]
    subdomains_dist_new = (subdom_dist_old[0] * (eval_size // train_size), subdom_dist_old[1] * (eval_size // train_size))
    settings['data']["subdomains_dist"] = subdomains_dist_new
print("Old and updated subdomains dist", subdom_dist_old, subdomains_dist_new)
settings['data']["patch_size"] = eval_size

# Initalize a model given the settings
model = MultiGPU_UNet_with_comm(settings, devices=devices)
print("Loading pretrained model from ", ddunet_path)
model.load_weights(load_path=ddunet_path, device=devices[0])
model.eval()

# Load the dataloaders
dataloaders = init_data_single_dataloader(
    settings["data"], 
    data_dir=settings["data"]["dir"], 
    dataset_name="scaling_test",
)

# Collect the metrics
collect_metrics(
    PATH_current_data=Path(settings["data"]["dir"]),
    PATH_current_model=ddunet_path,
    PATH_destination=destination_path,
    dataloaders=dataloaders,
    model = model,
    norm = norm,
    train_norm=train_norm,
    info = info,
    output_channels = settings["data"]["n_outputs"],
    metric_file_name="scaling_test",
    include_pressure=include_pressure
)
