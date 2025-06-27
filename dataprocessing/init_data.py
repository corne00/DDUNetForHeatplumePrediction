import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import json

from .dataloaders import DatasetMultipleSubdomains
from torch.utils.data import DataLoader

def init_data(data_settings, data_dir, num_samples_overfitting=None, 
              crop_size = (None, None), max_dataset_size=None, scaling_dataset=None):
    data_dir = pathlib.Path(data_dir)
    inputs_dir = data_dir / "Inputs"
    labels_dir = data_dir / "Labels"
    
    if num_samples_overfitting is not None:
        inputs_dir_test = inputs_dir
        labels_dir_test = labels_dir
    else:
        test_dir = pathlib.Path(data_dir).parent / f"{pathlib.Path(data_dir).name} TEST"
        inputs_dir_test = test_dir / "Inputs"
        labels_dir_test = test_dir / "Labels"

    include_pressure = data_settings.get('include_pressure', True)
    if include_pressure:
        print("Pressure included in training")
    else:
        print("Pressure not included in training")
        
    np.random.seed(0)
    image_names = os.listdir(inputs_dir)
    image_names.sort()
    np.random.shuffle(image_names)
    
    if num_samples_overfitting is not None:
        image_names_train = image_names[:num_samples_overfitting]
        image_names_val = image_names[:num_samples_overfitting]
        image_names_test = image_names[:num_samples_overfitting]
    else:
        split=(np.array([0.8,0.2,0.]) * len(image_names)).astype(int)
        
        image_names_train = image_names[:split[0]]
        image_names_val = image_names[split[0]:split[0]+split[1]]
        image_names_test = np.sort(os.listdir(inputs_dir_test))
        print("Image labels test:", image_names_test)

    if max_dataset_size is not None:
        image_names_train = image_names_train[:max_dataset_size]

    
    train_dataset = DatasetMultipleSubdomains(image_labels=image_names_train, image_dir=inputs_dir, mask_dir=labels_dir, transform=None,
                                        target_transform=None, data_augmentation=None, subdomains_dist=data_settings["subdomains_dist"], patch_size=data_settings["patch_size"], #2560, 
                                        crop_size=crop_size, max_images = None, include_pressure=include_pressure)

    val_dataset = DatasetMultipleSubdomains(image_labels=image_names_val, image_dir=inputs_dir, mask_dir=labels_dir, transform=None,
                                        target_transform=None, data_augmentation=None, subdomains_dist=data_settings["subdomains_dist"], patch_size=data_settings["patch_size"], #2560, 
                                        crop_size=crop_size, max_images = None, include_pressure=include_pressure)

    test_dataset = DatasetMultipleSubdomains(image_labels=image_names_test, image_dir=inputs_dir_test, mask_dir=labels_dir_test, transform=None,
                                        target_transform=None, data_augmentation=None, subdomains_dist=data_settings["subdomains_dist"], patch_size=data_settings["patch_size"], #2560, 
                                        crop_size=crop_size, max_images = None, include_pressure=include_pressure)

    if scaling_dataset is not None: 
        scaling_datadir = pathlib.Path(scaling_dataset)
        scaling_inputs_dir = scaling_datadir / "Inputs"
        scaling_labels_dir = scaling_datadir / "Labels"
        scaling_image_names = os.listdir(scaling_inputs_dir)
        scaling_image_names.sort()

    print("Length of train, test and validation dataset:", len(train_dataset), len(test_dataset), len(val_dataset))

    # Define dataloaders
    dataloader_train = DataLoader(train_dataset, batch_size=data_settings["batch_size_training"], shuffle=True) 
    dataloader_val = DataLoader(val_dataset, batch_size=data_settings["batch_size_testing"], shuffle=False)
    dataloader_test = DataLoader(test_dataset, batch_size=data_settings["batch_size_testing"], shuffle=False)

    return {"train": dataloader_train, "val": dataloader_val, "test": dataloader_test}



def init_data_single_dataloader(data_settings, data_dir, crop_size = (None, None), dataset_name="dataset"):
    data_dir = pathlib.Path(data_dir)
    inputs_dir = data_dir / "Inputs"
    labels_dir = data_dir / "Labels"

    image_names = os.listdir(inputs_dir)
    image_names.sort()
    
    include_pressure = data_settings.get('include_pressure', True)

    dataset = DatasetMultipleSubdomains(image_labels=image_names, image_dir=inputs_dir, mask_dir=labels_dir, transform=None,
                                        target_transform=None, data_augmentation=None, subdomains_dist=data_settings["subdomains_dist"], patch_size=data_settings["patch_size"], #2560, 
                                        crop_size=crop_size, max_images = None, include_pressure=include_pressure)

    # Define dataloaders
    dataloader_train = DataLoader(dataset, batch_size=data_settings["batch_size_training"], shuffle=True) 

    return {dataset_name: dataloader_train}
