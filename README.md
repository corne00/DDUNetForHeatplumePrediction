# Model Training and Hyperparameter Search
This repository contains code for hyperparameter optimization and training of a machine learning model using PyTorch. It also includes functionality for evaluating the trained model and reproducing results from the associated paper "Few-Shot Learning by Explicit Physics Integration: An Application to Groundwater Heat Transport".

The code is largely based on the following two GitHub repositories:
- https://github.com/JuliaPelzer/Heat-Plume-Prediction/
- https://github.com/corne00/DDU-Net/


## Table of Contents:

1. Some important files
2. Installation
2. Usage
3. Results
4. Pre-trained Models
5. Acknowledgments


## Some important files
- **`train.py`**: Contains code for training a single model or conducting a hyperparameter search.
- **`eval.py`**: Computes metrics for the train, test, validation, and scaling datasets using a given model.
- **`scenarios.yaml`**: Specifies the dataset paths for various modeling scenarios (step 1, step 2, step 3, and full).
- **`default_hyperparam_searchspace.yaml`**: Defines the default search ranges for hyperparameter optimization. Used when no specific hyperparameter search options are provided in the hyperparameter search folder.
- **`default_settings.yaml`**: Specifies the default model initialization settings, applied when the directory lacks a corresponding settings YAML file.

## Installation:

Install the required dependencies using the provided requirements.txt file:
    
    pip install -r requirements.txt

## Usage:

### Hyperparameter Search:

Hyperparameter optimization is an essential step in the development and fine-tuning of machine learning models. This repository leverages **Optuna**, a state-of-the-art framework for automated hyperparameter optimization, to efficiently explore the hyperparameter space and identify optimal settings for training the models.

1. **Define the Hyperparameter Search Space:**
   - The default search space for hyperparameters is specified in `default_hyperparam_searchspace.yaml`. 
   - This file includes ranges, distributions, and constraints for various parameters like learning rate, batch size, and architecture-specific configurations.

2. **Prepare the Working Directory:**
   - Create a directory for storing the results of the hyperparameter search:
     ```bash
     mkdir ./results/hyper_param_search/
     ```
   - Copy the required configuration files into this directory:
     ```bash
     cp default_hyperparam_searchspace.yaml ./results/hyper_param_search/hyperparam_search_options.yaml
     cp default_settings.yaml ./results/hyper_param_search/settings.yaml
     ```

3. **Adjust Search Space and Settings:**
   - Edit `hyperparam_search_options.yaml` and `settings.yaml` as needed to reflect the experiment setup. Note that settings in `hyperparam_search_options.yaml` will overwrite the settings in `settings.yaml`.

4. **Run the Hyperparameter Search:**
   - Execute the following command to initiate the search:
     ```bash
     python train.py --study_dir ./results/hyper_param_search
     ```
   - If all values in the search space are fixed, this process effectively evaluates the model with different random initializations.

5. **Evaluate Search Results:**
   - For each model in the results directory, evaluate its performance using:
     ```bash
     for /d %i in (.\results\best_full_scenario\*) do python eval.py --path "%i"
     ```
   - Metrics for each model (e.g., MSE, MAE, SSIM) are saved in their respective subdirectories.

### Train single model
To train a model with specified settings (so, no hyperparameter search), use the flag `--train_single_model`, and specify the path (`PATH`) containing the `settings.yaml` file:

    python train.py --study_dir PATH --train_single_model


## Results:

### Scenario full (pki $ \rightarrow $ T)
Below is a summary of the training metrics (mean plusminus standard deviation) obtained for the best network found in our hyperparameter search for the fully datadriven scenario (i.e., predict temperature field T from the inputs pki directly). The table corresponds to Table 10 in the paper:

| Model              | Data Case      | Huber                  | $L_\infty$         | MAE                  | MSE                  | PAT                  | SSIM            |
|--------------------|----------------|------------------------|--------------------|----------------------|----------------------|----------------------|-----------------|
| UNet $_{101dp} $   | rK101 train    |  $0.0011 \pm 0.0002 $  |  $4.37 \pm 0.20 $  |  $0.0182 \pm 0.0020 $|  $0.0023 \pm 0.0004 $|  $2.45 \pm 0.27 $    |  $0.995 \pm 0.002 $ |
|                    | val            |  $0.0055 \pm 0.0003 $  |  $4.36 \pm 0.16 $  |  $0.0454 \pm 0.0019 $|  $0.0114 \pm 0.0006 $|  $12.96 \pm 0.55 $   |  $0.984 \pm 0.002 $ |
|                    | test           |  $0.0052 \pm 0.0003 $  |  $4.35 \pm 0.23 $  |  $0.0441 \pm 0.0019 $|  $0.0110 \pm 0.0006 $|  $12.49 \pm 0.48 $   |  $0.985 \pm 0.002 $ |
|                    | LGCNN-test     |  $0.0049 \pm 0.0002 $  |  $4.30 \pm 0.16 $  |  $0.0470 \pm 0.0010 $|  $0.0102 \pm 0.0004 $|  $13.51 \pm 0.59 $   |  $0.983 \pm 0.002 $ |
|                    | scaling        |  $0.0017 \pm 0.0001 $  |  $4.48 \pm 0.15 $  |  $0.0208 \pm 0.0014 $|  $0.0035 \pm 0.0002 $|  $4.38 \pm 0.17 $    |  $0.995 \pm 0.001 $ |
|  $2 \times 2 $-DDUNet $_{101dp} $ | rK101 train |  $0.0014 \pm 0.0003 $ |  $4.11 \pm 0.25 $ |  $0.0203 \pm 0.0026 $ |  $0.0030 \pm 0.0007 $ |  $3.24 \pm 0.68 $ |  $0.995 \pm 0.001 $ |
|                    | val            |  $0.0079 \pm 0.0002 $  |  $4.20 \pm 0.25 $  |  $0.0564 \pm 0.0008 $|  $0.0165 \pm 0.0005 $|  $17.32 \pm 0.22 $   |  $0.981 \pm 0.002 $ |
|                    | test           |  $0.0075 \pm 0.0001 $  |  $4.05 \pm 0.22 $  |  $0.0552 \pm 0.0008 $|  $0.0158 \pm 0.0002 $|  $16.94 \pm 0.36 $   |  $0.982 \pm 0.001 $ |
|                    | LGCNN-test     |  $0.0057 \pm 0.0003 $  |  $4.00 \pm 0.20 $  |  $0.0526 \pm 0.0015 $|  $0.0117 \pm 0.0006 $|  $16.44 \pm 0.63 $   |  $0.981 \pm 0.002 $ |
|                    | scaling        |  $0.0025 \pm 0.0001 $  |  $4.04 \pm 0.20 $  |  $0.0251 \pm 0.0007 $|  $0.0051 \pm 0.0002 $|  $6.39 \pm 0.17 $    |  $0.994 \pm 0.001 $ |

To reproduce these results:

1. Create a folder `./results/best_full_scenario`

        mkdir ./results/best_full_scenario

2. Copy the files `default_hyperparam_searchspace.yaml` and `default_settings.yaml` to this folder

        cp default_hyperparam_searchspace.yaml ./results/best_full_scenario/hyperparam_search_options.yaml
        cp default_settings.yaml ./results/best_full_scenario/settings.yaml

3. Adjust the values in the hyperparam search space and settings folders to match the settings mentioned in Table 7 of the paper (the optimal hyperparameter values). Note that the UNet corresponds to setting the number of subdomains to $1 \times 1$, and the DDUNet $_{2\times 2}$ to setting the number of subdomains to $2 \times 2$. 

4. Run the hyperparameter search (note that if all values in the search are fixed to one, this just leads to repeatedly training the same network with different random initializations), using:

        python train.py --study_dir ./results/best_full_scenario.yaml

5. Evaluate the results for all models in the directory:

        for /d %i in (.\results\best_full\*) do python eval.py --path "%i"

    Then, the resulting metrics will be saved per model in the corresponding subdirectory, and these metrics can be used to compute mean and standard deviation.

### Scenario 1 and scenario 3
The results for scenario 1 (velocity field prediction) and scenario 3 (temperature field prediction based on pkixy) can be generated in a similar way as for the full scenario. Make sure to modify the values in the hyperparameter search space according to the values listed in Table 6 (Overview of used hyperparameters).

## Loading and evaluating pre-trained models:
Pre-trained models (`*.pth-files`) and their corresponding settings (`settings.yaml`) are also provided with the supplementary material. By downloading these and saving both model and the corresponding settings in a folder `PATH`, the `eval.py` script can be used to evaluate pre-trained models on your data or the provided test set, using:

    python eval.py --path PATH


## Acknowledgments:
This repository heavily relies on the following tools and libraries:

- PyTorch
- Optuna
