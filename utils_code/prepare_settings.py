import argparse
import datetime
import json
import yaml
import pathlib
from typing import Union

def prepare_settings():
    destination_dir = pathlib.Path("results")
    now = datetime.datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M_%S_") + f"{now.microsecond // 1000:03d}"

    parser = argparse.ArgumentParser(description='Train a U-Net model with optional communication network.')
    # Set the save path to include a timestamp
    parser.add_argument('--save_path', type=str, default=rf"{current_time}",
                        help="Path for saving results (default: ./results/<timestamp>).")

    args = parser.parse_args()
    save_path = destination_dir / args.save_path

    if save_path.exists():
        print(f"Path {save_path} already exists. Taking settings from there.")
        assert (save_path / "settings.yaml").exists(), f"No settings.yaml found in {save_path}"
        settings = yaml.safe_load(open(save_path / "settings.yaml"))

    else:
        settings = yaml.safe_load(open("default_settings.yaml"))
        # default_settings["save_path"] = STUDY_DIR + "/energy_loss_finetune"
        save_path.mkdir(parents=True, exist_ok=True)

    # automatically read n_inputs, n_outputs, data-dir from choice of scenario
    get_scenario_info(settings)
    
    if settings["model"]["padding"] is None:
        settings["model"]["padding"] = settings["model"]["kernel_size"] // 2

    return settings, save_path

def init_hyperparams_and_settings(path):
    if (path/"hyperparam_search_options.yaml").exists():
        load_hyperparam = path/"hyperparam_search_options.yaml"
        load_settings = path/"settings.yaml"
        to_dump = False

    else:
        load_hyperparam = "default_hyperparam_searchspace.yaml"
        load_settings = "default_settings.yaml"
        to_dump = True
    
    hyperparams = yaml.safe_load(open(load_hyperparam))
    settings = yaml.safe_load(open(load_settings))

    # automatically read n_inputs, n_outputs, data-dir from choice of scenario
    get_scenario_info(settings)

    if to_dump:
        with open(path/"hyperparam_search_options.yaml", 'w') as f:
            yaml.dump(hyperparams, f)
        with open(path/"settings.yaml", 'w') as f:
            yaml.dump(settings, f)
    return hyperparams, settings

def get_scenario_info(settings):
    assert settings["data"]["scenario"] in ["step1", "step2", "step3", "full"], f"scenario {settings['data']['scenario']} does not exist in scenarios.yaml"
    scenarios = yaml.safe_load(open("scenarios.yaml"))
    scenario = scenarios[settings["data"]["scenario"]]
    settings["data"]["dir"] = scenario["dir"]
    settings["data"]["n_inputs"] = scenario["n_inputs"]
    settings["data"]["n_outputs"] = scenario["n_outputs"]


def save_args_to_json(args, filename="args.json"):
    # Convert the argparse.Namespace object to a dictionary
    args_dict = vars(args)
    
    # Save the dictionary to a JSON file
    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)

    print(f"Arguments saved to {filename}")

if __name__ == "__main__":
    args = prepare_settings()
    print(args)
