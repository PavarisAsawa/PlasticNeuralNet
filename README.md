# Hebbian Network for Adaptive Locomotion

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Training](#training)
4. [Playing](#playing)
5. [Logging](#logging)
6. [License](#license)

## Overview
-

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
git clone https://github.com/PavarisAsawa/PlasticNeuralNet.git
```


- Using a python interpreter that has Isaac Lab installed, install the library

```bash
cd PlasticNeuralNet
python -m pip install -e source/PlasticNeuralNet
```

- Verify that the extension is correctly installed by running the following command:

```bash
python script/ES/train.py --task default --headless --num_envs 1
```


## Training
Train with default configuration (as defined in the config file)
```bash
python script/ES/train.py --task <TASK_NAME>
```
Customize training parameters via CLI (without modifying the config file)
```bash
python script/ES/train.py --task default --num_envs 1024 --ff --headless --wandb
```
Other important parameters for training ES agents are listed in the table
| Parameter | value    |Description                |
| :-------- | :------- |:------------------------- |
| `--task` | `TASK_NAME` | **Required**. Select the task by specifying its name or ID |
| `--num_envs` | **INT** | Set the number of parallel environments or population size.|
|`--ff` , `--hebb` ,`--lstm` | None | Use a other Neural Network model (instead model in configure file).
| `--headless` | None | Run simulation without GUI (useful for remote or server environments).|
| `--wandb` | None | Enable logging of training metrics to Weights & Biases.|


## Playing 
path to collect the model is following 
- logs/es/`TASK_NAME`/`MODEL`/`EXPERIMENT`/ model /`CHECKPOINT`
- models path are in
    - logs/es/`default`/`hebb`/`fixedbody`/model/`model_499.pickle`
```bash
python scripts/ES/play.py --hebb --task default --num_envs 1 --experiment fixedbody --checkpoint model_499.pickle --headless
```

Other important parameters for playing ES agents are listed in the table
| Parameter | value    |Description                |
| :-------- | :------- |:------------------------- |
| `--task` | `TASK_NAME` | **Required**. Select the task by specifying its name or ID |
| `--num_envs` | **INT** | Set the number of parallel environments or population size.|
|`--ff` , `--hebb` ,`--lstm` | None | Select model folder (must same with training agent)|
| `--experiment` | `EXPERIMENT` | Select the experiment folder (default name with date and time).|
| `--checkpoint` | `CHECKPOINT` | Select the model file in folder.|

## Logging
- To log training progress, we use wandb for real-time visualization. Before running any experiments, make sure you’re logged in by executing:
```bash
wandb login
```

- To add custom values to the dashboard, edit the Isaac Lab environment’s **`_reset_idx`** method and include your data in the `self.extras["log"]` dictionary. This ensures that any fields you populate there appear in WandB’s logs.

```python
    def _reset_idx(self, env_ids: torch.Tensor | None):
        ##-------------------------- Other code --------------------------##
        # # Logging Value
        extras = dict()
        extras["lin_vel"] = torch.mean(self.vel_loc[:,0])
        extras["heading"] = torch.mean(self.heading_proj)
        extras["up_right"] = torch.mean(self.up_proj)
        self.extras["log"].update(extras)
```
