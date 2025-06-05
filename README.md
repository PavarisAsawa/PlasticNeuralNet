# Hebbian Network for Adaptive Locomotion

## Content


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
python scripts/rsl_rl/train.py --task=Template-Isaac-Velocity-Rough-Anymal-D-v0
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


| Parameter | value    | Description                |
| :-------- | :------- | :------------------------- |
| `--task` | `TASK_NAME` | **Required**. Select the task by specifying its name or ID |
| `--num_envs` | `N` | Set the number of parallel environments or population size.|
|`--ff` , `--hebb` , `--lstm`| None | Use a other Neural Network model (instead of default model).
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