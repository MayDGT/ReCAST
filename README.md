# ReCAST
This is the repository for our paper "Revealing Safety Violations of Autonomous Driving Systems with Law-Compliant NPC Vehicles".

## Dependencies
* Ubuntu 22.04 LTS
* Apollo 7.0
* Docker-CE version 19.03 and above

## Usage
To reproduce the experimental results, users should follow the steps below:
### Preparation
* Clone this project and create a virtual environment: <br />
  ```conda create -n recast python=3.8```
* Install the required packages: <br />
  ```pip install -r npc_train/requirements.txt```
* Download Apollo 7.0 from [https://doi.org/10.5281/zenodo.7622089](https://doi.org/10.5281/zenodo.7622089)
* Create some necessary directories at the root directory of Apollo: <br />
  ```mkdir data data/log data/bag data/core```
* Start up Apollo container at the root directory of Apollo: <br />
  ```./docker/scripts/dev_start.sh -l```
* Enter the container and build Apollo: <br />
  ```./docker/scripts/dev_into.sh``` <br />
  ```./apollo.sh build```

### Training the NPC Controller
* Set the training configuration parameters (e.g., number of surrounding ADS vehicles) in the ```npc_train/config.yaml``` file
* Train a style-specific controller on a designated road type. For example, to train an aggressive controller on a straight road: <br />
  ```python npc_train/train/aggressive_straight.py ```
