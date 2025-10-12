# EE267 LAB0: Access Carla and Using Carla

This repository contains the code and resources for EE267 Lab 0, which focuses on setting up the CARLA simulator and getting familiar with its basic functionalities. This README provides a comprehensive guide on how to run the provided scripts and what to expect from each of them.

## 1. Environment Setup ##
Before running any of the scripts, you need to have a working CARLA environment. You have a few options to get this set up:

* Option 1: Local Installation (Recommended for powerful machines): If you have a machine with a performant GPU (>8GB of memory), you can install CARLA locally. We recommend using our pre-built Docker image for a smoother setup. You can find the instructions [here](https://github.com/UCR-CISL/Carla-dockers). 

* Option 2: University Resources: You can also use the BCOE GPU machines or the Nautilus National Research Platform (NRP), though these may have limited availability or occasional performance issues. 

Once you have CARLA running, you'll need to set up a Python environment to interact with it.

* Clone the Repository: Navigate to your CARLA installation folder in your terminal and clone this repository:

```python
cd /path/to/your/carla/root
git clone https://github.com/UCR-CISL/ee267-fall25-lab0-team-3.git
```
* Create a Python 3.10 environment: It's recommended to use `conda` for this.
```
conda create -n carla python=3.10
conda activate carla
```
Those that have choosen option 2 and went with the University BCOE machines will skip this step.

* Install the CARLA client library: this can be done using `pip`

```
pip install carla
```
* Install other required libraries:
```
pip install pygame numpy opencv-python pascal-voc-writer
```

## 2. Running the Project Files ##

With the environment set up, you can now run the different Python scripts included in this repository. Each script is designed to demonstrate a specific feature of the CARLA simulator.
