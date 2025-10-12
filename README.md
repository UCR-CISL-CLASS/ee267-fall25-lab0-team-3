# EE267 LAB0: Access Carla and Using Carla

This repository contains the code and resources for EE267 Lab 0, which focuses on setting up the CARLA simulator and getting familiar with its basic functionalities. This README provides a comprehensive guide on how to run the provided scripts and what to expect from each of them.

## 1. Environment Setup ##
Before running any of the scripts, you need to have a working CARLA environment. You have a few options to get this set up:

* Option 1: Local Installation (Recommended for powerful machines): If you have a machine with a performant GPU (>8GB of memory), you can install CARLA locally. We recommend using our pre-built Docker image for a smoother setup. You can find the instructions [here](https://github.com/UCR-CISL/Carla-dockers). 

* Option 2: University Resources: You can also use the BCOE GPU machines or the Nautilus National Research Platform (NRP), though these may have limited availability or occasional performance issues. 

Once you have CARLA running, you'll need to set up a Python environment to interact with it.

* Clone the Repository: Navigate to your CARLA installation folder in your terminal and clone this repository:

```
cd /path/to/your/carla/root
git clone https://github.com/UCR-CISL/ee267-fall25-lab0-team-3.git
```
* Create a Python 3.10 environment: It's recommended to use `conda` for this.
```python
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

**Important Note: Make sure the CARLA simulator is running before executing any of the Python scripts to avoid connection errors.**

This can be done by performing the following:
```
./CarlaUE4.sh
```
This will launch a instance of Carla.
However if you wish to run Carla in the packground you will add the `--RenderOffScreen` argument and it will look like the following
```
./CarlaUE4.sh --RenderOffScreen
```

### Part 1 & 2: Driving the Car (Manual and Automatic Control)

These scripts allow you to control a vehicle in the CARLA environment, either manually or through an autonomous agent. 

**Manual Control**:<br>To drive the car yourself using the keyboard, run the following command:
```
python3 manual_control.py
```
*For those that chose option 2, you will have to add `vglrun` before python to run this script.*

A Pygame window will open, and you can use the arrow keys or WASD to control the vechile. Other controls and functions are listed in the terminal window where you launched this python file. 

**Automatic Control**: <br>
To see an autonomous agent, run this script:
```
python3 automatic_control.py
```
*Again, for those that chose option 2, you will have to add `vglrun` before python to run this script.*

This will spawn a vehicle that navigates the environment on its own.

### Part 3: Generating Traffic ###
This part of the lab demonstrates how to use the CARLA Traffic Manager to populate the simulation with other vehicles and pedestrians, creating a more realistic and dynamic environment.<br>

To generate traffic, you wil run:
```
python3 generate_traffic.py
```
Here 30 vehicles will be spawned into the scene in using the Traffic Manager by taking a random sample of currently available spawn points on the map. The using the `try_spawn_actor()` function to spawn the vechiles then set setting the vehicles `set_autopilot()` method to `True`

### Part 4: Bounding Boxes ###

A key aspect of autonomous driving is object detection. These scripts show how to generate 3D bounding boxes for vehicles and other objects in the simulation and project them onto the 2D camera view. 

**Generate Bounding Boxes for Traffic Lights and Signs**

This script will draw bounding boxes for traffic lights and signs in the camera's view.
```
python3 generate_sign_boxes.py
```
You should see a OpenCV window with red boxes drawn on traffic lights/signs. 

![Image](/images/BoundingBox2.png)

**Generate Bounding Boxes for Vehicles and Pedestrians**

This script will draw bounding boxes for vehicles and pedestrians on the road in the camera's view.

```
python3 generate_vehicle_boxes.py
```
You should see a OpenCV window with blue boxes drawn on vehicles and people on bikes.
![Image](/images/BoundingBox5.png) 

**Generate Bounding Boxes for All Objects**

This script will spawn a vehicle with a camera and draw bounding boxes for all other vehicles and traffic signs in the camera's view.
```
python3 generate_all_boxes.py
```
![Image](/images/BoundingBox8.png) 

**Exporting Bounding Boxes (PASCAL VOC format)**

To save the bounding box information in the PASCAL VOC XML format, which is commonly used for training object detection models, run:
```
python3 generate_VOC_files.py
```
This will create an `output` directory containing the captured images and their corresponding XML annotation files in side of your current path.

### Part 5: Instance Segmentation ###

nstance segmentation provides a more detailed understanding of the scene by assigning a unique ID to each object instance. This is different from semantic segmentation, which only classifies pixels by object type (e.g., "car," "road"). 

**To generate an instance segmentation image, run:**
```
python3 segmentation.py
```
This script will spawn an instance segmentation camera, populate the scene with vehicles, and save the resulting image as `instance_segmentation.png`in your current path directory.

![Image](/images/instance_segmentation(3).png)

