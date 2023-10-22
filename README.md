# Efficient Reinforcement Learning for Jumping Monopods

Riccardo Bussola, Michele Focchi,  Andrea Del Prete, Daniele Fontanelli, Luigi Palopoli

Corresponding author's email: Riccardo Bussola

This repository is a reduced version of [Locosim](https://github.com/mfocchi/locosim) ([preprint](https://arxiv.org/abs/2305.02107)) and it is intended for reproducing simulations and experiments
presented in the manuscript  https://arxiv.org/abs/2309.07038



In this work we consider the complex problem of making a monopod perform an omni-directional jump on uneven terrain. We guide the learning process within an RL framework by injecting physical knowledge. This expedient brings to widespread benefits, such as a drastic reduction of the learning
time, and the ability to learn and compensate for possible errors in the low-level controller executing the motion. 

Check out our Youtube [video](https://www.youtube.com/watch?v=ARhoYwIrkU0).


## Installing Locosim

Locosim is composed by a **roscontrol** node called **ros_impedance_controller** (written in C++) that interfaces the 
Python ros node (where the controller is written) to a Gazebo simulator.

### SOFTWARE VERSIONS:

Locosim is compatible with Ubuntu 18/20. The installation instructions have been generalized accordingly. 
You need replace few strings with the appropriate values according to your operating systems as follows:

| Ubuntu 18:                   | **Ubuntu 20**:               |
| ---------------------------- | ---------------------------- |
| PYTHON_VERSION = 3.5         | PYTHON_VERSION = 3.8         |
| ROBOTPKG_PYTHON_VERSION=py35 | ROBOTPKG_PYTHON_VERSION=py38 |
| PIP_PREFIX = pip3            | PIP_PREFIX = pip3            |
| ROS_VERSION = bionic         | ROS_VERSION = noetic         |



### Install ROS 

setup your source list:

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```

Set up your keys:

```bash
curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | sudo apt-key add -
```

install ROS main distro:

```bash
sudo apt-get install ros-ROS_VERSION-desktop-full
```

install ROS packages:

```bash
sudo apt-get install ros-ROS_VERSION-urdfdom-py
```

```bash
sudo apt-get install ros-ROS_VERSION-srdfdom
```

```bash
sudo apt-get install ros-ROS_VERSION-joint-state-publisher
```

```bash
sudo apt-get install ros-ROS_VERSION-joint-state-publisher-gui
```

```bash
sudo apt-get install ros-ROS_VERSION-joint-state-controller 
```

```bash
sudo apt-get install ros-ROS_VERSION-gazebo-msgs
```

```bash
sudo apt-get install ros-ROS_VERSION-control-toolbox
```

```bash
sudo apt-get install ros-ROS_VERSION-gazebo-ros
```

```bash
sudo apt-get install ros-ROS_VERSION-controller-manager
```

```bash
sudo apt install ros-ROS_VERSION-joint-trajectory-controller
```



#### Pinocchio stuff

**Add robotpkg as source repository to apt:**

```bash
sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"
```

```bash
sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/wip/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"
```

**Register the authentication certificate of robotpkg:**

```bash
sudo apt install -qqy lsb-release gnupg2 curl
```

```bash
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
```

You need to run at least once apt update to fetch the package descriptions:

```bash
sudo apt-get update
```

Now you can install Pinocchio and the other dependencies:

```bash
sudo apt install robotpkg-PINOCCHIO_PYTHON_VERSION-crocoddyl
```

```bash
sudo apt install robotpkg-PINOCCHIO_PYTHON_VERSION-eigenpy	
```

```bash
sudo apt install robotpkg-PINOCCHIO_PYTHON_VERSION-pinocchio
```

```bash
sudo apt-get install robotpkg-PINOCCHIO_PYTHON_VERSION-quadprog  
```

**NOTE:** If you have issues in installing robotpkg libraries you can try to install them through ROS as:

```bash
sudo apt-get install ros-ROS_VERSION-LIBNAME
```



###  Python

```bash
sudo apt-get install python3-scipy
```

```bash
sudo apt-get install python3-matplotlib
```

```bash
sudo apt-get install python3-termcolor
```

```bash
sudo apt install python3-pip
```

```bash
sudo pip install numpy==1.17.4
```

```bash
sudo pip install joblib==1.2.0
```

```bash
sudo pip install torch==2.0.0
```

```bash
sudo pip install torchvision==0.15.1
```

```bash
sudo pip install tensorboard==2.11.0
```

```bash
sudo pip install torch==2.0.0
```



### Download code and setup ROS workspace

Now that you installed all the dependencies you are ready to get the code, but first you need to create a ros workspace to out the code in:

```bash
mkdir -p ~/ros_ws/src
```

```bash
cd ~/ros_ws/src
```

Now you need to call the following line manually (next you will see that it will be done automatically in the .bashrc)

```bash
source /opt/ros/ROS_VERSION/setup.bash
```

```bash
cd ~/ros_ws/
```

```bash
 catkin_make
```

```bash
 cd ~/ros_ws/src/ 
```

Now you can clone the repository inside the ROS workspace you just created:

```bash
git clone https://github.com/mfocchi/rl_pipeline.git
```

now recompile again (then this step won't bee needed anymore if you just work in Python unless you do not modify / create additional ROS packages)

```bash
cd ~/ros_ws/ 
```

```bash
 catkin_make install
```

the install step install the ros packages inside the "$HOME/ros_ws/install" folder rather than the devel folder. This folder will be added to the ROS_PACKAGE_PATH instead of the devel one.

Finally, run (you should do it any time you add a new ros package)

```bash
 rospack profile
```

There are some additional utilities that I strongly suggest to install. You can find the list  [here](https://github.com/mfocchi/locosim/blob/develop/utils.md).



### Configure environment variables 

```bash
gedit  ~/.bashrc
```

copy the following lines (at the end of the .bashrc), remember to replace the string PYTHON_VERSION with the appropriate version name as explained in [software versions](#software-versions) section:

```bash
source /opt/ros/ROS_VERSION/setup.bash
source $HOME/ros_ws/install/setup.bash
export PATH=/opt/openrobots/bin:$PATH
export LOCOSIM_DIR=$HOME/ros_ws/src/rl_pipeline
export PYTHONPATH=/opt/openrobots/lib/pythonPYTHON_VERSION/site-packages:$PYTHONPATH
export PYSOLO_FROSCIA=$LOCOSIM_DIR/fddp_optimization
export PYTHONPATH=$LOCOSIM_DIR/robot_control:$PYSOLO_FROSCIA:$PYTHONPATH
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/opt/openrobots/share/
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH
```

The .bashrc is a file that is **automatically** sourced whenever you open a new terminal.



#### Compile/Install the code

Whenever you modify some of the ROS packages (e.g. the ones that contain the xacro fles inside the robot_description folder), you need to install them to be sure they are been updated in the ROS install folder. 

```bash
cd ~/ros_ws/ 
```

```bash
 catkin_make install 
```

**IMPORTANT!**

The first time you compile the code the install folder is not existing, therefore won't be added to the PYTHONPATH with the command **source $HOME/ros_ws/install/setup.bash**, and you won't be able to import the package ros_impedance_controller. Therefore, **only once**, after the first time that you compile, run again :

```bash
source .bashrc
```



## Code usage	

The repository contains the implementation of the three approaches presented in the paper: Guided reinforcement learning (GRL), End-to-end reinforcement learning (E2E), and FDDP-based nonlinear trajectory optimization. Both the GRL and the E2E solutions can execute the RL agent in three different modes:
- **train**: to start the policy training process
- **test**: to test the learned policy in the pre-defined test-region
- **inference**: the policy is used without performing the training process  

### GRL & E2E
#### Policy weights
To try the learned policies, download the network weights and decompress them in the `base_controllers/jumpleg_rl` folder. The GRL policy weights are loaded from the `runs` folder, while the ones for the E2E policy are loaded from the `runs_joints` folder.
> You can download the weights directly from the latest release of this repository.

#### Configuring the agent
Inside the `base_controller` folder are the two files responsible for the execution of GRL and E2E implementation, respectively `jumpleg_controller.py` and `jumpleg_controller_instant_conf.py`  

Inside the constructor of the JumplegController class, there are several configuration parameters.

```python
class JumpLegController(BaseControllerFixed):

    def __init__(self, robot_name="jumpleg"):
        super().__init__(robot_name=robot_name)
        self.agentMode = 'inference'
        self.restoreTrain = False
        self.gui = False
        self.model_name = 'latest'
        self.EXTERNAL_FORCE = False
        self.DEBUG = False
        ...
```
PARAMETERS:
- **agentMode**(`str`): RL agent mode,  {"train", "test", "inference"}:  Set it to 'train' to train the NN. The NN weights will be updated and stored in a local folder (robot_control/jumpleg_rl/runs). To evaluate the NN on the test set (test region) set it to 'test', set it to 'inference' for random targets evaluation.
- **restoreTrain**(`bool`): Allows to restore training from a saved run
- **gui**(`bool`): Enable/Disable the launch of Gazebo view
- **model_name**(`str`): Specify the model's weight name to load in the rl agent  
    **! ATTENTION !** the weights have to be in the `base_controllers/jumpleg_rl/runs` folder
- **DEBUG**(`bool`): Enable/Disable the plotting of robot's telemetry

#### Running the agent
Once configured, you can run the agent directly from your IDE or by executing the following command
##### GRL
``` bash
python3 -i $LOCOSIM_DIR/robot_control/base_controllers/jumpleg_controller.py 
```
##### E2E
``` bash
python3 -i $LOCOSIM_DIR/robot_control/base_controllers/jumpleg_controller_instant_conf.py 
```
#### Monitoring the execution

Each time the agent is executed, the corresponding agent mode folder is created/updated inside the `runs` folder. Inside each folder, there is a `logs` folder where Tensorboard event files are saved.
```
robot_control
├── base_controllers
│   ├── ...
|
└── jumpleg_rl
    ├── runs_joints
    │   │── train
    │   │   ├── logs
    │   │   └── partial_weights
    │   └── inference
    │       ├── logs
    ├── runs
    │   │── train
    │   │   ├── logs
    │   │   └── partial_weights
    │
    ├── ...
```
By launching Tensorboard in the desired folder, you can visualize some telemetries regarding the experiment execution.
```bash
tensorboard --logdir runs_joints/train/logs/
```

### FDDP

To run the FDDP optimization, run the script:

``` bash
python3 -i $LOCOSIM_DIR/fddp_optimization/scripts/simple_task_monopod.py 
```

This will solve the optimal control problem for all the point in **test_points.txt** and generate a file **test_optim.csv** that contains: target, error, landing position, computation time.

## Plots
You can find all the plots present in the video and the paper in the [plots folder](/robot_control/jumpleg_rl/utils/plots/)