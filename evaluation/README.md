# VIP Evaluation: Trajectory Optimization with Pre-Trained Reward Models

The codebase contains the evaluation codebase for [VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training](https://arxiv.org/abs/2210.00030). It uses trajectory optimization to synthesize a trajectory that optimizes the reward function given by the pre-trained representation model (e.g., VIP). 

# Environment Installation
Install the FrankaKitchen evaluation environment (Assuming Mujoco dependency installed):
```
cd evaluation;
cd mj_envs; pip install -e .
```
The environment also depends on the [mjrl] package:
```
git clone git@github.com:aravindr93/mjrl.git
cd mjrl; pip install -e .
```

Optionally, you can speed up Mujoco rendering by switching to its GPU backend: 
```
cd ~/.conda/envs/vip/lib/python3.9/site-packages/mujoco_py
```
Change line 74 of ```builder.py``` to
```
Builder = LinuxGPUExtensionBuilder
```
Then, open Python and ```import mujoco_py```. This should re-prompt a Mujoco build. Then, ```print(mujoco_py.cymj)``` to make sure that you see ```linuxgpuextensionbuilder``` in the output. 

# TrajOpt Installation
Install the Trajopt package:
```
cd evaluation/trajopt/trajopt; pip install -e .
```

## Generate Demonstrations
Generate demonstrations that will provide a goal image and optionally the initial pose of the robot. We have already generated the demonstration for task ```kitchen_sdoor_open-v3``` in ```./evaluation/dataset```, and you may generate demonstrations for all tasks
```
cd dataset; ./scripts/generate_all_demo.sh
```
Note that the generated trajectory may not actually solve the task! This can be resolved by usually (1) re-running the script, or (2) make the MPPI solver stronger by increasing MPPI related hyperparameters in ```./evaluation/trajopt/trajopt/config/mppi_config.yaml```. 

Once the demonstration is generated, change ```DATASET_ABS_PATH``` in ```./evaluation/trajopt/__init__.py``` to point to the absolute path of your dataset folder. 

I apologize I cannot share the original demonstrations used in the paper since I am no longer at Meta, but the original demonstrations are generated in the same way.

## Running TrajOpt Evaluation
To quickly verify whether your installation has worked, run:
```
python run_mppi.py env=kitchen_sdoor_open-v3 embedding=vip
```
In the paper, the "Easy" setting is enabled by setting ```env_kwargs.init_timestep=20```, which initializes the robot pose to the robot pose achieved in the 20th frame in the task demonstration. The "Hard" setting is enabled by setting ```env_kwargs.init_timestep=0```. 
