# Sawyer Model
Mujoco model for Rething Robotics Sawyer with different actuator options. See [sawyer.xml](sawyer.xml) for details.
- Joint postion control 
- End effector control
- Joint motor control
- Parallel jaw gripper position control

![Alt text](assets/sawyer.png?raw=false "sawyer")

# Change log

#### April'20:: Model update to MuJoCo 2.0

- adding actuator options: postion, motor, gripper
- re-org geoms into viz and col group
- joint and actuator properties updated
- re-org model to be easily importable
- removed dummy bodies
- update to the collision geometries
- updated assets paths for better include support
- gripper follows include structure now