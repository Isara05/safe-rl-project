# Simple Robot Simulation Environment

## Overview
This project is a simple 2D robot simulation environment created using Python and Pygame. It is designed as the starting point for a Safe Reinforcement Learning for Human–Robot Collaboration project.

The simulation includes:
- a robot shown as a blue circle
- a human shown as a red moving circle
- a goal shown as a star
- obstacles shown as black blocks

This basic environment can later be extended for collision detection, distance sensing, reward design, and reinforcement learning training.

Requirements

Make sure Python is installed on your system.

Install Pygame using:

pip install pygame

Move to the correct folder:

cd /d D:\SWIN_3\Project_A (According to your saved drive and folder)

Run the program:

python main.py
Controls

Use the arrow keys to move the robot:

Up arrow → move up
Down arrow → move down
Left arrow → move left
Right arrow → move right
Project Components
Robot

**##Current Limitations**

This is only the initial version of the simulation. At this stage:

the robot is manually controlled
there is no collision detection
there is no reward function
there is no reinforcement learning yet
Future Improvements

**##Possible future extensions include:**

collision detection with obstacles and human
distance calculation to nearby objects
state, action, and reward definitions
reinforcement learning integration
safe navigation logic
automatic robot decision-making
