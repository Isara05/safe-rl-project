# Warehouse Safety + RL PoC Simulation

## Overview
This project is a 2D warehouse safety simulation built with Python and Arcade. The current source of truth is [`simulation.py`](./simulation.py), which models a robot navigating toward a shared goal while interacting with static warehouse infrastructure and dynamic actors.

The simulation includes:
- a robot that can run in Q-learning mode or manual override mode
- a transit AGV with battery-aware charging behavior
- a human worker, a humanoid worker, and an inspection quadruped
- static warehouse objects such as racks, a cupboard, a table, chairs, a forklift, and a crane base
- adaptive safety zones, collision blocking, near-miss tracking, and live safety metrics

## Features
- Warehouse floor layout with aisles, packing area, loading bay, and charging station
- Q-learning proof of concept with discretized observations and reward shaping
- Manual control fallback using the arrow keys
- Safety-aware motion adjustment around dynamic and static obstacles
- Live dashboard showing robot status, AGV state, safety metrics, and reward statistics
- Goal tracking and task completion timing

## Project Files
- [`simulation.py`](./simulation.py): primary simulation implementation and main runtime
- [`main.py`](./main.py): lightweight launcher that starts the simulation
- [`gym_env.py`](./gym_env.py): auxiliary environment file in the repository

## Installation and Setup

To run this project from scratch, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/safe-rl-project.git
   cd safe-rl-project
   ```

2. **Ensure Python 3.9+ is installed:**
   - Check your Python version: `python --version`
   - If not installed, download from [python.org](https://www.python.org/downloads/)

3. **Install required libraries:**
   This project requires the following Python library:
   - `arcade` (for graphics and simulation)

   Install it using pip:
   ```bash
   pip install arcade
   ```

## Running the Simulation

Once installed, run the simulation from the project directory:

```bash
python main.py
```

Alternatively, you can run the simulation directly:

```bash
python simulation.py
```

## Environment and Object Design

The warehouse is separated into static and dynamic objects.

Static objects do not move and act as obstacles:

- walls and aisle boundaries
- racks/shelves
- cupboard
- table and chairs
- parked forklift
- crane control base
- charger, packing area, and loading bay

Dynamic objects move and are checked continuously for safety:

- yellow RL robot, which is the learning agent
- AGV, which follows routes and charges when battery is low
- human worker
- humanoid robot
- quadruped robot

Forklift and crane objects are included to make the warehouse more realistic and to force the navigation policy to account for static industrial equipment.

## Motion and Safety Analysis

The simulation calculates motion and safety from object geometry at each clock tick:

- relative distance between dynamic objects is measured using Euclidean distance between object centres
- clearance subtracts the two object radii from that distance
- speed is calculated from each object's velocity vector
- direction is calculated from the movement vector heading
- closing speed estimates whether two objects are moving toward each other

The safety mechanism adapts to distance and motion:

- green circle means safe
- yellow circle means warning/slow down
- red circle means danger/high risk/stop or avoid
- blue circle means the AGV is charging

The yellow RL robot changes behaviour based on risk: it continues in safe areas, slows near warning thresholds, and stops or avoids movement when high risk is detected.

## Reinforcement Learning (RL) Details

The robot uses a simple tabular Q-learning controller to learn a safe path toward the shared goal. RL is used because it lets the robot learn from interaction, adapt to dynamic humans and obstacles, and improve safety decisions over time.

Core RL components in this project:

- **Agent:** the decision-maker, represented by the yellow RL warehouse robot
- **Environment:** the warehouse, including humans, shelves/racks, static obstacles, charging area, and moving robots
- **State:** the current situation, including distance to humans/objects, speed, position, goal direction, static clearance, and danger flag
- **Action:** movement decisions such as move forward/sideways, slow down, stop, or hold
- **Reward function:** +10 for safe successful movement and -10 for unsafe behaviour, blocked movement, warning state, or collision risk
- **Policy:** the Q-table strategy used by the agent to select actions

- **State** is discretized from the robot's observation:
  - goal direction sector (8 compass sectors)
  - closest dynamic threat clearance bucket
  - closing speed bucket of the most dangerous moving actor
  - static obstacle clearance bucket
  - danger flag if a dynamic threat is too close
- **Actions** are the 9 discrete movement directions:
  - `LEFT`, `RIGHT`, `UP`, `DOWN`, `UP-LEFT`, `UP-RIGHT`, `DOWN-LEFT`, `DOWN-RIGHT`, `HOLD`
- **Reward shaping** encourages progress while penalizing unsafe behavior:
  - positive reward for reducing distance to the goal
  - small step penalty to discourage dithering
  - +10 reward for safe successful movement
  - -10 penalty for warning, high-risk, blocked, or collision-risk states
  - large reward for reaching the goal

The implementation cycle is: state -> action -> reward -> policy update. At each clock tick all objects move, the environment updates, the agent observes the new state, the robot takes an action, and Q-learning updates the policy from the reward.

### AI / RL strategy

The implemented AI approach is **tabular Q-learning**. This is appropriate for the current proof of concept because the observation is discretised into a small state representation:

- goal direction sector
- nearest dynamic-object clearance
- closing speed
- nearest static-object clearance
- danger flag

The Q-table stores the expected value of each action in each state. During the simulation the agent chooses actions with epsilon-greedy exploration, receives reward or penalty, then updates the Q-table.

A future **Deep Reinforcement Learning (DRL)** version could use DQN. In that version, a neural network would estimate Q-values directly from richer continuous inputs such as LiDAR distances, object velocities, relative positions, and map features. The same environment cycle would still apply: state -> action -> reward -> policy update.

### Input data and sensing model

The RL observation uses:

- distance between objects, measured as Euclidean centre distance minus object radii
- speed and direction of dynamic objects
- simulated human speed, with a real-world reference of 2-3 km/h
- obstacle positions from the known 2D warehouse map

The simulation represents object detection geometrically. In a real warehouse robot, this data would be supplied by sensors such as LiDAR (Light Detection and Ranging), which measures distance, detects surroundings, and helps avoid collisions.

### Safety mechanism

Safety is the main outcome:

- **Safe:** continue at normal speed
- **Warning:** slow down near a threshold breach
- **Danger/high risk:** stop or avoid the conflict

The thresholds adapt using distance, object speed, and closing motion. This is more flexible than traditional fixed-rule logic because the learned policy can improve through experience in a dynamic environment.

## Evaluation Metrics and Report Export

The simulation records RMSE and MAE for 2D path tracking. Actual values are the robot's measured positions, predicted values are the nearest points on the planned start-to-goal reference path, and absolute error is the Euclidean distance between them.

Units are reported in metres using the simulation scale `20 pixels = 1 metre`. The live dashboard also displays RMSE and MAE in metres so the on-screen values match the exported report.

Press `E` while the simulation is running to export [`simulation_report.md`](./simulation_report.md), which includes:

- actual values
- predicted values
- absolute error values
- RMSE and MAE
- an explanation of why the metrics are increasing, decreasing, or stable
- RL definitions and implementation details aligned with the marking criteria

## 3D Simulation Extension

The suggested 3D extension should focus on manipulation strategy and a realistic warehouse environment. Blender can be used for modelling and animation, and NVIDIA Omniverse libraries can be used for physics-based simulation and digital-twin workflows.

## Literature Reference

Use Sutton, R. S., and Barto, A. G. (2018), *Reinforcement Learning: An Introduction*, in the literature review for the RL framework, including agent, environment, state, action, reward, policy, and value-learning concepts.

### Mode behavior

- `R` toggles between RL mode and manual override.
- pressing any arrow key automatically switches the robot to manual control.
- in manual mode, the arrow keys control robot motion while safety-aware adjustments still avoid collisions.

### Notes for third-party users

This repository is intended as a proof of concept. The Q-learning controller does not yet persist a trained policy to disk, but it demonstrates learning from simulation experience during a session.

## Controls
- `R`: toggle between RL mode and manual mode
- `Left Arrow`: move left in manual mode
- `Right Arrow`: move right in manual mode
- `Up Arrow`: move up in manual mode
- `Down Arrow`: move down in manual mode

Pressing any arrow key while RL mode is active switches the robot into manual control.

## What The Simulation Tracks
The dashboard in the top-left of the window reports:
- robot safety status and behavior
- AGV battery level and charging state
- collisions and near misses
- minimum and average clearance
- reward average
- path-tracking RMSE and MAE in metres
- elapsed or completion time

## Current Simulation Logic
According to `simulation.py`, the robot:
- starts in RL mode by default
- moves toward a shared goal in the warehouse
- slows down or stops when adaptive safety thresholds are violated
- avoids both dynamic actors and nearby static objects

The AGV:
- follows a route while battery is sufficient
- switches to charging behavior when battery is low
- resumes operation after recharging

Dynamic actors:
- follow waypoint routes
- adjust speed and heading based on local safety conditions
- maintain their own warning and danger zones

## Notes
- `main.py` is now only an entry point; the actual simulation logic lives in `simulation.py`.
- The older Pygame-based prototype is no longer the active implementation described by this README.
