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

2. **Ensure Python 3.7+ is installed:**
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
- path-tracking RMSE and MAE
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
