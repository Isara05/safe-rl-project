"""Warehouse safety simulation with a simple Q-learning robot controller.

This simulation uses a discretized observation space, a small discrete action
set, and reward shaping to guide the robot toward a shared goal while
avoiding static and dynamic obstacles.
"""

import arcade
import math
import random
import time
from dataclasses import dataclass, field


SCREEN_WIDTH = 1680
SCREEN_HEIGHT = 960
SCREEN_TITLE = "Warehouse Safety + RL PoC Simulation"

FLOOR_COLOR = (229, 232, 226)
GRID_COLOR = (216, 220, 214)
AISLE_COLOR = (201, 205, 198)
AISLE_MARKING = (245, 211, 79)
WALL_STRIPE = (150, 158, 150)

TEXT_COLOR = arcade.color.BLACK
INFO_COLOR = arcade.color.DARK_BLUE
SUCCESS_COLOR = arcade.color.DARK_GREEN
WARNING_COLOR = arcade.color.ORANGE
COLLISION_COLOR = arcade.color.RED

ROBOT_SAFE = (60, 180, 90)
ROBOT_WARNING = (245, 200, 60)
ROBOT_DANGER = (220, 70, 70)
ROBOT_TOP = (220, 230, 240)
ROBOT_DARK = (35, 55, 95)
ROBOT_WHEEL = (35, 35, 35)
ROBOT_SENSOR = (255, 220, 70)

AGV_SAFE = (70, 130, 255)
AGV_WARNING = (255, 185, 60)
AGV_DANGER = (235, 85, 85)
AGV_CHARGING = (120, 220, 255)

HUMAN_SKIN = (230, 190, 160)
HUMAN_HELMET = (245, 210, 60)
HUMAN_PANTS = (60, 70, 90)
HUMAN_COLORS = {
    "human": (231, 101, 72),
    "humanoid": (72, 162, 115),
    "quadruped": (120, 92, 190),
}

STATIC_COLORS = {
    "wall": (120, 118, 116),
    "rack": (108, 108, 110),
    "cupboard": (147, 113, 84),
    "table": (164, 136, 92),
    "chair": (110, 84, 62),
    "forklift": (224, 170, 44),
    "crane": (120, 140, 170),
}

GOAL_FILL = (219, 238, 213)
GOAL_BORDER = (67, 138, 73)
GOAL_TEXT = (44, 84, 48)

CHARGER_FILL = (208, 231, 241)
CHARGER_BORDER = (54, 113, 144)
CHARGER_TEXT = (28, 70, 92)

SAFE_ZONE_COLOR = (255, 207, 92, 10)
DANGER_ZONE_COLOR = (230, 94, 81, 14)
PREDICTION_COLOR = (95, 101, 109, 70)
PATH_COLOR = (86, 124, 164, 55)
PANEL_COLOR = (245, 246, 241, 235)
PANEL_BORDER = (174, 180, 170)

ROBOT_RADIUS = 22
AGV_RADIUS = 20
HUMAN_RADIUS = 18
HUMANOID_RADIUS = 19
QUADRUPED_RADIUS = 16

ROBOT_BASE_SPEED = 4.6
ROBOT_SLOW_SPEED = 2.0
ROBOT_DANGER_SPEED = 0.0

AGV_SPEED_RANGE = (1.2, 3.2)
HUMAN_SPEED_RANGE = (0.7, 1.4)
HUMANOID_SPEED_RANGE = (0.8, 1.6)
QUADRUPED_SPEED_RANGE = (1.4, 2.2)

GOAL_REACHED_DISTANCE = 30
CHARGING_REACHED_DISTANCE = 28
WAYPOINT_REACHED_DISTANCE = 28

ANGLE_SMOOTHING = 0.22
RL_STEP_INTERVAL = 0.14
MESSAGE_HOLD_TIME = 0.40
NEAR_MISS_MARGIN = 22
RL_GAMMA = 0.92
RL_ALPHA = 0.14
RL_EPSILON = 0.12
PATH_SAMPLE_LIMIT = 240

ACTIONS = [
    (-1, 0),
    (1, 0),
    (0, 1),
    (0, -1),
    (-1, 1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (0, 0),
]

ACTION_LABELS = [
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "UP-LEFT",
    "UP-RIGHT",
    "DOWN-LEFT",
    "DOWN-RIGHT",
    "HOLD",
]


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize(dx, dy):
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return 0.0, 0.0
    return dx / length, dy / length


def wrap_text_lines(text, max_chars):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def normalize_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def smooth_angle(current, target, factor=ANGLE_SMOOTHING):
    delta = normalize_angle(target - current)
    return current + delta * factor


def circle_rect_collision(circle_x, circle_y, radius, rect):
    rect_x, rect_y, rect_w, rect_h = rect
    left = rect_x - rect_w / 2
    right = rect_x + rect_w / 2
    bottom = rect_y - rect_h / 2
    top = rect_y + rect_h / 2
    closest_x = max(left, min(circle_x, right))
    closest_y = max(bottom, min(circle_y, top))
    dx = circle_x - closest_x
    dy = circle_y - closest_y
    return (dx * dx + dy * dy) < (radius * radius)


def distance_circle_to_rect_edge(circle_x, circle_y, rect):
    rect_x, rect_y, rect_w, rect_h = rect
    left = rect_x - rect_w / 2
    right = rect_x + rect_w / 2
    bottom = rect_y - rect_h / 2
    top = rect_y + rect_h / 2
    closest_x = max(left, min(circle_x, right))
    closest_y = max(bottom, min(circle_y, top))
    dx = circle_x - closest_x
    dy = circle_y - closest_y
    return math.sqrt(dx * dx + dy * dy)


@dataclass
class StaticObject:
    name: str
    category: str
    x: float
    y: float
    width: float
    height: float
    color: tuple[int, int, int]

    @property
    def rect(self):
        return (self.x, self.y, self.width, self.height)


@dataclass
class DynamicObject:
    name: str
    kind: str
    x: float
    y: float
    radius: float
    color: tuple[int, int, int]
    route: list[tuple[float, float]]
    speed_min: float
    speed_max: float
    speed: float
    vx: float = 0.0
    vy: float = 0.0
    angle: float = 0.0
    route_index: int = 0
    status: str = "SAFE"
    safety_behavior: str = "CRUISE"
    last_clearance: float = 999.0
    adaptive_safe_distance: float = 150.0
    adaptive_danger_distance: float = 90.0
    message: str = ""

    def step_route(self):
        if not self.route:
            return 0.0, 0.0
        target_x, target_y = self.route[self.route_index]
        dx = target_x - self.x
        dy = target_y - self.y
        if math.sqrt(dx * dx + dy * dy) <= WAYPOINT_REACHED_DISTANCE:
            self.route_index = (self.route_index + 1) % len(self.route)
            target_x, target_y = self.route[self.route_index]
            dx = target_x - self.x
            dy = target_y - self.y
        return normalize(dx, dy)

    def scalar_speed(self):
        return math.sqrt(self.vx * self.vx + self.vy * self.vy)


@dataclass
class SafetyMetrics:
    collisions: int = 0
    near_misses: int = 0
    unsafe_steps: int = 0
    total_reward: float = 0.0
    reward_samples: int = 0
    min_clearance: float = 999.0
    clearance_sum: float = 0.0
    clearance_samples: int = 0
    path_error_sum_sq: float = 0.0
    path_error_abs_sum: float = 0.0
    path_error_samples: int = 0
    tracking_errors: list[float] = field(default_factory=list)

    def record_clearance(self, clearance):
        self.min_clearance = min(self.min_clearance, clearance)
        self.clearance_sum += clearance
        self.clearance_samples += 1

    def record_path_error(self, error_value):
        self.path_error_sum_sq += error_value * error_value
        self.path_error_abs_sum += abs(error_value)
        self.path_error_samples += 1
        self.tracking_errors.append(error_value)
        if len(self.tracking_errors) > PATH_SAMPLE_LIMIT:
            self.tracking_errors.pop(0)

    @property
    def avg_clearance(self):
        if not self.clearance_samples:
            return 0.0
        return self.clearance_sum / self.clearance_samples

    @property
    def rmse(self):
        if not self.path_error_samples:
            return 0.0
        return math.sqrt(self.path_error_sum_sq / self.path_error_samples)

    @property
    def mae(self):
        if not self.path_error_samples:
            return 0.0
        return self.path_error_abs_sum / self.path_error_samples

    @property
    def average_reward(self):
        if not self.reward_samples:
            return 0.0
        return self.total_reward / self.reward_samples


class QLearningAgent:
    """Simple tabular Q-learning agent for the warehouse robot."""

    def __init__(self):
        self.q_table = {}
        self.last_state = None
        self.last_action = None

    def discretize(self, observation):
        """Convert raw observations into a compact discrete state tuple."""
        goal_sector = int(observation["goal_sector"])
        clearance_bucket = int(clamp(observation["clearance"] // 35, 0, 8))
        closing_bucket = int(clamp((observation["closing_speed"] + 2.5) // 1.0, 0, 6))
        static_bucket = int(clamp(observation["static_clearance"] // 35, 0, 8))
        danger_flag = 1 if observation["danger"] else 0
        return (goal_sector, clearance_bucket, closing_bucket, static_bucket, danger_flag)

    def select_action(self, state):
        """Choose an action with epsilon-greedy selection."""
        if random.random() < RL_EPSILON or state not in self.q_table:
            return random.randrange(len(ACTIONS))
        values = self.q_table[state]
        best_value = max(values)
        best_actions = [index for index, value in enumerate(values) if value == best_value]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        """Update the Q-table using the Bellman equation."""
        self.q_table.setdefault(state, [0.0] * len(ACTIONS))
        self.q_table.setdefault(next_state, [0.0] * len(ACTIONS))
        old_value = self.q_table[state][action]
        next_best = max(self.q_table[next_state])
        self.q_table[state][action] = old_value + RL_ALPHA * (reward + RL_GAMMA * next_best - old_value)


class WarehouseSimulation(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(FLOOR_COLOR)

        self.start_time = time.time()
        self.last_rl_step = 0.0
        self.rl_enabled = True
        self.rl_mode_name = "Q-Learning PoC"
        self.rl_agent = QLearningAgent()
        self.current_reward = 0.0
        self.last_observation_state = None
        self.last_action_index = None
        self.last_action_label = "HOLD"

        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False

        self.robot_start_x = 180.0
        self.robot_start_y = 790.0
        self.robot_x = self.robot_start_x
        self.robot_y = self.robot_start_y
        self.robot_angle = 0.0
        self.robot_color = ROBOT_SAFE
        self.robot_status = "SAFE"
        self.robot_behavior = "CRUISE"
        self.robot_speed = ROBOT_BASE_SPEED
        self.robot_message = ""
        self.robot_message_color = WARNING_COLOR
        self.robot_message_until = 0.0
        self.last_near_miss_time = 0.0

        self.goal_x = 1490.0
        self.goal_y = 170.0
        self.goal_reached = False
        self.completion_time = None

        self.charger_x = 1500.0
        self.charger_y = 820.0
        self.agv_battery = 100.0
        self.agv_low_battery_threshold = 20.0
        self.agv_resume_battery_threshold = 95.0
        self.agv_drain_rate = 0.05
        self.agv_charge_rate = 0.45
        self.agv_mode = "WORKING"
        self.agv_charge_pulse_time = 0.0

        self.metrics = SafetyMetrics()

        self.static_objects = self.build_static_objects()
        self.dynamic_objects = self.build_dynamic_objects()
        self.agv_actor = next(actor for actor in self.dynamic_objects if actor.kind == "agv")

        self.robot_path_trace = [(self.robot_x, self.robot_y)]
        self.dashboard_lines = self.build_strategy_summary()

    def build_static_objects(self):
        objects = []
        rack_positions = [
            (520, 250), (520, 450), (520, 650),
            (980, 250), (980, 450), (980, 650),
        ]
        for index, (x, y) in enumerate(rack_positions, start=1):
            objects.append(StaticObject(f"Rack {index}", "rack", x, y, 120, 46, STATIC_COLORS["rack"]))

        objects.extend([
            StaticObject("Cupboard A", "cupboard", 170, 560, 100, 50, STATIC_COLORS["cupboard"]),
            StaticObject("Table A", "table", 1230, 520, 150, 68, STATIC_COLORS["table"]),
            StaticObject("Chair A", "chair", 1142, 520, 34, 34, STATIC_COLORS["chair"]),
            StaticObject("Chair B", "chair", 1318, 520, 34, 34, STATIC_COLORS["chair"]),
            StaticObject("Parked Forklift", "forklift", 1460, 530, 92, 54, STATIC_COLORS["forklift"]),
            StaticObject("Crane Control Base", "crane", 1490, 720, 92, 60, STATIC_COLORS["crane"]),
        ])
        return objects

    def build_dynamic_objects(self):
        return [
            DynamicObject(
                name="Transit AGV",
                kind="agv",
                x=280,
                y=790,
                radius=AGV_RADIUS,
                color=AGV_SAFE,
                route=[
                    (640, 790),
                    (1120, 790),
                    (1400, 790),
                    (1400, 650),
                    (1400, 300),
                    (1160, 300),
                    (1160, 790),
                    (780, 790),
                ],
                speed_min=AGV_SPEED_RANGE[0],
                speed_max=AGV_SPEED_RANGE[1],
                speed=2.5,
            ),
            DynamicObject(
                name="Worker 1",
                kind="human",
                x=300,
                y=560,
                radius=HUMAN_RADIUS,
                color=HUMAN_COLORS["human"],
                route=[(300, 560), (300, 340), (420, 340), (420, 560)],
                speed_min=HUMAN_SPEED_RANGE[0],
                speed_max=HUMAN_SPEED_RANGE[1],
                speed=1.1,
            ),
            DynamicObject(
                name="Worker 2",
                kind="humanoid",
                x=1110,
                y=575,
                radius=HUMANOID_RADIUS,
                color=HUMAN_COLORS["humanoid"],
                route=[(1110, 575), (1210, 575), (1210, 700), (1110, 700)],
                speed_min=HUMANOID_SPEED_RANGE[0],
                speed_max=HUMANOID_SPEED_RANGE[1],
                speed=1.2,
            ),
            DynamicObject(
                name="Inspection Quadruped",
                kind="quadruped",
                x=1180,
                y=180,
                radius=QUADRUPED_RADIUS,
                color=HUMAN_COLORS["quadruped"],
                route=[(1180, 180), (1450, 180), (1450, 250), (1180, 250)],
                speed_min=QUADRUPED_SPEED_RANGE[0],
                speed_max=QUADRUPED_SPEED_RANGE[1],
                speed=1.7,
            ),
        ]

    def build_strategy_summary(self):
        return [
            "Zones: aisles, packing, loading, charging.",
            "Static: racks, cupboard, table, chairs.",
            "Dynamic: robot, AGV, human, humanoid.",
            "Safety: distance, speed, heading aware.",
            "RL: Q-learning with reward shaping.",
        ]

    def distance_between(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def position_blocked(self, x, y, radius):
        if not (radius <= x <= SCREEN_WIDTH - radius):
            return "wall"
        if not (radius <= y <= SCREEN_HEIGHT - radius):
            return "wall"

        for obj in self.static_objects:
            if circle_rect_collision(x, y, radius, obj.rect):
                return obj.category
        return None

    def nearest_static_clearance(self, x, y, radius):
        min_dist = float("inf")
        for obj in self.static_objects:
            edge_dist = distance_circle_to_rect_edge(x, y, obj.rect)
            clearance = max(0.0, edge_dist - radius)
            min_dist = min(min_dist, clearance)
        wall_clearance = min(x - radius, y - radius, SCREEN_WIDTH - x - radius, SCREEN_HEIGHT - y - radius)
        min_dist = min(min_dist, wall_clearance)
        return max(0.0, min_dist)

    def nearest_dynamic_threat(self, x, y, radius, own_vx=0.0, own_vy=0.0, ignore=None):
        best = None
        best_score = -float("inf")

        for actor in self.dynamic_objects:
            if actor is ignore:
                continue
            dx = actor.x - x
            dy = actor.y - y
            distance = math.sqrt(dx * dx + dy * dy)
            clearance = distance - (radius + actor.radius)
            rel_vx = actor.vx - own_vx
            rel_vy = actor.vy - own_vy
            toward_dx, toward_dy = normalize(dx, dy)
            closing_speed = -(rel_vx * toward_dx + rel_vy * toward_dy)
            score = (180 - clearance) + max(0.0, closing_speed) * 40

            if score > best_score:
                best_score = score
                best = {
                    "actor": actor,
                    "distance": distance,
                    "clearance": clearance,
                    "closing_speed": closing_speed,
                    "relative_dx": dx,
                    "relative_dy": dy,
                }

        return best

    def adaptive_safety_distances(self, own_speed, other_speed, closing_speed):
        safe_distance = 70 + own_speed * 10 + other_speed * 8 + max(0.0, closing_speed) * 18
        danger_distance = safe_distance * 0.52
        return safe_distance, danger_distance

    def distance_to_reference_path(self, x, y):
        x1, y1 = self.robot_start_x, self.robot_start_y
        x2, y2 = self.goal_x, self.goal_y
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return 0.0
        t = ((x - x1) * dx + (y - y1) * dy) / length_sq
        t = clamp(t, 0.0, 1.0)
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return self.distance_between(x, y, proj_x, proj_y)

    def build_robot_observation(self):
        """Build the current observation used by the RL agent."""
        threat = self.nearest_dynamic_threat(self.robot_x, self.robot_y, ROBOT_RADIUS)
        goal_dx = self.goal_x - self.robot_x
        goal_dy = self.goal_y - self.robot_y
        goal_angle = math.degrees(math.atan2(goal_dy, goal_dx))
        sector = int(((normalize_angle(goal_angle - self.robot_angle) + 180) // 45) % 8)

        clearance = 999.0 if threat is None else threat["clearance"]
        closing_speed = 0.0 if threat is None else threat["closing_speed"]
        static_clearance = self.nearest_static_clearance(self.robot_x, self.robot_y, ROBOT_RADIUS)
        danger = False
        if threat is not None:
            other = threat["actor"]
            safe_dist, danger_dist = self.adaptive_safety_distances(
                self.robot_speed, other.scalar_speed(), threat["closing_speed"]
            )
            danger = threat["clearance"] < danger_dist

        return {
            "goal_sector": sector,
            "goal_dx": goal_dx,
            "goal_dy": goal_dy,
            "clearance": clearance,
            "closing_speed": closing_speed,
            "static_clearance": static_clearance,
            "danger": danger,
        }

    def compute_rl_reward(self, previous_goal_dist, new_goal_dist, moved):
        """Compute reward for the RL agent based on progress and safety."""
        reward = (previous_goal_dist - new_goal_dist) * 0.7
        reward -= 0.01
        reward += 0.04 if moved else -0.08
        reward -= 0.35 if self.robot_status == "WARNING" else 0.0
        reward -= 0.9 if self.robot_status == "HIGH RISK" else 0.0
        reward += 15.0 if self.goal_reached else 0.0
        return reward

    def set_robot_message(self, message, color=WARNING_COLOR):
        self.robot_message = message
        self.robot_message_color = color
        self.robot_message_until = time.time() + MESSAGE_HOLD_TIME

    def update_robot_risk_state(self):
        threat = self.nearest_dynamic_threat(self.robot_x, self.robot_y, ROBOT_RADIUS)
        static_clearance = self.nearest_static_clearance(self.robot_x, self.robot_y, ROBOT_RADIUS)

        if threat is None:
            clearance = static_clearance
            safe_distance = 110
            danger_distance = 65
            threat_label = "static"
            closing_speed = 0.0
        else:
            other = threat["actor"]
            clearance = min(threat["clearance"], static_clearance)
            safe_distance, danger_distance = self.adaptive_safety_distances(
                ROBOT_BASE_SPEED, other.scalar_speed(), threat["closing_speed"]
            )
            threat_label = other.kind
            closing_speed = threat["closing_speed"]

        self.metrics.record_clearance(clearance)

        if clearance <= danger_distance:
            self.robot_status = "HIGH RISK"
            self.robot_behavior = "STOP"
            self.robot_color = ROBOT_DANGER
            self.robot_speed = ROBOT_DANGER_SPEED
            self.metrics.unsafe_steps += 1
            self.set_robot_message(f"Stop: {threat_label} conflict")
        elif clearance <= safe_distance:
            self.robot_status = "WARNING"
            self.robot_behavior = "SLOW"
            self.robot_color = ROBOT_WARNING
            self.robot_speed = ROBOT_SLOW_SPEED
            if closing_speed > 0.4:
                self.set_robot_message(f"Slow down: closing on {threat_label}")
            else:
                self.set_robot_message(f"Slow zone near {threat_label}")
        else:
            self.robot_status = "SAFE"
            self.robot_behavior = "CRUISE"
            self.robot_color = ROBOT_SAFE
            self.robot_speed = ROBOT_BASE_SPEED

        if clearance <= NEAR_MISS_MARGIN and (time.time() - self.last_near_miss_time) > 0.75:
            self.metrics.near_misses += 1
            self.last_near_miss_time = time.time()

        tracking_error = self.distance_to_reference_path(self.robot_x, self.robot_y)
        self.metrics.record_path_error(tracking_error)

    def build_safety_vector_for_robot(self, intended_dx, intended_dy):
        final_dx = intended_dx
        final_dy = intended_dy
        threat = self.nearest_dynamic_threat(
            self.robot_x,
            self.robot_y,
            ROBOT_RADIUS,
            intended_dx * self.robot_speed,
            intended_dy * self.robot_speed,
        )

        if threat is not None:
            actor = threat["actor"]
            safe_distance, danger_distance = self.adaptive_safety_distances(
                self.robot_speed, actor.scalar_speed(), threat["closing_speed"]
            )
            if threat["clearance"] < safe_distance:
                away_dx, away_dy = normalize(-threat["relative_dx"], -threat["relative_dy"])
                side_dx = -away_dy
                side_dy = away_dx
                if side_dx * intended_dx + side_dy * intended_dy < 0:
                    side_dx *= -1
                    side_dy *= -1
                weight = 2.2 if threat["clearance"] < danger_distance else 1.1
                final_dx += away_dx * weight + side_dx * 0.8
                final_dy += away_dy * weight + side_dy * 0.8

        future_x = self.robot_x + intended_dx * 26
        future_y = self.robot_y + intended_dy * 26
        for obj in self.static_objects:
            future_clearance = max(0.0, distance_circle_to_rect_edge(future_x, future_y, obj.rect) - ROBOT_RADIUS)
            if future_clearance < 14:
                away_dx, away_dy = normalize(self.robot_x - obj.x, self.robot_y - obj.y)
                final_dx += away_dx * 0.7
                final_dy += away_dy * 0.7

        return normalize(final_dx, final_dy)

    def try_move_robot(self, move_dx, move_dy):
        if move_dx == 0 and move_dy == 0:
            return False

        step = self.robot_speed
        if step == 0:
            return False

        candidates = [
            (self.robot_x + move_dx * step, self.robot_y + move_dy * step),
            (self.robot_x + move_dx * step, self.robot_y),
            (self.robot_x, self.robot_y + move_dy * step),
            (self.robot_x + move_dy * step * 0.8, self.robot_y - move_dx * step * 0.8),
            (self.robot_x - move_dy * step * 0.8, self.robot_y + move_dx * step * 0.8),
        ]

        for nx, ny in candidates:
            if self.position_blocked(nx, ny, ROBOT_RADIUS):
                continue

            collision = False
            for actor in self.dynamic_objects:
                if self.distance_between(nx, ny, actor.x, actor.y) < (ROBOT_RADIUS + actor.radius + 2):
                    collision = True
                    self.metrics.collisions += 1
                    self.set_robot_message(f"Collision blocked by {actor.kind}", COLLISION_COLOR)
                    break

            if collision:
                continue

            move_angle = math.degrees(math.atan2(ny - self.robot_y, nx - self.robot_x))
            self.robot_angle = smooth_angle(self.robot_angle, move_angle)
            self.robot_x = nx
            self.robot_y = ny
            self.robot_path_trace.append((self.robot_x, self.robot_y))
            if len(self.robot_path_trace) > PATH_SAMPLE_LIMIT:
                self.robot_path_trace.pop(0)
            return True

        return False

    def build_manual_input_vector(self):
        dx = 0
        dy = 0
        if self.left_pressed:
            dx -= 1
        if self.right_pressed:
            dx += 1
        if self.up_pressed:
            dy += 1
        if self.down_pressed:
            dy -= 1
        return normalize(dx, dy)

    def rl_control_step(self):
        if not self.rl_enabled or self.goal_reached:
            return

        now = time.time()
        if now - self.last_rl_step < RL_STEP_INTERVAL:
            return
        self.last_rl_step = now

        observation = self.build_robot_observation()
        state = self.rl_agent.discretize(observation)
        action_index = self.rl_agent.select_action(state)
        self.last_action_label = ACTION_LABELS[action_index]

        intended_dx, intended_dy = ACTIONS[action_index]
        intended_dx, intended_dy = normalize(intended_dx, intended_dy)
        safe_dx, safe_dy = self.build_safety_vector_for_robot(intended_dx, intended_dy)

        previous_goal_dist = self.distance_between(self.robot_x, self.robot_y, self.goal_x, self.goal_y)
        moved = self.try_move_robot(safe_dx, safe_dy)
        self.check_goal()
        new_goal_dist = self.distance_between(self.robot_x, self.robot_y, self.goal_x, self.goal_y)

        reward = self.compute_rl_reward(previous_goal_dist, new_goal_dist, moved)
        next_state = self.rl_agent.discretize(self.build_robot_observation())
        self.rl_agent.update(state, action_index, reward, next_state)

        self.current_reward = reward
        self.metrics.total_reward += reward
        self.metrics.reward_samples += 1
        self.last_observation_state = state
        self.last_action_index = action_index

    def check_goal(self):
        distance = self.distance_between(self.robot_x, self.robot_y, self.goal_x, self.goal_y)
        if distance < GOAL_REACHED_DISTANCE and not self.goal_reached:
            self.goal_reached = True
            self.completion_time = time.time() - self.start_time
            self.set_robot_message("Goal reached", SUCCESS_COLOR)

    def can_dynamic_occupy(self, actor, x, y):
        if self.position_blocked(x, y, actor.radius):
            return False
        if self.distance_between(x, y, self.robot_x, self.robot_y) < actor.radius + ROBOT_RADIUS + 4:
            return False

        for other in self.dynamic_objects:
            if other is actor:
                continue
            if self.distance_between(x, y, other.x, other.y) < actor.radius + other.radius + 4:
                return False
        return True

    def update_agv_battery(self):
        if self.agv_mode == "CHARGING":
            self.agv_battery = min(100.0, self.agv_battery + self.agv_charge_rate)
            self.agv_charge_pulse_time += 0.08
            self.agv_actor.x = self.charger_x
            self.agv_actor.y = self.charger_y
            self.agv_actor.vx = 0.0
            self.agv_actor.vy = 0.0
            self.agv_actor.speed = 0.0
        else:
            self.agv_battery = max(0.0, self.agv_battery - self.agv_drain_rate)

    def update_agv_mode(self):
        distance_to_charger = self.distance_between(self.agv_actor.x, self.agv_actor.y, self.charger_x, self.charger_y)
        if self.agv_mode in ["WORKING", "RETURNING"] and self.agv_battery <= self.agv_low_battery_threshold:
            self.agv_mode = "GOING_TO_CHARGE"
        if self.agv_mode == "GOING_TO_CHARGE" and distance_to_charger <= CHARGING_REACHED_DISTANCE:
            self.agv_mode = "CHARGING"
        if self.agv_mode == "CHARGING" and self.agv_battery >= self.agv_resume_battery_threshold:
            self.agv_mode = "RETURNING"

    def get_agv_target(self):
        if self.agv_mode in ["GOING_TO_CHARGE", "CHARGING"]:
            return self.charger_x, self.charger_y
        target = self.agv_actor.route[self.agv_actor.route_index]
        dx = target[0] - self.agv_actor.x
        dy = target[1] - self.agv_actor.y
        if math.sqrt(dx * dx + dy * dy) <= WAYPOINT_REACHED_DISTANCE:
            self.agv_actor.route_index = (self.agv_actor.route_index + 1) % len(self.agv_actor.route)
            target = self.agv_actor.route[self.agv_actor.route_index]
        return target

    def update_dynamic_actor_safety(self, actor, intended_dx, intended_dy):
        threat = self.nearest_dynamic_threat(actor.x, actor.y, actor.radius, actor.vx, actor.vy, ignore=actor)
        static_clearance = self.nearest_static_clearance(actor.x, actor.y, actor.radius)
        clearance = static_clearance
        closing_speed = 0.0
        other_speed = 0.0
        nearest_label = "static"

        if threat is not None:
            clearance = min(clearance, threat["clearance"])
            closing_speed = threat["closing_speed"]
            other_speed = threat["actor"].scalar_speed()
            nearest_label = threat["actor"].kind

        safe_distance, danger_distance = self.adaptive_safety_distances(actor.speed, other_speed, closing_speed)
        actor.adaptive_safe_distance = safe_distance
        actor.adaptive_danger_distance = danger_distance
        actor.last_clearance = clearance

        if clearance <= danger_distance:
            actor.status = "HIGH RISK"
            actor.safety_behavior = "AVOID"
            actor.message = f"Avoid {nearest_label}"
            target_speed = max(actor.speed_min * 0.35, 0.6)
        elif clearance <= safe_distance:
            actor.status = "WARNING"
            actor.safety_behavior = "SLOW"
            actor.message = f"Slow near {nearest_label}"
            target_speed = actor.speed_min
        else:
            actor.status = "SAFE"
            actor.safety_behavior = "CRUISE"
            actor.message = ""
            target_speed = actor.speed_max

        actor.speed = clamp(target_speed, 0.0, actor.speed_max)

        final_dx = intended_dx
        final_dy = intended_dy
        if threat is not None and threat["clearance"] < safe_distance:
            away_dx, away_dy = normalize(-threat["relative_dx"], -threat["relative_dy"])
            side_dx = -away_dy
            side_dy = away_dx
            if side_dx * intended_dx + side_dy * intended_dy < 0:
                side_dx *= -1
                side_dy *= -1
            weight = 2.0 if threat["clearance"] < danger_distance else 1.0
            final_dx += away_dx * weight + side_dx * 0.6
            final_dy += away_dy * weight + side_dy * 0.6

        future_x = actor.x + intended_dx * 24
        future_y = actor.y + intended_dy * 24
        for obj in self.static_objects:
            future_clearance = max(0.0, distance_circle_to_rect_edge(future_x, future_y, obj.rect) - actor.radius)
            if future_clearance < 12:
                away_dx, away_dy = normalize(actor.x - obj.x, actor.y - obj.y)
                final_dx += away_dx * 0.65
                final_dy += away_dy * 0.65

        return normalize(final_dx, final_dy)

    def try_move_dynamic_actor(self, actor, move_dx, move_dy):
        if actor.speed <= 0 or (move_dx == 0 and move_dy == 0):
            actor.vx = 0.0
            actor.vy = 0.0
            return False

        step = actor.speed
        candidates = [
            (actor.x + move_dx * step, actor.y + move_dy * step),
            (actor.x + move_dx * step * 0.7, actor.y + move_dy * step * 0.7),
            (actor.x + move_dx * step, actor.y),
            (actor.x, actor.y + move_dy * step),
        ]

        for nx, ny in candidates:
            if not self.can_dynamic_occupy(actor, nx, ny):
                continue
            actor.vx = nx - actor.x
            actor.vy = ny - actor.y
            actor.angle = smooth_angle(actor.angle, math.degrees(math.atan2(actor.vy, actor.vx)))
            actor.x = nx
            actor.y = ny
            return True

        actor.vx = 0.0
        actor.vy = 0.0
        return False

    def update_dynamic_actors(self):
        self.update_agv_battery()
        self.update_agv_mode()

        for actor in self.dynamic_objects:
            if actor.kind == "agv" and self.agv_mode == "CHARGING":
                actor.status = "CHARGING"
                actor.color = AGV_CHARGING
                actor.message = "Charging"
                continue

            if actor.kind == "agv":
                target_x, target_y = self.get_agv_target()
            else:
                if actor.route:
                    target_x, target_y = actor.route[actor.route_index]
                    if self.distance_between(actor.x, actor.y, target_x, target_y) <= WAYPOINT_REACHED_DISTANCE:
                        actor.route_index = (actor.route_index + 1) % len(actor.route)
                        target_x, target_y = actor.route[actor.route_index]
                else:
                    target_x, target_y = actor.x, actor.y

            intended_dx, intended_dy = normalize(target_x - actor.x, target_y - actor.y)
            safe_dx, safe_dy = self.update_dynamic_actor_safety(actor, intended_dx, intended_dy)
            self.try_move_dynamic_actor(actor, safe_dx, safe_dy)

            if actor.kind == "agv":
                if actor.status == "HIGH RISK":
                    actor.color = AGV_DANGER
                elif actor.status == "WARNING":
                    actor.color = AGV_WARNING
                else:
                    actor.color = AGV_SAFE

    def draw_grid(self):
        for x in range(0, SCREEN_WIDTH, 66):
            arcade.draw_line(x, 0, x, SCREEN_HEIGHT, GRID_COLOR, 1)
        for y in range(0, SCREEN_HEIGHT, 66):
            arcade.draw_line(0, y, SCREEN_WIDTH, y, GRID_COLOR, 1)

    def draw_floor_markings(self):
        arcade.draw_lbwh_rectangle_filled(90, 120, 1500, 120, AISLE_COLOR)
        arcade.draw_lbwh_rectangle_filled(90, 735, 1500, 110, AISLE_COLOR)
        arcade.draw_lbwh_rectangle_filled(440, 120, 120, 725, AISLE_COLOR)
        arcade.draw_lbwh_rectangle_filled(900, 120, 120, 725, AISLE_COLOR)

        arcade.draw_lbwh_rectangle_filled(1060, 400, 390, 190, (224, 226, 220))
        arcade.draw_lbwh_rectangle_outline(1060, 400, 390, 190, PANEL_BORDER, 2)
        arcade.draw_text("PACKING AREA", 1255, 560, (84, 92, 82), 12, bold=True, anchor_x="center")

        arcade.draw_lbwh_rectangle_filled(1340, 90, 220, 120, (220, 224, 216))
        arcade.draw_lbwh_rectangle_outline(1340, 90, 220, 120, PANEL_BORDER, 2)
        arcade.draw_text("LOADING BAY", 1450, 168, (84, 92, 82), 13, bold=True, anchor_x="center")

        for x in range(130, 1510, 94):
            arcade.draw_lbwh_rectangle_filled(x, 173, 42, 8, AISLE_MARKING)
            arcade.draw_lbwh_rectangle_filled(x, 785, 42, 8, AISLE_MARKING)

        for y in range(150, 800, 94):
            arcade.draw_lbwh_rectangle_filled(486, y, 8, 42, AISLE_MARKING)
            arcade.draw_lbwh_rectangle_filled(946, y, 8, 42, AISLE_MARKING)

        arcade.draw_line(70, 100, 1590, 100, WALL_STRIPE, 3)
        arcade.draw_line(70, 860, 1590, 860, WALL_STRIPE, 3)

    def draw_static_object(self, obj):
        left = obj.x - obj.width / 2
        bottom = obj.y - obj.height / 2
        arcade.draw_lbwh_rectangle_filled(left, bottom, obj.width, obj.height, obj.color)
        arcade.draw_lbwh_rectangle_outline(left, bottom, obj.width, obj.height, (55, 55, 55), 2)

        if obj.category == "rack":
            for row in range(2):
                box_y = bottom + 5 + row * 15
                for col in range(2):
                    box_x = left + 12 + col * 28
                    box_color = (170, 115, 65) if row == 0 else (205, 155, 95)
                    arcade.draw_lbwh_rectangle_filled(box_x, box_y, 22, 12, box_color)
                    arcade.draw_lbwh_rectangle_outline(box_x, box_y, 22, 12, (110, 80, 40), 1)
        elif obj.category == "forklift":
            arcade.draw_circle_filled(obj.x - 18, obj.y - 18, 7, (40, 40, 40))
            arcade.draw_circle_filled(obj.x + 18, obj.y - 18, 7, (40, 40, 40))
            arcade.draw_line(obj.x + 20, obj.y - 6, obj.x + 34, obj.y - 6, (70, 70, 70), 3)
            arcade.draw_line(obj.x + 20, obj.y + 6, obj.x + 34, obj.y + 6, (70, 70, 70), 3)
        elif obj.category == "crane":
            arcade.draw_line(obj.x, obj.y + 20, obj.x, obj.y + 72, (70, 80, 100), 4)
            arcade.draw_line(obj.x, obj.y + 72, obj.x + 48, obj.y + 72, (70, 80, 100), 4)
            arcade.draw_circle_filled(obj.x + 48, obj.y + 58, 6, (120, 100, 60))

        if obj.category != "rack":
            arcade.draw_text(obj.category.upper(), obj.x, bottom + obj.height + 8, (78, 82, 76), 8, anchor_x="center")

    def draw_goal(self):
        arcade.draw_lbwh_rectangle_filled(1360, 110, 200, 86, GOAL_FILL)
        arcade.draw_lbwh_rectangle_outline(1360, 110, 200, 86, GOAL_BORDER, 3)
        arcade.draw_circle_outline(self.goal_x, self.goal_y, 24, GOAL_BORDER, 4)
        arcade.draw_circle_filled(self.goal_x, self.goal_y, 8, GOAL_BORDER)
        arcade.draw_text("SHARED GOAL", 1460, 160, GOAL_TEXT, 14, bold=True, anchor_x="center")

    def draw_charging_station(self):
        arcade.draw_lbwh_rectangle_filled(1380, 690, 180, 96, CHARGER_FILL)
        arcade.draw_lbwh_rectangle_outline(1380, 690, 180, 96, CHARGER_BORDER, 3)
        arcade.draw_circle_outline(self.charger_x, self.charger_y, 26, CHARGER_BORDER, 4)
        arcade.draw_circle_filled(self.charger_x, self.charger_y, 8, CHARGER_BORDER)
        if self.agv_mode == "CHARGING":
            pulse_radius = 32 + 8 * math.sin(self.agv_charge_pulse_time)
            arcade.draw_circle_outline(self.charger_x, self.charger_y, pulse_radius, AGV_CHARGING, 4)
        arcade.draw_text("CHARGER", 1470, 760, CHARGER_TEXT, 12, bold=True, anchor_x="center")

    def draw_dynamic_actor(self, actor):
        if actor.kind in ["agv", "human"]:
            arcade.draw_circle_filled(actor.x, actor.y, actor.adaptive_safe_distance, SAFE_ZONE_COLOR)
            arcade.draw_circle_filled(actor.x, actor.y, actor.adaptive_danger_distance, DANGER_ZONE_COLOR)
        arcade.draw_line(actor.x, actor.y, actor.x + actor.vx * 8, actor.y + actor.vy * 8, PREDICTION_COLOR, 2)

        if actor.kind == "agv":
            self.draw_robot_body(actor.x, actor.y, actor.angle, actor.color, 0.92)
        elif actor.kind in ["human", "humanoid"]:
            self.draw_human(actor.x, actor.y, actor.color)
        else:
            self.draw_quadruped(actor.x, actor.y, actor.color)

        arcade.draw_text(actor.kind.upper(), actor.x, actor.y + actor.radius + 16, TEXT_COLOR, 8, bold=True, anchor_x="center")

    def draw_human(self, x, y, shirt_color):
        arcade.draw_ellipse_filled(x, y - 34, 28, 10, (160, 160, 160, 100))
        arcade.draw_arc_filled(x, y + 23, 14, 10, HUMAN_HELMET, 0, 180)
        arcade.draw_circle_filled(x, y + 15, 11, HUMAN_SKIN)
        arcade.draw_lbwh_rectangle_filled(x - 11, y - 12, 22, 28, shirt_color)
        arcade.draw_lbwh_rectangle_outline(x - 11, y - 12, 22, 28, (40, 40, 40), 1)
        arcade.draw_line(x - 10, y + 6, x - 18, y - 8, HUMAN_SKIN, 4)
        arcade.draw_line(x + 10, y + 6, x + 18, y - 8, HUMAN_SKIN, 4)
        arcade.draw_line(x - 5, y - 12, x - 8, y - 34, HUMAN_PANTS, 4)
        arcade.draw_line(x + 5, y - 12, x + 8, y - 34, HUMAN_PANTS, 4)

    def draw_quadruped(self, x, y, color):
        arcade.draw_ellipse_filled(x, y, 36, 20, color)
        arcade.draw_circle_filled(x + 18, y + 6, 8, color)
        for leg_offset in [-12, -4, 6, 14]:
            arcade.draw_line(x + leg_offset, y - 8, x + leg_offset, y - 24, (50, 50, 50), 3)
        arcade.draw_line(x - 18, y + 4, x - 28, y + 10, (50, 50, 50), 2)

    def draw_robot_body(self, x, y, angle, color, radius_scale=1.0):
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        arcade.draw_ellipse_filled(x, y - 26 * radius_scale, 34 * radius_scale, 10 * radius_scale, (150, 150, 150, 90))
        arcade.draw_rect_filled(arcade.XYWH(x, y, 48 * radius_scale, 34 * radius_scale), color, angle)
        arcade.draw_rect_outline(arcade.XYWH(x, y, 48 * radius_scale, 34 * radius_scale), ROBOT_DARK, 2, angle)
        arcade.draw_rect_filled(arcade.XYWH(x, y + 2 * radius_scale, 26 * radius_scale, 16 * radius_scale), ROBOT_TOP, angle)
        for ox, oy in [(-17, -15), (17, -15), (-17, 15), (17, 15)]:
            wx = x + ox * radius_scale * cos_a - oy * radius_scale * sin_a
            wy = y + ox * radius_scale * sin_a + oy * radius_scale * cos_a
            arcade.draw_circle_filled(wx, wy, 4.5 * radius_scale, ROBOT_WHEEL)
        fx = x + 20 * radius_scale * cos_a
        fy = y + 20 * radius_scale * sin_a
        arcade.draw_circle_filled(fx, fy, 5 * radius_scale, ROBOT_SENSOR)

    def draw_robot_path(self):
        if len(self.robot_path_trace) < 2:
            return
        for index in range(1, len(self.robot_path_trace)):
            x1, y1 = self.robot_path_trace[index - 1]
            x2, y2 = self.robot_path_trace[index]
            arcade.draw_line(x1, y1, x2, y2, PATH_COLOR, 2)

    def draw_info_panel(self):
        panel_x = 18
        panel_y = SCREEN_HEIGHT - 210
        panel_w = 520
        panel_h = 170

        arcade.draw_lbwh_rectangle_filled(panel_x, panel_y, panel_w, panel_h, PANEL_COLOR)
        arcade.draw_lbwh_rectangle_outline(panel_x, panel_y, panel_w, panel_h, PANEL_BORDER, 2)

        current_time = self.completion_time if self.goal_reached else (time.time() - self.start_time)
        arcade.draw_text("Safety Dashboard", panel_x + 14, SCREEN_HEIGHT - 42, TEXT_COLOR, 14, bold=True)
        arcade.draw_text(f"{current_time:.1f}s", panel_x + 445, SCREEN_HEIGHT - 42, TEXT_COLOR, 10)

        agv_mode_short = {
            "WORKING": "WORKING",
            "GOING_TO_CHARGE": "TO CHARGE",
            "CHARGING": "CHARGING",
            "RETURNING": "RETURNING",
        }.get(self.agv_mode, self.agv_mode)

        row_y = SCREEN_HEIGHT - 72
        arcade.draw_text(f"Mode: {'RL' if self.rl_enabled else 'Manual'}", panel_x + 14, row_y, INFO_COLOR, 10, bold=True)
        arcade.draw_text(self.rl_mode_name, panel_x + 120, row_y, TEXT_COLOR, 10)

        row_y -= 24
        arcade.draw_text(f"Robot: {self.robot_status}", panel_x + 14, row_y, TEXT_COLOR, 10)
        arcade.draw_text(f"Behavior: {self.robot_behavior}", panel_x + 170, row_y, TEXT_COLOR, 10)
        arcade.draw_text(f"Speed: {self.robot_speed:.1f}", panel_x + 330, row_y, TEXT_COLOR, 10)

        row_y -= 22
        arcade.draw_text(f"Action: {self.last_action_label}", panel_x + 14, row_y, TEXT_COLOR, 10)
        arcade.draw_text(f"Reward: {self.current_reward:.2f}", panel_x + 170, row_y, TEXT_COLOR, 10)

        row_y -= 22
        message = self.robot_message if time.time() <= self.robot_message_until else ""
        arcade.draw_text(f"Alert: {message or 'None'}", panel_x + 14, row_y, self.robot_message_color if message else TEXT_COLOR, 10)

        row_y -= 22
        arcade.draw_text(f"AGV: {agv_mode_short}", panel_x + 14, row_y, TEXT_COLOR, 10)
        arcade.draw_text(f"Battery: {self.agv_battery:.0f}%", panel_x + 145, row_y, TEXT_COLOR, 10)
        arcade.draw_text(f"State: {self.agv_actor.status}", panel_x + 265, row_y, TEXT_COLOR, 10)

        row_y -= 22
        arcade.draw_text("Metrics", panel_x + 14, row_y, INFO_COLOR, 11, bold=True)

        row_y -= 22
        arcade.draw_text(f"Collisions {self.metrics.collisions}", panel_x + 14, row_y, TEXT_COLOR, 10)
        arcade.draw_text(f"Near misses {self.metrics.near_misses}", panel_x + 130, row_y, TEXT_COLOR, 10)
        arcade.draw_text(f"Min clr {self.metrics.min_clearance:.1f}", panel_x + 280, row_y, TEXT_COLOR, 10)
        arcade.draw_text(f"Reward {self.metrics.average_reward:.2f}", panel_x + 395, row_y, TEXT_COLOR, 10)

        row_y -= 20
        arcade.draw_text(f"Avg clr {self.metrics.avg_clearance:.1f}", panel_x + 14, row_y, TEXT_COLOR, 10)
        arcade.draw_text(f"RMSE {self.metrics.rmse:.1f}", panel_x + 130, row_y, TEXT_COLOR, 10)
        arcade.draw_text(f"MAE {self.metrics.mae:.1f}", panel_x + 235, row_y, TEXT_COLOR, 10)
        arcade.draw_text(f"AGV {AGV_SPEED_RANGE[0]:.1f}-{AGV_SPEED_RANGE[1]:.1f}", panel_x + 330, row_y, TEXT_COLOR, 10)

        row_y -= 18
        arcade.draw_text("Arrows = manual | R = toggle mode", panel_x + 14, row_y, (70, 76, 88), 9)

    def draw_requirement_panel(self):
        panel_w = 300
        panel_h = 110
        panel_x = 18
        panel_y = 600
        arcade.draw_lbwh_rectangle_filled(panel_x, panel_y, panel_w, panel_h, PANEL_COLOR)
        arcade.draw_lbwh_rectangle_outline(panel_x, panel_y, panel_w, panel_h, PANEL_BORDER, 2)
        arcade.draw_text("Summary", panel_x + 14, panel_y + panel_h - 24, INFO_COLOR, 13, bold=True)

        y = panel_y + panel_h - 44
        for line in self.dashboard_lines:
            wrapped_lines = wrap_text_lines(line, 38)
            for wrapped_line in wrapped_lines:
                arcade.draw_text(wrapped_line, panel_x + 14, y, TEXT_COLOR, 8)
                y -= 14
            y -= 2

    def on_draw(self):
        self.clear()
        self.draw_grid()
        self.draw_floor_markings()

        for obj in self.static_objects:
            self.draw_static_object(obj)

        self.draw_goal()
        self.draw_charging_station()

        for actor in self.dynamic_objects:
            self.draw_dynamic_actor(actor)

        self.draw_robot_body(self.robot_x, self.robot_y, self.robot_angle, self.robot_color, 1.0)
        arcade.draw_text("ROBOT", self.robot_x, self.robot_y + 34, arcade.color.BLACK, 9, bold=True, anchor_x="center")
        self.draw_info_panel()
        self.draw_requirement_panel()

    def on_update(self, delta_time):
        self.update_dynamic_actors()

        if self.rl_enabled:
            self.rl_control_step()
        else:
            intended_dx, intended_dy = self.build_manual_input_vector()
            if intended_dx != 0 or intended_dy != 0:
                # Manual override should allow the driver to move again after a red stop.
                self.robot_speed = ROBOT_BASE_SPEED
                safe_dx, safe_dy = self.build_safety_vector_for_robot(intended_dx, intended_dy)
                moved = self.try_move_robot(safe_dx, safe_dy)
                if not moved:
                    self.try_move_robot(intended_dx, intended_dy)

        self.update_robot_risk_state()
        self.check_goal()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.LEFT:
            if self.rl_enabled:
                self.rl_enabled = False
                self.set_robot_message("Manual control enabled", INFO_COLOR)
            self.left_pressed = True
        elif key == arcade.key.RIGHT:
            if self.rl_enabled:
                self.rl_enabled = False
                self.set_robot_message("Manual control enabled", INFO_COLOR)
            self.right_pressed = True
        elif key == arcade.key.UP:
            if self.rl_enabled:
                self.rl_enabled = False
                self.set_robot_message("Manual control enabled", INFO_COLOR)
            self.up_pressed = True
        elif key == arcade.key.DOWN:
            if self.rl_enabled:
                self.rl_enabled = False
                self.set_robot_message("Manual control enabled", INFO_COLOR)
            self.down_pressed = True
        elif key == arcade.key.R:
            self.rl_enabled = not self.rl_enabled
            mode = "RL enabled" if self.rl_enabled else "Manual override"
            self.set_robot_message(mode, INFO_COLOR)

    def on_key_release(self, key, modifiers):
        if key == arcade.key.LEFT:
            self.left_pressed = False
        elif key == arcade.key.RIGHT:
            self.right_pressed = False
        elif key == arcade.key.UP:
            self.up_pressed = False
        elif key == arcade.key.DOWN:
            self.down_pressed = False


def main():
    WarehouseSimulation()
    arcade.run()


if __name__ == "__main__":
    main()
