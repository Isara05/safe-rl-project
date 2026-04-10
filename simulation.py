import arcade
import math
import time

# -----------------------------
# Window settings
# -----------------------------
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1000
SCREEN_TITLE = "Warehouse Simulation - Shared Goal + AGV Charging"

# -----------------------------
# Colors
# -----------------------------
FLOOR_COLOR = (229, 224, 214)
GRID_COLOR = (210, 205, 195)

ROBOT_SAFE = (60, 180, 90)
ROBOT_WARNING = (245, 200, 60)
ROBOT_DANGER = (220, 70, 70)

AGV_SAFE = (70, 130, 255)
AGV_WARNING = (255, 185, 60)
AGV_DANGER = (235, 85, 85)
AGV_CHARGING = (120, 220, 255)

ROBOT_TOP = (220, 230, 240)
ROBOT_DARK = (35, 55, 95)
ROBOT_WHEEL = (35, 35, 35)
ROBOT_SENSOR = (255, 220, 70)

HUMAN_SKIN = (230, 190, 160)
HUMAN_SHIRT_1 = (231, 101, 72)
HUMAN_SHIRT_2 = (72, 162, 115)
HUMAN_PANTS = (60, 70, 90)
HUMAN_HELMET = (245, 210, 60)

SHELF_FILL = (108, 108, 110)
SHELF_BORDER = (70, 70, 75)
BOX_BROWN = (170, 115, 65)
BOX_LIGHT = (205, 155, 95)

GOAL_FILL = (252, 230, 160)
GOAL_BORDER = (225, 170, 40)
GOAL_TEXT = (120, 80, 0)

CHARGER_FILL = (170, 230, 255)
CHARGER_BORDER = (40, 120, 170)
CHARGER_TEXT = (20, 70, 110)

SAFE_ZONE_COLOR = (255, 120, 120, 45)
DANGER_ZONE_COLOR = (255, 60, 60, 70)

PANEL_COLOR = (248, 247, 242)
PANEL_BORDER = (185, 182, 172)
TEXT_COLOR = arcade.color.BLACK
WARNING_COLOR = arcade.color.ORANGE
SUCCESS_COLOR = arcade.color.DARK_GREEN
COLLISION_COLOR = arcade.color.RED
INFO_COLOR = arcade.color.DARK_BLUE

# -----------------------------
# Settings
# -----------------------------
ROBOT_RADIUS = 22
AGV_RADIUS = 20
HUMAN_RADIUS = 18

SAFE_DISTANCE = 150       # 1.5m
DANGER_DISTANCE = 110

ROBOT_BASE_SPEED = 4.6
ROBOT_SLOW_SPEED = 2.0
ROBOT_DANGER_SPEED = 0.9

AGV_BASE_SPEED = 3.0
AGV_SLOW_SPEED = 1.5
AGV_DANGER_SPEED = 0.75

HUMAN_SPEED = 1.0
GOAL_REACHED_DISTANCE = 30
CHARGING_REACHED_DISTANCE = 28
WAYPOINT_REACHED_DISTANCE = 32


def normalize(dx, dy):
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return 0, 0
    return dx / length, dy / length


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


class WarehouseSimulation(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(FLOOR_COLOR)

        # Main robot
        self.robot_x = 120
        self.robot_y = 120
        self.robot_angle = 0
        self.robot_color = ROBOT_SAFE
        self.robot_status = "SAFE"
        self.robot_speed = ROBOT_BASE_SPEED
        self.robot_collision_message = ""
        self.robot_proximity_message = ""

        # Shared goal
        self.goal_x = 1410
        self.goal_y = 110
        self.goal_reached = False
        self.completion_time = None

        # AGV
        self.agv_x = 620
        self.agv_y = 160
        self.agv_angle = 0
        self.agv_color = AGV_SAFE
        self.agv_status = "SAFE"
        self.agv_speed = AGV_BASE_SPEED
        self.agv_collision_message = ""
        self.agv_proximity_message = ""
        self.agv_charge_pulse_time = 0.0

        # Charging point
        self.charger_x = 140
        self.charger_y = 900

        self.agv_battery = 100.0
        self.agv_low_battery_threshold = 20.0
        self.agv_resume_battery_threshold = 100.0
        self.agv_drain_rate = 0.05
        self.agv_charge_rate = 0.45
        self.agv_mode = "WORKING"  # WORKING, GOING_TO_CHARGE, CHARGING, RETURNING

        # AGV route
        self.agv_demo_waypoints = [
            (1220, 160),
            (1220, 420),
            (980, 420),
            (980, 720),
            (1320, 720),
            (620, 820),
            (620, 520),
            (1120, 250),
        ]
        self.agv_waypoint_index = 0

        # Humans
        self.humans = [
            {"x": 430, "y": 760, "dx": HUMAN_SPEED, "dy": 0, "shirt": HUMAN_SHIRT_1},
            {"x": 1090, "y": 320, "dx": 0, "dy": HUMAN_SPEED, "shirt": HUMAN_SHIRT_2},
        ]

        # Smaller obstacles with wider spacing
        self.obstacles = [
            (260, 820, 90, 36),
            (260, 420, 90, 36),

            (760, 820, 90, 36),
            (760, 420, 90, 36),

            (1260, 820, 90, 36),
            (1260, 420, 90, 36),
        ]

        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False

        self.start_time = time.time()

    # -----------------------------
    # Utility
    # -----------------------------
    def distance_between(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def position_blocked(self, x, y, radius):
        if not (radius <= x <= SCREEN_WIDTH - radius):
            return "wall"
        if not (radius <= y <= SCREEN_HEIGHT - radius):
            return "wall"

        for obstacle in self.obstacles:
            if circle_rect_collision(x, y, radius, obstacle):
                return "obstacle"

        return None

    def nearest_human_clearance(self, x, y, radius):
        nearest = min(self.distance_between(x, y, h["x"], h["y"]) for h in self.humans)
        return nearest - (radius + HUMAN_RADIUS)

    def nearest_obstacle_distance(self, x, y, radius):
        min_dist = float("inf")
        for obstacle in self.obstacles:
            edge_dist = distance_circle_to_rect_edge(x, y, obstacle)
            clearance = max(0, edge_dist - radius)
            min_dist = min(min_dist, clearance)
        return min_dist

    def robot_clearance_from_point(self, x, y, radius):
        return self.distance_between(x, y, self.robot_x, self.robot_y) - (radius + ROBOT_RADIUS)

    def human_position_valid(self, test_x, test_y, current_index):
        if test_x < HUMAN_RADIUS or test_x > SCREEN_WIDTH - HUMAN_RADIUS:
            return False
        if test_y < HUMAN_RADIUS or test_y > SCREEN_HEIGHT - HUMAN_RADIUS:
            return False

        for obstacle in self.obstacles:
            if circle_rect_collision(test_x, test_y, HUMAN_RADIUS, obstacle):
                return False

        for i, other in enumerate(self.humans):
            if i == current_index:
                continue
            d = self.distance_between(test_x, test_y, other["x"], other["y"])
            if d < (HUMAN_RADIUS * 2 + 6):
                return False

        return True

    # -----------------------------
    # Human movement
    # -----------------------------
    def move_humans(self):
        human1 = self.humans[0]
        next_x1 = human1["x"] + human1["dx"]
        if self.human_position_valid(next_x1, human1["y"], 0):
            human1["x"] = next_x1
        else:
            human1["dx"] *= -1

        human2 = self.humans[1]
        next_y2 = human2["y"] + human2["dy"]
        if self.human_position_valid(human2["x"], next_y2, 1):
            human2["y"] = next_y2
        else:
            human2["dy"] *= -1

    # -----------------------------
    # Manual robot
    # -----------------------------
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

    def update_robot_risk_state(self):
        human_clearance = self.nearest_human_clearance(self.robot_x, self.robot_y, ROBOT_RADIUS)
        obstacle_clearance = self.nearest_obstacle_distance(self.robot_x, self.robot_y, ROBOT_RADIUS)
        nearest_clearance = min(human_clearance, obstacle_clearance)

        if nearest_clearance < DANGER_DISTANCE:
            self.robot_status = "HIGH RISK"
            self.robot_color = ROBOT_DANGER
            self.robot_speed = ROBOT_DANGER_SPEED
        elif nearest_clearance <= SAFE_DISTANCE:
            self.robot_status = "WARNING"
            self.robot_color = ROBOT_WARNING
            self.robot_speed = ROBOT_SLOW_SPEED
        else:
            self.robot_status = "SAFE"
            self.robot_color = ROBOT_SAFE
            self.robot_speed = ROBOT_BASE_SPEED

    def update_robot_proximity_warning(self):
        human_clearance = self.nearest_human_clearance(self.robot_x, self.robot_y, ROBOT_RADIUS)
        obstacle_clearance = self.nearest_obstacle_distance(self.robot_x, self.robot_y, ROBOT_RADIUS)

        if human_clearance < DANGER_DISTANCE and human_clearance <= obstacle_clearance:
            self.robot_proximity_message = "High risk: human nearby"
        elif obstacle_clearance < DANGER_DISTANCE:
            self.robot_proximity_message = "High risk: obstacle nearby"
        elif human_clearance <= SAFE_DISTANCE and human_clearance <= obstacle_clearance:
            self.robot_proximity_message = "Too close to human"
        elif obstacle_clearance <= SAFE_DISTANCE:
            self.robot_proximity_message = "Too close to obstacle"
        else:
            self.robot_proximity_message = ""

    def build_safety_vector_for_robot(self, intended_dx, intended_dy):
        final_dx = intended_dx
        final_dy = intended_dy
        active_avoidance = False

        for human in self.humans:
            d = self.distance_between(self.robot_x, self.robot_y, human["x"], human["y"])
            if d < SAFE_DISTANCE:
                away_dx = self.robot_x - human["x"]
                away_dy = self.robot_y - human["y"]
                away_dx, away_dy = normalize(away_dx, away_dy)

                side_dx = -away_dy
                side_dy = away_dx

                dot = side_dx * intended_dx + side_dy * intended_dy
                if dot < 0:
                    side_dx *= -1
                    side_dy *= -1

                if d < DANGER_DISTANCE:
                    final_dx += away_dx * 2.1 + side_dx * 1.2
                    final_dy += away_dy * 2.1 + side_dy * 1.2
                else:
                    final_dx += away_dx * 1.1 + side_dx * 0.7
                    final_dy += away_dy * 1.1 + side_dy * 0.7

                active_avoidance = True

        if active_avoidance:
            final_dx, final_dy = normalize(final_dx, final_dy)

        return final_dx, final_dy

    def try_move_robot(self, move_dx, move_dy):
        self.robot_collision_message = ""

        if move_dx == 0 and move_dy == 0:
            return

        step = self.robot_speed
        candidates = [
            (self.robot_x + move_dx * step, self.robot_y + move_dy * step),
            (self.robot_x + move_dx * step, self.robot_y),
            (self.robot_x, self.robot_y + move_dy * step),
            (self.robot_x + move_dy * step * 0.8, self.robot_y - move_dx * step * 0.8),
            (self.robot_x - move_dy * step * 0.8, self.robot_y + move_dx * step * 0.8),
            (self.robot_x + move_dx * step * 0.5, self.robot_y + move_dy * step * 0.5),
        ]

        for nx, ny in candidates:
            blocked_type = self.position_blocked(nx, ny, ROBOT_RADIUS)
            if blocked_type == "obstacle":
                self.robot_collision_message = "Obstacle collision"
                continue
            elif blocked_type == "wall":
                continue

            safe = True
            for human in self.humans:
                d = self.distance_between(nx, ny, human["x"], human["y"])
                if d < (ROBOT_RADIUS + HUMAN_RADIUS + 2):
                    self.robot_collision_message = "Human collision"
                    safe = False
                    break

            if safe:
                self.robot_angle = math.degrees(math.atan2(ny - self.robot_y, nx - self.robot_x))
                self.robot_x = nx
                self.robot_y = ny
                return

    def check_goal(self):
        d = self.distance_between(self.robot_x, self.robot_y, self.goal_x, self.goal_y)
        if d < GOAL_REACHED_DISTANCE:
            self.goal_reached = True
            self.completion_time = time.time() - self.start_time

    # -----------------------------
    # AGV logic
    # -----------------------------
    def update_agv_battery(self):
        if self.agv_mode == "CHARGING":
            self.agv_battery += self.agv_charge_rate
            if self.agv_battery > 100:
                self.agv_battery = 100

            self.agv_x = self.charger_x
            self.agv_y = self.charger_y
            self.agv_speed = 0
            self.agv_charge_pulse_time += 0.08
        else:
            self.agv_battery -= self.agv_drain_rate
            if self.agv_battery < 0:
                self.agv_battery = 0

    def update_agv_mode(self):
        dist_to_charger = self.distance_between(self.agv_x, self.agv_y, self.charger_x, self.charger_y)

        if self.agv_mode in ["WORKING", "RETURNING"] and self.agv_battery <= self.agv_low_battery_threshold:
            self.agv_mode = "GOING_TO_CHARGE"

        if self.agv_mode == "GOING_TO_CHARGE" and dist_to_charger <= CHARGING_REACHED_DISTANCE:
            self.agv_mode = "CHARGING"
            self.agv_x = self.charger_x
            self.agv_y = self.charger_y
            self.agv_angle = 0

        if self.agv_mode == "CHARGING" and self.agv_battery >= self.agv_resume_battery_threshold:
            self.agv_battery = 100.0
            self.agv_mode = "RETURNING"

    def update_agv_route_progress(self):
        if self.agv_mode not in ["WORKING", "RETURNING"]:
            return

        target_x, target_y = self.agv_demo_waypoints[self.agv_waypoint_index]
        d = self.distance_between(self.agv_x, self.agv_y, target_x, target_y)

        if d <= WAYPOINT_REACHED_DISTANCE:
            self.agv_waypoint_index = (self.agv_waypoint_index + 1) % len(self.agv_demo_waypoints)

    def get_current_agv_target(self):
        if self.agv_mode in ["GOING_TO_CHARGE", "CHARGING"]:
            return self.charger_x, self.charger_y
        return self.agv_demo_waypoints[self.agv_waypoint_index]

    def update_agv_risk_state(self):
        if self.agv_mode == "CHARGING":
            self.agv_status = "CHARGING"
            self.agv_color = AGV_CHARGING
            self.agv_speed = 0
            return

        human_clearance = self.nearest_human_clearance(self.agv_x, self.agv_y, AGV_RADIUS)
        obstacle_clearance = self.nearest_obstacle_distance(self.agv_x, self.agv_y, AGV_RADIUS)
        robot_clearance = self.robot_clearance_from_point(self.agv_x, self.agv_y, AGV_RADIUS)
        nearest_clearance = min(human_clearance, obstacle_clearance, robot_clearance)

        if nearest_clearance < DANGER_DISTANCE:
            self.agv_status = "HIGH RISK"
            self.agv_color = AGV_DANGER
            self.agv_speed = AGV_DANGER_SPEED
        elif nearest_clearance <= SAFE_DISTANCE:
            self.agv_status = "WARNING"
            self.agv_color = AGV_WARNING
            self.agv_speed = AGV_SLOW_SPEED
        else:
            self.agv_status = "SAFE"
            self.agv_color = AGV_SAFE
            self.agv_speed = AGV_BASE_SPEED

    def update_agv_proximity_warning(self):
        if self.agv_mode == "CHARGING":
            self.agv_proximity_message = "Charging at station"
            return

        human_clearance = self.nearest_human_clearance(self.agv_x, self.agv_y, AGV_RADIUS)
        obstacle_clearance = self.nearest_obstacle_distance(self.agv_x, self.agv_y, AGV_RADIUS)
        robot_clearance = self.robot_clearance_from_point(self.agv_x, self.agv_y, AGV_RADIUS)

        nearest_type = min(
            [
                ("human", human_clearance),
                ("obstacle", obstacle_clearance),
                ("robot", robot_clearance),
            ],
            key=lambda item: item[1]
        )

        nearest_name, nearest_value = nearest_type

        if nearest_value < DANGER_DISTANCE:
            self.agv_proximity_message = f"High risk: {nearest_name} nearby"
        elif nearest_value <= SAFE_DISTANCE:
            self.agv_proximity_message = f"Too close to {nearest_name}"
        else:
            self.agv_proximity_message = ""

    def build_safety_vector_for_agv(self, intended_dx, intended_dy):
        final_dx = intended_dx
        final_dy = intended_dy
        active_avoidance = False

        # Avoid humans
        for human in self.humans:
            d = self.distance_between(self.agv_x, self.agv_y, human["x"], human["y"])
            if d < SAFE_DISTANCE:
                away_dx = self.agv_x - human["x"]
                away_dy = self.agv_y - human["y"]
                away_dx, away_dy = normalize(away_dx, away_dy)

                side_dx = -away_dy
                side_dy = away_dx

                dot = side_dx * intended_dx + side_dy * intended_dy
                if dot < 0:
                    side_dx *= -1
                    side_dy *= -1

                final_dx += away_dx * 1.8 + side_dx * 1.0
                final_dy += away_dy * 1.8 + side_dy * 1.0
                active_avoidance = True

        # Avoid main robot
        d_robot = self.distance_between(self.agv_x, self.agv_y, self.robot_x, self.robot_y)
        if d_robot < SAFE_DISTANCE:
            away_dx = self.agv_x - self.robot_x
            away_dy = self.agv_y - self.robot_y
            away_dx, away_dy = normalize(away_dx, away_dy)

            side_dx = -away_dy
            side_dy = away_dx

            dot = side_dx * intended_dx + side_dy * intended_dy
            if dot < 0:
                side_dx *= -1
                side_dy *= -1

            final_dx += away_dx * 1.8 + side_dx * 1.0
            final_dy += away_dy * 1.8 + side_dy * 1.0
            active_avoidance = True

        # Avoid obstacles before collision
        look_ahead = 70
        future_x = self.agv_x + intended_dx * look_ahead
        future_y = self.agv_y + intended_dy * look_ahead

        for obstacle in self.obstacles:
            future_clearance = max(0, distance_circle_to_rect_edge(future_x, future_y, obstacle) - AGV_RADIUS)
            if future_clearance < SAFE_DISTANCE:
                ox, oy, _, _ = obstacle
                away_dx = self.agv_x - ox
                away_dy = self.agv_y - oy
                away_dx, away_dy = normalize(away_dx, away_dy)

                side_dx = -away_dy
                side_dy = away_dx

                dot = side_dx * intended_dx + side_dy * intended_dy
                if dot < 0:
                    side_dx *= -1
                    side_dy *= -1

                weight = 2.2 if future_clearance < DANGER_DISTANCE else 1.3
                final_dx += away_dx * weight + side_dx * 1.0
                final_dy += away_dy * weight + side_dy * 1.0
                active_avoidance = True

        if active_avoidance:
            final_dx, final_dy = normalize(final_dx, final_dy)

        return final_dx, final_dy

    def try_move_agv(self, move_dx, move_dy):
        self.agv_collision_message = ""

        if move_dx == 0 and move_dy == 0 or self.agv_mode == "CHARGING":
            return

        step = self.agv_speed
        candidates = [
            (self.agv_x + move_dx * step, self.agv_y + move_dy * step),
            (self.agv_x + move_dx * step, self.agv_y),
            (self.agv_x, self.agv_y + move_dy * step),
            (self.agv_x + move_dy * step * 0.8, self.agv_y - move_dx * step * 0.8),
            (self.agv_x - move_dy * step * 0.8, self.agv_y + move_dx * step * 0.8),
            (self.agv_x + move_dx * step * 0.5, self.agv_y + move_dy * step * 0.5),
        ]

        for nx, ny in candidates:
            blocked_type = self.position_blocked(nx, ny, AGV_RADIUS)
            if blocked_type == "obstacle":
                continue
            elif blocked_type == "wall":
                continue

            safe = True

            for human in self.humans:
                d = self.distance_between(nx, ny, human["x"], human["y"])
                if d < (AGV_RADIUS + HUMAN_RADIUS + 2):
                    safe = False
                    break

            if safe:
                d_robot = self.distance_between(nx, ny, self.robot_x, self.robot_y)
                if d_robot < (AGV_RADIUS + ROBOT_RADIUS + 4):
                    safe = False

            if safe:
                self.agv_angle = math.degrees(math.atan2(ny - self.agv_y, nx - self.agv_x))
                self.agv_x = nx
                self.agv_y = ny
                return

    def update_agv(self):
        self.update_agv_battery()
        self.update_agv_mode()
        self.update_agv_risk_state()
        self.update_agv_proximity_warning()

        if self.agv_mode == "CHARGING":
            return

        self.update_agv_route_progress()

        target_x, target_y = self.get_current_agv_target()
        dx = target_x - self.agv_x
        dy = target_y - self.agv_y
        move_dx, move_dy = normalize(dx, dy)

        safe_dx, safe_dy = self.build_safety_vector_for_agv(move_dx, move_dy)
        self.try_move_agv(safe_dx, safe_dy)

    # -----------------------------
    # Drawing
    # -----------------------------
    def draw_grid(self):
        for x in range(0, SCREEN_WIDTH, 50):
            arcade.draw_line(x, 0, x, SCREEN_HEIGHT, GRID_COLOR, 1)
        for y in range(0, SCREEN_HEIGHT, 50):
            arcade.draw_line(0, y, SCREEN_WIDTH, y, GRID_COLOR, 1)

    def draw_floor_markings(self):
        arcade.draw_lbwh_rectangle_filled(70, 60, 1380, 95, (214, 210, 200))
        arcade.draw_lbwh_rectangle_filled(70, 190, 95, 720, (214, 210, 200))

        for x in range(100, 1430, 55):
            arcade.draw_lbwh_rectangle_filled(x, 110, 28, 8, (250, 240, 120))

        for y in range(220, 900, 55):
            arcade.draw_lbwh_rectangle_filled(120, y, 8, 28, (250, 240, 120))

    def draw_obstacles(self):
        for ox, oy, ow, oh in self.obstacles:
            left = ox - ow / 2
            bottom = oy - oh / 2

            arcade.draw_lbwh_rectangle_filled(left, bottom, ow, oh, SHELF_FILL)
            arcade.draw_lbwh_rectangle_outline(left, bottom, ow, oh, SHELF_BORDER, 2)

            box_y = bottom + 5
            for row in range(2):
                box_x = left + 12
                for _ in range(2):
                    arcade.draw_lbwh_rectangle_filled(
                        box_x, box_y, 22, 12,
                        BOX_BROWN if row == 0 else BOX_LIGHT
                    )
                    arcade.draw_lbwh_rectangle_outline(box_x, box_y, 22, 12, (110, 80, 40), 1)
                    box_x += 28
                box_y += 15

    def draw_goal(self):
        arcade.draw_lbwh_rectangle_filled(1325, 40, 180, 145, GOAL_FILL)
        arcade.draw_lbwh_rectangle_outline(1325, 40, 180, 145, GOAL_BORDER, 3)
        arcade.draw_circle_outline(self.goal_x, self.goal_y, 24, GOAL_BORDER, 4)
        arcade.draw_circle_filled(self.goal_x, self.goal_y, 8, GOAL_BORDER)
        arcade.draw_text("SHARED", 1370, 145, GOAL_TEXT, 18, bold=True)
        arcade.draw_text("GOAL", 1390, 118, GOAL_TEXT, 16, bold=True)

    def draw_charging_station(self):
        arcade.draw_lbwh_rectangle_filled(40, 820, 210, 130, CHARGER_FILL)
        arcade.draw_lbwh_rectangle_outline(40, 820, 210, 130, CHARGER_BORDER, 3)

        arcade.draw_circle_outline(self.charger_x, self.charger_y, 26, CHARGER_BORDER, 4)
        arcade.draw_circle_filled(self.charger_x, self.charger_y, 8, CHARGER_BORDER)

        if self.agv_mode == "CHARGING":
            pulse_radius = 32 + 8 * math.sin(self.agv_charge_pulse_time)
            arcade.draw_circle_outline(
                self.charger_x,
                self.charger_y,
                pulse_radius,
                AGV_CHARGING,
                4
            )

        arcade.draw_text("AGV", 105, 922, CHARGER_TEXT, 18, bold=True)
        arcade.draw_text("CHARGER", 78, 895, CHARGER_TEXT, 18, bold=True)

    def draw_human(self, x, y, shirt_color):
        arcade.draw_circle_filled(x, y, SAFE_DISTANCE, SAFE_ZONE_COLOR)
        arcade.draw_circle_filled(x, y, DANGER_DISTANCE, DANGER_ZONE_COLOR)

        arcade.draw_ellipse_filled(x, y - 34, 28, 10, (160, 160, 160, 100))
        arcade.draw_arc_filled(x, y + 23, 14, 10, HUMAN_HELMET, 0, 180)
        arcade.draw_circle_filled(x, y + 15, 11, HUMAN_SKIN)

        arcade.draw_lbwh_rectangle_filled(x - 11, y - 12, 22, 28, shirt_color)
        arcade.draw_lbwh_rectangle_outline(x - 11, y - 12, 22, 28, (40, 40, 40), 1)

        arcade.draw_line(x - 10, y + 6, x - 18, y - 8, HUMAN_SKIN, 4)
        arcade.draw_line(x + 10, y + 6, x + 18, y - 8, HUMAN_SKIN, 4)
        arcade.draw_line(x - 5, y - 12, x - 8, y - 34, HUMAN_PANTS, 4)
        arcade.draw_line(x + 5, y - 12, x + 8, y - 34, HUMAN_PANTS, 4)

    def draw_robot_body(self, x, y, angle, color, radius_scale=1.0):
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        arcade.draw_ellipse_filled(x, y - 26 * radius_scale, 34 * radius_scale, 10 * radius_scale, (150, 150, 150, 90))

        arcade.draw_rect_filled(
            arcade.XYWH(x, y, 48 * radius_scale, 34 * radius_scale),
            color,
            angle
        )
        arcade.draw_rect_outline(
            arcade.XYWH(x, y, 48 * radius_scale, 34 * radius_scale),
            ROBOT_DARK,
            2,
            angle
        )

        arcade.draw_rect_filled(
            arcade.XYWH(x, y + 2 * radius_scale, 26 * radius_scale, 16 * radius_scale),
            ROBOT_TOP,
            angle
        )

        wheel_offsets = [(-17, -15), (17, -15), (-17, 15), (17, 15)]
        for ox, oy in wheel_offsets:
            wx = x + ox * radius_scale * cos_a - oy * radius_scale * sin_a
            wy = y + ox * radius_scale * sin_a + oy * radius_scale * cos_a
            arcade.draw_circle_filled(wx, wy, 4.5 * radius_scale, ROBOT_WHEEL)

        fx = x + 20 * radius_scale * cos_a
        fy = y + 20 * radius_scale * sin_a
        arcade.draw_circle_filled(fx, fy, 5 * radius_scale, ROBOT_SENSOR)

    def draw_info_panel(self):
        panel_x = 15
        panel_y = SCREEN_HEIGHT - 295
        panel_w = 560
        panel_h = 260

        arcade.draw_lbwh_rectangle_filled(panel_x, panel_y, panel_w, panel_h, PANEL_COLOR)
        arcade.draw_lbwh_rectangle_outline(panel_x, panel_y, panel_w, panel_h, PANEL_BORDER, 2)

        current_time = self.completion_time if self.goal_reached else (time.time() - self.start_time)

        arcade.draw_text("Warehouse Safety Dashboard", 28, SCREEN_HEIGHT - 50, TEXT_COLOR, 14, bold=True)
        arcade.draw_text(f"Time: {current_time:.2f}s", 28, SCREEN_HEIGHT - 78, TEXT_COLOR, 11)

        arcade.draw_text("MAIN ROBOT", 28, SCREEN_HEIGHT - 108, INFO_COLOR, 12, bold=True)
        arcade.draw_text(f"State: {self.robot_status}", 28, SCREEN_HEIGHT - 130, TEXT_COLOR, 11)
        arcade.draw_text(f"Speed: {self.robot_speed:.1f}", 28, SCREEN_HEIGHT - 150, TEXT_COLOR, 11)

        if self.goal_reached:
            arcade.draw_text("Robot reached goal", 250, SCREEN_HEIGHT - 130, SUCCESS_COLOR, 11, bold=True)
        elif self.robot_collision_message:
            arcade.draw_text(self.robot_collision_message, 250, SCREEN_HEIGHT - 130, COLLISION_COLOR, 11, bold=True)
        elif self.robot_proximity_message:
            arcade.draw_text(self.robot_proximity_message, 250, SCREEN_HEIGHT - 130, WARNING_COLOR, 11, bold=True)

        arcade.draw_text("AGV", 28, SCREEN_HEIGHT - 210, INFO_COLOR, 12, bold=True)
        arcade.draw_text(f"Mode: {self.agv_mode}", 28, SCREEN_HEIGHT - 232, TEXT_COLOR, 11)
        arcade.draw_text(f"Battery: {self.agv_battery:.1f}%", 28, SCREEN_HEIGHT - 252, TEXT_COLOR, 11)
        arcade.draw_text(f"State: {self.agv_status}", 190, SCREEN_HEIGHT - 232, TEXT_COLOR, 11)
        arcade.draw_text(f"Speed: {self.agv_speed:.1f}", 190, SCREEN_HEIGHT - 252, TEXT_COLOR, 11)

        if self.agv_collision_message:
            arcade.draw_text(self.agv_collision_message, 350, SCREEN_HEIGHT - 232, COLLISION_COLOR, 11, bold=True)
        elif self.agv_proximity_message:
            arcade.draw_text(self.agv_proximity_message, 350, SCREEN_HEIGHT - 232, WARNING_COLOR, 11, bold=True)

        if self.agv_mode == "CHARGING":
            arcade.draw_text("AGV charging: 20% -> 100%", 28, SCREEN_HEIGHT - 278, SUCCESS_COLOR, 11, bold=True)

    # -----------------------------
    # Main draw/update
    # -----------------------------
    def on_draw(self):
        self.clear()
        self.draw_grid()
        self.draw_floor_markings()
        self.draw_obstacles()
        self.draw_goal()
        self.draw_charging_station()

        for human in self.humans:
            self.draw_human(human["x"], human["y"], human["shirt"])

        self.draw_robot_body(self.robot_x, self.robot_y, self.robot_angle, self.robot_color, 1.0)
        self.draw_robot_body(self.agv_x, self.agv_y, self.agv_angle, self.agv_color, 0.92)

        if self.agv_mode == "CHARGING":
            pulse_radius = 28 + 6 * math.sin(self.agv_charge_pulse_time)
            arcade.draw_circle_outline(self.agv_x, self.agv_y, pulse_radius, AGV_CHARGING, 3)

        arcade.draw_text("ROBOT", self.robot_x - 28, self.robot_y + 34, arcade.color.BLACK, 10, bold=True)
        arcade.draw_text("AGV", self.agv_x - 16, self.agv_y + 32, arcade.color.BLACK, 10, bold=True)

        self.draw_info_panel()

    def on_update(self, delta_time: float):
        self.move_humans()

        self.robot_proximity_message = ""
        self.update_robot_risk_state()

        intended_dx, intended_dy = self.build_manual_input_vector()
        if intended_dx != 0 or intended_dy != 0:
            safe_dx, safe_dy = self.build_safety_vector_for_robot(intended_dx, intended_dy)
            self.try_move_robot(safe_dx, safe_dy)

        self.update_robot_risk_state()
        self.update_robot_proximity_warning()
        self.check_goal()

        self.update_agv()

    # -----------------------------
    # Keyboard
    # -----------------------------
    def on_key_press(self, key, modifiers):
        if key == arcade.key.LEFT:
            self.left_pressed = True
        elif key == arcade.key.RIGHT:
            self.right_pressed = True
        elif key == arcade.key.UP:
            self.up_pressed = True
        elif key == arcade.key.DOWN:
            self.down_pressed = True

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