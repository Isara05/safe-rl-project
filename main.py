import pygame
import sys
import random
import math

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Robot Simulation with Collision Detection")

# Colors
WHITE = (255, 255, 255)
BLUE = (50, 100, 255)     # Robot
RED = (255, 80, 80)       # Human
YELLOW = (255, 215, 0)    # Goal
BLACK = (40, 40, 40)      # Obstacles
GREEN = (0, 180, 0)
ORANGE = (255, 140, 0)

clock = pygame.time.Clock()

# Robot
robot_x, robot_y = 100, 100
robot_radius = 20
robot_speed = 4

# Human
human_x, human_y = 400, 300
human_radius = 20
human_speed = 2

# Goal
goal_x, goal_y = 700, 500
goal_size = 30

# Obstacles (blocks)
obstacles = [
    pygame.Rect(250, 150, 80, 80),
    pygame.Rect(500, 250, 100, 60),
    pygame.Rect(300, 400, 120, 50)
]

font = pygame.font.SysFont(None, 36)

def draw_star(surface, x, y, size, color):
    pygame.draw.line(surface, color, (x, y - size), (x, y + size), 3)
    pygame.draw.line(surface, color, (x - size, y), (x + size, y), 3)
    pygame.draw.line(surface, color, (x - size // 2, y - size // 2), (x + size // 2, y + size // 2), 3)
    pygame.draw.line(surface, color, (x + size // 2, y - size // 2), (x - size // 2, y + size // 2), 3)

def circle_rect_collision(circle_x, circle_y, radius, rect):
    # Find the closest point on the rectangle to the circle center
    closest_x = max(rect.left, min(circle_x, rect.right))
    closest_y = max(rect.top, min(circle_y, rect.bottom))

    # Calculate distance from circle center to closest point
    distance_x = circle_x - closest_x
    distance_y = circle_y - closest_y

    return (distance_x ** 2 + distance_y ** 2) < (radius ** 2)

def circle_circle_collision(x1, y1, r1, x2, y2, r2):
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance < (r1 + r2)

def move_human():
    global human_x, human_y

    dx = random.choice([-human_speed, 0, human_speed])
    dy = random.choice([-human_speed, 0, human_speed])

    new_x = human_x + dx
    new_y = human_y + dy

    # Keep human inside screen
    if not (human_radius <= new_x <= WIDTH - human_radius):
        new_x = human_x
    if not (human_radius <= new_y <= HEIGHT - human_radius):
        new_y = human_y

    # Prevent human from going through obstacles
    collided_with_obstacle = False
    for obstacle in obstacles:
        if circle_rect_collision(new_x, new_y, human_radius, obstacle):
            collided_with_obstacle = True
            break

    if not collided_with_obstacle:
        human_x = new_x
        human_y = new_y

def robot_reached_goal():
    return abs(robot_x - goal_x) < 25 and abs(robot_y - goal_y) < 25

running = True
collision_message = ""

while running:
    clock.tick(30)
    screen.fill(WHITE)
    collision_message = ""

    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Keyboard movement for robot
    keys = pygame.key.get_pressed()
    new_robot_x, new_robot_y = robot_x, robot_y

    if keys[pygame.K_LEFT]:
        new_robot_x -= robot_speed
    if keys[pygame.K_RIGHT]:
        new_robot_x += robot_speed
    if keys[pygame.K_UP]:
        new_robot_y -= robot_speed
    if keys[pygame.K_DOWN]:
        new_robot_y += robot_speed

    # Check screen boundary collision
    boundary_collision = False
    if not (robot_radius <= new_robot_x <= WIDTH - robot_radius):
        boundary_collision = True
    if not (robot_radius <= new_robot_y <= HEIGHT - robot_radius):
        boundary_collision = True

    # Check obstacle collision
    obstacle_collision = False
    for obstacle in obstacles:
        if circle_rect_collision(new_robot_x, new_robot_y, robot_radius, obstacle):
            obstacle_collision = True
            break

    # Check human collision
    human_collision = circle_circle_collision(
        new_robot_x, new_robot_y, robot_radius,
        human_x, human_y, human_radius
    )

    # Only move robot if no collision
    if not boundary_collision and not obstacle_collision and not human_collision:
        robot_x = new_robot_x
        robot_y = new_robot_y
    else:
        if boundary_collision:
            collision_message = "Boundary collision!"
        elif obstacle_collision:
            collision_message = "Obstacle collision!"
        elif human_collision:
            collision_message = "Human collision!"

    # Move human
    move_human()

    # Draw obstacles
    for obstacle in obstacles:
        pygame.draw.rect(screen, BLACK, obstacle)

    # Draw goal
    draw_star(screen, goal_x, goal_y, goal_size, YELLOW)

    # Draw human
    pygame.draw.circle(screen, RED, (human_x, human_y), human_radius)

    # Draw robot
    pygame.draw.circle(screen, BLUE, (robot_x, robot_y), robot_radius)

    # Show goal reached message
    if robot_reached_goal():
        text = font.render("Goal Reached!", True, GREEN)
        screen.blit(text, (300, 30))

    # Show collision message
    if collision_message:
        text = font.render(collision_message, True, ORANGE)
        screen.blit(text, (280, 70))

    pygame.display.flip()

pygame.quit()
sys.exit()