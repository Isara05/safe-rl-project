import pygame
import sys
import random

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Robot Simulation")

# Colors
WHITE = (255, 255, 255)
BLUE = (50, 100, 255)     # Robot
RED = (255, 80, 80)       # Human
YELLOW = (255, 215, 0)    # Goal
BLACK = (40, 40, 40)      # Obstacles
GREEN = (0, 180, 0)

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

def draw_star(surface, x, y, size, color):
    # Simple star-like shape using lines
    pygame.draw.line(surface, color, (x, y - size), (x, y + size), 3)
    pygame.draw.line(surface, color, (x - size, y), (x + size, y), 3)
    pygame.draw.line(surface, color, (x - size // 2, y - size // 2), (x + size // 2, y + size // 2), 3)
    pygame.draw.line(surface, color, (x + size // 2, y - size // 2), (x - size // 2, y + size // 2), 3)

def move_human():
    global human_x, human_y

    dx = random.choice([-human_speed, 0, human_speed])
    dy = random.choice([-human_speed, 0, human_speed])

    new_x = human_x + dx
    new_y = human_y + dy

    # Keep human inside screen
    if 20 <= new_x <= WIDTH - 20:
        human_x = new_x
    if 20 <= new_y <= HEIGHT - 20:
        human_y = new_y

def robot_reached_goal():
    return abs(robot_x - goal_x) < 25 and abs(robot_y - goal_y) < 25

running = True
while running:
    clock.tick(30)
    screen.fill(WHITE)

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

    # Keep robot inside screen
    if 20 <= new_robot_x <= WIDTH - 20:
        robot_x = new_robot_x
    if 20 <= new_robot_y <= HEIGHT - 20:
        robot_y = new_robot_y

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

    # Check if robot reached goal
    if robot_reached_goal():
        font = pygame.font.SysFont(None, 48)
        text = font.render("Goal Reached!", True, GREEN)
        screen.blit(text, (300, 40))

    pygame.display.flip()

pygame.quit()
sys.exit()
