import pygame
import sys
from environment import MarsRoverEnv
from networks import PolicyNetwork
import torch

GRID_SIZE = 8
CELL_SIZE = 75
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

COLORS = {
    0: (194, 158, 120),   # EMPTY - tan/terrain
    1: (101, 67, 33),     # CRATER - dark brown
    2: (178, 34, 34),     # CLIFF - firebrick red
}
BACKGROUND = (255, 255, 255)
ROVER_COLOR = (20, 20, 20)      # black
SAMPLE_COLOR = (140, 140, 140)  # grey
LANDER_COLOR = (255, 255, 255)  # white
LANDER_BORDER = (0, 0, 0)       # so a white lander square is visible on light cells


def draw_grid(screen, env):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            terrain = env.grid[row, col]
            color = COLORS[terrain]
            rect = (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)  # cell border


def cell_center(pos):
    row, col = pos
    return (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2)


def draw_rover(screen, pos):
    # circle
    center = cell_center(pos)
    pygame.draw.circle(screen, ROVER_COLOR, center, CELL_SIZE // 3)


def draw_sample(screen, pos):
    # diamond
    cx, cy = cell_center(pos)
    size = CELL_SIZE // 3
    points = [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)]
    pygame.draw.polygon(screen, SAMPLE_COLOR, points)


def draw_lander(screen, pos):
    # square
    cx, cy = cell_center(pos)
    size = CELL_SIZE // 3
    rect = (cx - size, cy - size, size * 2, size * 2)
    pygame.draw.rect(screen, LANDER_COLOR, rect)
    pygame.draw.rect(screen, LANDER_BORDER, rect, 2)


def main():
    pygame.init()

    env = MarsRoverEnv(GRID_SIZE, 5, 5, 150, True)
    policy_net = PolicyNetwork(env.observation_space.shape[0])
    policy_net.load_state_dict(torch.load("src/rover_policy_network.pt"))
    policy_net.eval()

    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Mars Rover Navigation")
    clock = pygame.time.Clock()

    state, _ = env.reset()
    state = torch.as_tensor(state, dtype=torch.float32)
    hidden, cell = torch.zeros(policy_net.hidden_dim), torch.zeros(policy_net.hidden_dim)

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not running:
            break

        with torch.no_grad():
            logits, (hidden, cell) = policy_net(state, (hidden, cell))
            action = torch.argmax(logits)
            next_state, reward, terminated, truncated, info = env.step(action.item())
            state = torch.as_tensor(next_state, dtype=torch.float32)

        done = terminated or truncated
        if done:
            state, _ = env.reset()
            state = torch.as_tensor(state, dtype=torch.float32)
            hidden, cell = torch.zeros(policy_net.hidden_dim), torch.zeros(policy_net.hidden_dim)

        screen.fill(BACKGROUND)
        draw_grid(screen, env)
        if not env.has_sample:
            draw_sample(screen, env.sample_pos)
        draw_lander(screen, env.lander_pos)
        draw_rover(screen, env.rover_pos)

        pygame.display.flip()
        clock.tick(4)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()