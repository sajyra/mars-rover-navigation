import gymnasium
from gymnasium import spaces
import numpy as np

EMPTY = 0
CRATER = 1
CLIFF = 2
OUTSIDE = 3

class MarsRoverEnv(gymnasium.Env):

    def __init__(self, grid_size=20, window_size=5, num_bins=None, max_steps=200, randomize=True):
        super().__init__()

        self.grid_size = grid_size
        self.window_size = window_size
        self.randomize = randomize
        
        self.rover_pos = None
        self.sample_pos = None
        self.lander_pos = None
        self.has_sample = False
        if num_bins:
            self.num_bins = num_bins
        else:
            self.num_bins = max(5, grid_size // 4)
        self.steps = 0
        self.max_steps = max_steps
        self.corners = [
                    (0, 0),
                    (0, self.grid_size - 1),
                    (self.grid_size - 1, 0),
                    (self.grid_size - 1, self.grid_size - 1)
                ]

        # 0: South, 1: North, 2: West, 3: East
        self.action_space = spaces.Discrete(4)

        obs_size = 1 + self.num_bins + ((self.window_size ** 2) * 4)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.grid = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize:
            rover_idx = self.np_random.integers(4)
            self.rover_pos = self.corners[rover_idx]

            lander_idx = (rover_idx + 2) % 4      # opposite corner
            self.lander_pos = self.corners[lander_idx]

            while True:
                r = self.np_random.integers(2, self.grid_size - 2)
                c = self.np_random.integers(2, self.grid_size - 2)

                if (r, c) not in (self.rover_pos, self.lander_pos):
                    self.sample_pos = (r, c)
                    break
        else:
            # fixed static layout, no randomness — for isolating whether
            # PPO can learn navigation at all before adding generalization
            self.rover_pos = (0, 0)
            self.lander_pos = (0, self.grid_size - 1)
            self.sample_pos = (self.grid_size - 1, int(self.grid_size * 0.8))

        self.has_sample = False
        self.steps = 0

        self.grid = self.create_grid()

        return self.get_observation(), {}
    

    def step(self, action):

        row, col = self.rover_pos

        if action == 0:
            self.rover_pos = (row + 1, col)
        elif action == 1:
            self.rover_pos = (row - 1, col)
        elif action == 2:
            self.rover_pos = (row, col - 1)
        elif action == 3:
            self.rover_pos = (row, col + 1)
        
        prev_row = row
        prev_col = col

        row, col = self.rover_pos

        terminated = False
        truncated = False

        reward = 0

        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            reward += -5
            self.rover_pos = (prev_row, prev_col)
    
        elif self.grid[row, col] == CRATER:
            reward += -10
        
        elif self.grid[row, col] == CLIFF:
            reward += -50
            terminated = True
        
        elif self.grid[row, col] == EMPTY:
            reward += -1
        
        self.steps += 1
        if self.steps == self.max_steps:
            truncated = True
        
        if not self.has_sample and self.rover_pos == self.sample_pos:
            self.has_sample = True
            reward += 20
        
        if self.has_sample and self.rover_pos == self.lander_pos:
            reward += 100
            terminated = True
        
        row, col = self.rover_pos
        if self.has_sample:
            target_row, target_col = self.lander_pos
        else:
            target_row, target_col = self.sample_pos

        old_distance = abs((prev_row - target_row)) + abs((prev_col - target_col))
        distance = abs((row - target_row)) + abs((col - target_col))
        reward += 0.2 * (old_distance - distance)
        
        obs = self.get_observation()

        return obs, reward, terminated, truncated, {}


    def create_grid(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        if not self.randomize:
            # no obstacles in static debug mode — isolating pure navigation
            for col in range(20):
                if col not in (9, 10):
                    grid[9, col] = CRATER
            grid[8, 9] = CRATER
            grid[8, 10] = CRATER

            # Wall 2 — blocks the return leg (sample → lander), spans column 15
            # entirely except a 2-cell gap at rows 9-10, same flanking cliffs.
            for row in range(20):
                if row not in (9, 10):
                    grid[row, 15] = CRATER
            grid[9, 14] = CLIFF
            grid[10, 14] = CLIFF

            return grid

        num_craters = self.grid_size // 2
        num_cliffs = self.grid_size // 3

        forbidden = {
            self.rover_pos,
            self.lander_pos,
            self.sample_pos
        }

        placed = 0

        while placed < num_craters:
            r = self.np_random.integers(self.grid_size)
            c = self.np_random.integers(self.grid_size)

            candidate = (r, c)

            if any(self.near(candidate, pos) for pos in forbidden):
                continue

            if grid[r, c] != EMPTY:
                continue

            grid[r, c] = CRATER
            placed += 1

        placed = 0

        while placed < num_cliffs:
            r = self.np_random.integers(self.grid_size)
            c = self.np_random.integers(self.grid_size)

            candidate = (r, c)

            if any(self.near(candidate, pos) for pos in forbidden):
                continue

            if grid[r, c] != EMPTY:
                continue

            grid[r, c] = CLIFF
            placed += 1
        

        return grid
    

    def local_window(self):

        radius = self.window_size // 2

        padded_grid = np.pad(self.grid, radius, mode="constant", constant_values=3)

        row, col = self.rover_pos
        row += radius
        col += radius

        window = padded_grid[row-radius:row+radius+1, col-radius:col+radius+1]

        encoded = np.eye(4)[window].flatten()

        return encoded.astype(np.float32)


    def encode_distance(self):
        if self.has_sample:
            target = self.lander_pos
        else:
            target = self.sample_pos

        target_row, target_col = target
        row, col = self.rover_pos

        distance = abs((row - target_row)) + abs((col - target_col))

        bins = np.zeros(self.num_bins, dtype=np.float32)

        max_distance = 2 * (self.grid_size - 1)
        bin_size = max_distance / self.num_bins
        bin_index = min(int(distance // bin_size), self.num_bins - 1)
        bins[bin_index] = 1

        return bins


    def get_observation(self):
        window = self.local_window()
        distance = self.encode_distance()
        sample_flag = np.array([self.has_sample], dtype=np.float32)

        obs = np.concatenate([window, sample_flag, distance]).astype(np.float32)

        return obs

    def near(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1


        

if __name__ == "__main__":
    env = MarsRoverEnv(grid_size=8, randomize=True)
    for i in range(5):
        env.reset()
        print(f"Rover pos: {env.rover_pos}")
        print(f"Sample pos: {env.sample_pos}")
        print(f"Lander pos: {env.lander_pos}")
        print(f"Grid: {env.grid}")

