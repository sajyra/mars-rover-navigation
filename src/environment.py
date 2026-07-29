import gymnasium
from gymnasium import spaces
import numpy as np

EMPTY = 0
CRATER = 1
CLIFF = 2
OUTSIDE = 3

class MarsRoverEnv(gymnasium.Env):

    def __init__(self, grid_size=20, window_size=5, num_bins=7, max_steps=200):
        super().__init__()

        self.grid_size = grid_size
        self.window_size = window_size
        
        self.rover_pos = None
        self.sample_pos = None
        self.lander_pos = None
        self.has_sample = False
        self.num_bins = num_bins
        self.steps = 0
        self.max_steps = max_steps

        # 0: South, 1: North, 2: West, 3: East
        self.action_space = spaces.Discrete(4)

        obs_size = 1 + num_bins + ((window_size ** 2) * 4)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.grid = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self.create_grid()
        self.rover_pos = (0, 0)
        self.lander_pos = (0, self.grid_size - 1)
        self.sample_pos = (self.grid_size - 1, int(self.grid_size * 0.8))
        self.has_sample = False
        self.steps = 0

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
            reward += -1
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
        
        obs = self.get_observation()

        return obs, reward, terminated, truncated, {}

    

    def create_grid(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # add craters
        grid[5, 5] = CRATER
        grid[10, 12] = CRATER
        grid[15, 3] = CRATER

        # add cliffs
        grid[7, 8] = CLIFF
        grid[12, 15] = CLIFF
        grid[3, 17] = CLIFF
        

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




        

if __name__ == "__main__":
    env = MarsRoverEnv()
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    total_reward = 0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {step}: action={action}, reward={reward}, "
              f"terminated={terminated}, truncated={truncated}, pos={env.rover_pos}")

        if terminated or truncated:
            print(f"Episode ended. Total reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0

