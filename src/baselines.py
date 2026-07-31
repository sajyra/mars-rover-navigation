from environment import MarsRoverEnv
import heapq
import environment

def random_policy(env, episodes=100):

    all_steps = []
    all_rewards = []
    all_successes = []
    success_steps = []

    for episode in range(episodes):

        env.reset()

        done = False
        steps = 0
        rewards = 0
        success = False 

        while not done:

            action = env.action_space.sample()
            steps += 1

            obs, reward, terminated, truncated, info = env.step(action)

            rewards += reward

            done = terminated or truncated

            if truncated:
                print(f"Total reward this episode: {rewards}")
            
            

            if terminated:
                print(f"Total reward this episode: {rewards}")
                if env.has_sample and env.rover_pos == env.lander_pos:
                    print("Successful episode")
                    success = True
                else:
                    print("Fell off cliff")
        
        all_steps.append(steps)
        all_rewards.append(rewards)
        if success:
            all_successes.append(episode + 1)
            success_steps.append(steps)
    
    success_rate = len(all_successes) / episodes * 100
    avg_reward = sum(all_rewards) / len(all_rewards)
    avg_steps = sum(all_steps) / len(all_steps)
    avg_success_steps = -1
    if len(success_steps) > 0:
        avg_success_steps = sum(success_steps) / len(success_steps)

    print(f"\n--- Random Policy Baseline ({episodes} episodes) ---")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average success steps: {avg_success_steps:.1f}")


def heuristic(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def a_star(grid, start, goal):
    grid_size = grid.shape[0]

    def step_cost(pos):
        row, col = pos
        if grid[row, col] == environment.CRATER:
            return 10
        return 1  

    def get_neighbors(pos):
        row, col = pos
        candidates = [(row+1, col), (row-1, col), (row, col+1), (row, col-1)]
        valid = []
        for r, c in candidates:
            if 0 <= r < grid_size and 0 <= c < grid_size:
                if grid[r, c] != environment.CLIFF:  # cliffs are impassable
                    valid.append((r, c))
        return valid

    open_set = [(heuristic(start, goal), start)]
    g_score = {start: 0}
    came_from = {}

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if current_f > g_score[current] + heuristic(current, goal):
            continue  # stale entry

        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + step_cost(neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    return None 


if __name__ == "__main__":
    env = MarsRoverEnv(8, 5, max_steps=150, randomize=True)
    path_lengths = []
    for _ in range(100):
        env.reset()
        path1 = a_star(env.grid, env.rover_pos, env.sample_pos)
        path2 = a_star(env.grid, env.sample_pos, env.lander_pos)
        if path1 and path2:
            path_lengths.append(len(path1) + len(path2) - 2)

    avg_optimal = sum(path_lengths) / len(path_lengths)
    print(f"\n--- A* Baseline (100 episodes) ---")
    print(f"Average optimal path length: {avg_optimal:.1f}")

    env.reset()
    random_policy(env, 100)
