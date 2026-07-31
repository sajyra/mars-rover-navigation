from environment import MarsRoverEnv
import torch
import networks
from buffer import RolloutBuffer
from gae import compute_gae
from loss import compute_ppo_loss
import environment

def train(num_iterations, num_epochs, grid_size=20, window_size=5, max_steps=200, rollout_length=1500):

    env = MarsRoverEnv(grid_size, window_size, max_steps=max_steps, randomize=True)
    input_dim = env.observation_space.shape[0]
    policy_net = networks.PolicyNetwork(input_dim, 4)
    value_net = networks.ValueNetwork(input_dim)
    buffer = RolloutBuffer()

    params = list(policy_net.parameters()) + list(value_net.parameters())
    optimizer = torch.optim.Adam(params, 3e-4)

    for iteration in range(num_iterations):

        buffer.clear()
        state, __ = env.reset()
        state = torch.as_tensor(state, dtype=torch.float32)
        hidden, cell = torch.zeros(policy_net.hidden_dim), torch.zeros(policy_net.hidden_dim)
        curr_episode_reward = 0

        for step in range(rollout_length):

            with torch.no_grad():

                logits, (new_hidden, new_cell) = policy_net(state, (hidden, cell))
                action, log_prob = networks.act(logits)
                value = value_net(state)
                value = value.squeeze()
            
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            done_bool = 1 if done else 0
            buffer.store(state, action, log_prob, reward, value, done_bool)
            hidden, cell = new_hidden, new_cell
            curr_episode_reward += reward

            if done:
                state, __ = env.reset()
                print(f"Reward: {curr_episode_reward}")
                curr_episode_reward = 0
                hidden, cell = torch.zeros(policy_net.hidden_dim), torch.zeros(policy_net.hidden_dim)
            else:
                state = next_state
            state = torch.as_tensor(state, dtype=torch.float32)
        
        with torch.no_grad():
            next_val = value_net(state)
            next_val = next_val.squeeze()
        
        states, actions, log_probs, rewards, values, dones = buffer.get()
        advantages, returns = compute_gae(rewards, values, dones, next_val)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        chunk_length = 200

    
        for epoch in range(num_epochs):

            hidden, cell = torch.zeros(policy_net.hidden_dim), torch.zeros(policy_net.hidden_dim)

            for start in range(0, len(states), chunk_length):

                end = start + chunk_length

                optimizer.zero_grad()

                loss, (hidden, cell) = compute_ppo_loss(policy_net, value_net, states[start:end], actions[start:end], dones[start:end], log_probs[start:end], advantages[start:end], returns[start:end], (hidden, cell))
                loss.backward()
                optimizer.step()

                hidden = hidden.detach()
                cell = cell.detach()
        
        print(f"Iteration: {iteration}")
        if iteration % 50 == 0:
            torch.save(policy_net.state_dict(), f"src/policy_net_8_{iteration + 1}.pt")
            evaluate(policy_net, env)
        
    
    evaluate(policy_net, env)


def evaluate(policy_net, env):

    total_rewards = []
    total_successes = []
    total_cliffs = []
    total_timeouts = []
    success_steps = []
    for i in range(100):
        
        curr_rewards = 0
        hidden, cell = torch.zeros(policy_net.hidden_dim), torch.zeros(policy_net.hidden_dim)
        state, __ = env.reset()
        state = torch.as_tensor(state, dtype=torch.float32)
        done = False
        steps = 0

        while not done:

            with torch.no_grad():

                logits, (hidden, cell) = policy_net(state, (hidden, cell))
                action = torch.argmax(logits)
                next_state, reward, terminated, truncated, info = env.step(action.item())
                state = torch.as_tensor(next_state, dtype=torch.float32)
                curr_rewards += reward
                done = terminated or truncated
                print(env.rover_pos)
                steps += 1

                if truncated:
                    total_timeouts.append(i + 1)
                if terminated:
                    if env.has_sample and env.rover_pos == env.lander_pos:
                        total_successes.append(i + 1)
                        success_steps.append(steps)
                    else:
                        total_cliffs.append(i + 1)

        print(curr_rewards)
        total_rewards.append(curr_rewards)
    
    print(f"Average reward: {sum(total_rewards) / len(total_rewards)}")
    print(f"Total timeouts: {len(total_timeouts)}")
    print(f"Total cliffs: {len(total_cliffs)}")
    print(f"Percentage success: {len(total_successes)}")
    print(f"Average success steps: {sum(success_steps) / len(success_steps)}")

if __name__ == "__main__":
    # train(2000, 4, 8, 5, 150, 5000)
    env = MarsRoverEnv(8, 5, max_steps=150, randomize=True)
    input_dim = env.observation_space.shape[0]
    policy_net = networks.PolicyNetwork(input_dim, 4)
    policy_net.load_state_dict(torch.load("src/policy_net_8_551.pt"))
    evaluate(policy_net, env)
    
