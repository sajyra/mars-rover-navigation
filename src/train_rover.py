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
        
        curr_episode_reward = 0

        for step in range(rollout_length):

            with torch.no_grad():

                logits = policy_net(state)
                action, log_prob = networks.act(logits)
                value = value_net(state)
                value = value.squeeze()
            
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            done_bool = 1 if done else 0
            buffer.store(state, action, log_prob, reward, value, done_bool)
            curr_episode_reward += reward

            if done:
                state, __ = env.reset()
                print(f"Reward: {curr_episode_reward}")
                curr_episode_reward = 0
            else:
                state = next_state
            state = torch.as_tensor(state, dtype=torch.float32)
        
        with torch.no_grad():
            next_val = value_net(state)
            next_val = next_val.squeeze()
        
        states, actions, log_probs, rewards, values, dones = buffer.get()
        advantages, returns = compute_gae(rewards, values, dones, next_val)

    
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = compute_ppo_loss(policy_net, value_net, states, actions, log_probs, advantages, returns)
            loss.backward()
            optimizer.step()
        
        if iteration % 500 == 0 and iteration > 2000:
            torch.save(policy_net.state_dict(), f"policy_net{iteration}.pt")
        
    
    
    evaluate(policy_net, env)


def evaluate(policy_net, env):

    total_rewards = []
    total_successes = []
    total_cliffs = []
    total_timeouts = []
    for i in range(100):
        
        curr_rewards = 0
        state, __ = env.reset()
        state = torch.as_tensor(state, dtype=torch.float32)
        done = False

        while not done:

            with torch.no_grad():

                logits = policy_net(state)
                action = torch.argmax(logits)
                next_state, reward, terminated, truncated, info = env.step(action.item())
                state = torch.as_tensor(next_state, dtype=torch.float32)
                curr_rewards += reward
                done = terminated or truncated
                print(env.rover_pos)

                if truncated:
                    total_timeouts.append(i + 1)
                if terminated:
                    if env.has_sample and env.rover_pos == env.lander_pos:
                        total_successes.append(i + 1)
                    else:
                        total_cliffs.append(i + 1)

        print(curr_rewards)
        total_rewards.append(curr_rewards)
    
    print(f"Average reward: {sum(total_rewards) / len(total_rewards)}")
    print(f"Total timeouts: {len(total_timeouts)}")
    print(f"Total cliffs: {len(total_cliffs)}")
    print(f"Percentage success: {len(total_successes)}")

if __name__ == "__main__":
    train(2500, 4, 8, 7, 150, 5500)