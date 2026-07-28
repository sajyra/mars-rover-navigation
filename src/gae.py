import torch

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    advantages = torch.zeros(len(rewards))
    next_val = next_value
    prev_advantage = 0

    for t in range(len(rewards) - 1, -1, -1):
        td_error = rewards[t] + (gamma * next_val * (1 - dones[t])) - values[t]
        advantages[t] = td_error + (gamma * lam * (1 - dones[t]) * prev_advantage)
        prev_advantage = advantages[t]
        next_val = values[t]
    
    returns = advantages + values
    return advantages, returns
