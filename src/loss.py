import torch
import torch.nn.functional as F

def compute_ppo_loss(policy_net, value_net, states, actions, dones, old_log_probs, advantages, returns, hidden, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01):

    
    hidden, cell = hidden[0], hidden[1]

    logits_list = []

    for t in range(len(states)):
        
        logits, (hidden, cell) = policy_net(states[t], (hidden, cell))

        logits_list.append(logits)

        if dones[t]:
            hidden, cell = torch.zeros_like(hidden), torch.zeros_like(cell)


    logits = torch.stack(logits_list)
    probabilities = torch.distributions.Categorical(logits=logits)
    new_log_probs = probabilities.log_prob(actions)


    ratios = torch.exp(new_log_probs - old_log_probs)

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(value_net(states).squeeze(), returns)

    entropy = probabilities.entropy().mean()

    loss = policy_loss + (value_coef * value_loss) - (entropy_coef * entropy)
    return loss, (hidden, cell)
