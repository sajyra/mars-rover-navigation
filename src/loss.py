import torch
import torch.nn.functional as F

def compute_ppo_loss(policy_net, value_net, states, actions, old_log_probs, advantages, returns, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01):

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    logits = policy_net(states)
    probabilities = torch.distributions.Categorical(logits=logits)
    new_log_probs = probabilities.log_prob(actions)
    ratios = torch.exp(new_log_probs - old_log_probs)

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(value_net(states).squeeze(), returns)

    entropy = probabilities.entropy().mean()

    loss = policy_loss + (value_coef * value_loss) - (entropy_coef * entropy)
    return loss
