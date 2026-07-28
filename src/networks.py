import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, state):
        x = state
        x = self.mlp(x)
        return x
    
    def act(self, state):
        logits = self.forward(state)
        probabilities = torch.distributions.Categorical(logits=logits)
        action = probabilities.sample()
        log_prob = probabilities.log_prob(action)
        return action, log_prob


class ValueNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim=1, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, state):
        x = state
        x = self.mlp(x)
        return x

