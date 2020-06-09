import torch
import torch.nn as nn
import torch.nn.functional as F



class DDPGNet(nn.Module):
    """
    Actor Critic model that gets trained by DDPG agent
    """
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor = False):
        super(DDPGNet, self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim,output_dim)
        self.relu = F.relu
        self.is_actor = actor
        
    def actor_network(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = (self.fc3(h2))
        return torch.tanh(h3)
        
    def critic_network(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = (self.fc3(h2))
        return h3
        
    def forward(self, x):
        if self.is_actor:
            return self.actor_network(x)
        else:
            # critic network simply outputs a number
            return self.critic_network(x)