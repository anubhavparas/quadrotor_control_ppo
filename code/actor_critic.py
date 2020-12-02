
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
            #nn.Tanh()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * action_std).to(device)
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):

        action_mean = self.actor(state)
        std   = self.log_std.exp().expand_as(action_mean)
        dist  = Normal(action_mean, std)


        
        #cov_mat = torch.diag(self.action_var).to(device)
        #dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = torch.squeeze(self.actor(state))

        #action_var = self.action_var.expand_as(action_mean)
        #cov_mat = torch.diag_embed(action_var).to(device)
        #dist = MultivariateNormal(action_mean, cov_mat)

        std   = self.log_std.exp().expand_as(action_mean)
        dist  = Normal(action_mean, std)

        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy