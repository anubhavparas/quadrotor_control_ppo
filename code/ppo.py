import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from actor_critic import ActorCritic

class PPO:
    def __init__(self, device, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.device = device
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        #self.optimizer = RAdam(self.policy.parameters(), lr=lr, betas=betas)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)



        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        if np.any(np.isnan(state)):
          print('in select action: state is nan',  state)
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states_ = torch.squeeze(torch.stack(memory.states).to(self.device)).detach()
        old_actions_ = torch.squeeze(torch.stack(memory.actions).to(self.device)).detach()
        old_logprobs_ = torch.squeeze(torch.stack(memory.logprobs)).to(self.device).detach()
        
        batch_size = old_states_.shape[0]
        mini_batch_size = batch_size // 8 # 64
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            for i in range(batch_size // mini_batch_size):
              rand_ids = np.random.randint(0, batch_size, mini_batch_size)
              old_states = old_states_[rand_ids, :]
              old_actions = old_actions_[rand_ids, :] 
              old_logprobs = old_logprobs_[rand_ids, :]
              rewards_batch = rewards[rand_ids]

              logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

              # Finding the ratio (pi_theta / pi_theta__old):
              ratios = torch.exp(logprobs - old_logprobs.detach())

              # Finding Surrogate Loss:
              advantages = rewards_batch - state_values.detach()
              ## torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
              #surr = -torch.min(ratios, 1) * advantages  # as per the paper

              len_adv = advantages.shape[0]
              advantages = advantages.reshape((len_adv, 1))
              surr1 = ratios * advantages
              surr2 = 1 * advantages   ## as per the paper

              surr  = -torch.min(surr1, surr2).mean()
              w_crit_loss = 1
              loss = surr + w_crit_loss * (rewards_batch - state_values).pow(2).mean() #- 0.01 * dist_entropy

              # take gradient step
              self.optimizer.zero_grad()
              loss.mean().backward()
              self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
