import torch, numpy as np
from torch import nn as nn
from softactorcritic.agent import Agent
from softactorcritic.replaybuffer import ReplayBuffer

class SACTrainer:
    def __init__(self, agent, gamma, lr, alpha, buffer_size, batch_size, tau=0.005, data_type=torch.float32):
        self.agent=agent
        self.gamma=gamma
        self.lr=lr
        self.tau=tau
        self.batch_size=batch_size
        self.data_type=data_type

        self.replay_buffer=ReplayBuffer(buffer_size, agent.input_shape, agent.action_shape, data_type)

        # Alpha as a learnable parameter
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, dtype=data_type)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self.target_entropy = -np.prod(agent.action_shape)  # target entropy = -action_dim

        self.q1_optimizer=torch.optim.Adam(self.agent.q1.parameters(), lr=self.lr)
        self.q2_optimizer=torch.optim.Adam(self.agent.q2.parameters(), lr=self.lr)
        self.actor_optimizer=torch.optim.Adam(self.agent.actor.parameters(), lr=self.lr)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    
    def compute_q_loss(self, state, action, reward, next_state, done):
        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.agent.actor(next_state)
            q1_next = self.agent.q1_target(next_state, next_action)
            q2_next = self.agent.q2_target(next_state, next_action)
            min_q_next = torch.min(q1_next, q2_next)
            v_next = min_q_next - self.alpha * next_log_prob.unsqueeze(-1)
            target_q = reward + self.gamma * (1 - done) * v_next

        q1_val = self.agent.q1(state, action)
        q2_val = self.agent.q2(state, action)

        q1_loss = nn.MSELoss()(q1_val, target_q)
        q2_loss = nn.MSELoss()(q2_val, target_q)

        return q1_loss, q2_loss

    def compute_actor_loss(self, state):

        new_action, log_prob, _, _ = self.agent.actor(state)
        q1_new_action = self.agent.q1(state, new_action)
        q2_new_action = self.agent.q2(state, new_action)
        min_q_new_action = torch.min(q1_new_action, q2_new_action)
        actor_loss = (self.alpha * log_prob.unsqueeze(-1) - min_q_new_action).mean()

        # Alpha loss for entropy regularization
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        return actor_loss, alpha_loss, log_prob
    
    def train_step(self):
        if self.replay_buffer.size < self.batch_size:
            return None, None, None, None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Update Q-networks
        q1_loss, q2_loss = self.compute_q_loss(states, actions, rewards, next_states, dones)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update Actor and Alpha (needs fresh forward pass to avoid graph issues)
        actor_loss, alpha_loss, _ = self.compute_actor_loss(states)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.agent.update_target_networks(tau=self.tau)

        return q1_loss.item(), q2_loss.item(), actor_loss.item(), alpha_loss.item()
    
    def add_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def save_trainer(self, path):
        self.agent.save_models(path)
        
    def load_trainer(self, path):
        self.agent.load_models(path)
