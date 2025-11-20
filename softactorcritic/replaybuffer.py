import torch, numpy as np
from torch import nn as nn

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, data_type):
        """
        Initialize the replay buffer.

        Args:
            buffer_size: Maximum number of experiences to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            data_type: PyTorch data type (e.g., torch.float32)
        """
        self.buffer_size=buffer_size
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.data_type=data_type
        self.reset()

    # init the tensors with the buffer size # of states and # of values
    def reset(self):
        self.states=torch.zeros((self.buffer_size, self.state_dim),dtype=self.data_type,requires_grad=False)
        self.actions=torch.zeros((self.buffer_size, self.action_dim),dtype=self.data_type,requires_grad=False)
        self.rewards=torch.zeros((self.buffer_size,1),dtype=self.data_type,requires_grad=False)
        self.next_states=torch.zeros((self.buffer_size, self.state_dim),dtype=self.data_type,requires_grad=False)
        self.dones=torch.zeros((self.buffer_size, 1),dtype=self.data_type,requires_grad=False)

        self.index=0
        self.size=0

    # each time interacts with environemnet, create new experience and store at index, it should be in a circulasr buffer so as new data comes in, it fills up old data
    def add(self, state, action, reward, next_state, done):
        # Remove batch dimension if present (state/action come with batch=1)
        if state.dim() > 1:
            state = state.squeeze(0)
        if action.dim() > 1:
            action = action.squeeze(0)
        if next_state.dim() > 1:
            next_state = next_state.squeeze(0)
        if reward.dim() > 1:
            reward = reward.squeeze(0)
        if done.dim() > 1:
            done = done.squeeze(0)

        self.states[self.index]=state
        self.actions[self.index]=action
        self.rewards[self.index]=reward
        self.next_states[self.index]=next_state
        self.dones[self.index]=done
        self.index=(self.index+1)%self.buffer_size # ciruclar buffer
        if self.size<self.buffer_size:
            self.size+=1

    # replay buffer onyl used for memory storage, not for training NN
    def no_gradient(self):
        self.states=self.states.detach() # detacch=does not store gradients
        self.actions=self.actions.detach()
        self.rewards=self.rewards.detach()
        self.next_states=self.next_states.detach()
        self.dones=self.dones.detach()

    #sample a random batch
    def sample(self, batch_size):
        indexes=np.random.choice(self.size,batch_size,False) # parameters are: range, how many, duplicates?
        states=self.states[indexes].detach()
        actions=self.actions[indexes].detach()
        rewards=self.rewards[indexes].detach()
        next_states=self.next_states[indexes].detach()
        dones=self.dones[indexes].detach()
        return states, actions, rewards, next_states, dones


