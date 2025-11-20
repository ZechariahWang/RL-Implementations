import torch
from torch import nn as nn
import os
from softactorcritic.networks import MLP, QNetwork, ActionNetwork

class Agent:
    def __init__(self, state_dim, action_dim, hidden_dims, layer_size, activation=nn.ReLU, data_type=torch.float32):
        self.input_shape=state_dim
        self.action_shape=action_dim
        self.hidden_dims=hidden_dims
        self.layer_size=layer_size
        self.activation=activation
        self.data_type=data_type

        self.actor=ActionNetwork(state_dim, action_dim, hidden_dims, layer_size, activation).to(dtype=self.data_type)
        self.q1=QNetwork(state_dim, action_dim, hidden_dims, layer_size, activation).to(dtype=self.data_type)
        self.q2=QNetwork(state_dim, action_dim, hidden_dims, layer_size, activation).to(dtype=self.data_type)
        self.q1_target=QNetwork(state_dim, action_dim, hidden_dims, layer_size, activation).to(dtype=self.data_type)
        self.q2_target=QNetwork(state_dim, action_dim, hidden_dims, layer_size, activation).to(dtype=self.data_type)

        self.networks=[self.actor, self.q1, self.q2, self.q1_target, self.q2_target]
        self.q_networks=[self.q1, self.q2]

    def update_target_networks(self, tau=0.01):
        with torch.no_grad():
            for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def get_action(self, state):
        self.set_eval()
        with torch.no_grad():
            action, _, _, _ = self.actor(state)
        return action
    
    def set_eval(self):
        for net in self.networks:
            net.eval()
    
    def set_train(self):
        for net in self.networks:
            net.train()
    
    def save_models(self, path):
        os.makedirs(path, exist_ok=True)
        for i, net in enumerate(self.networks):
            torch.save(net.state_dict(), f"{path}/network_{i}.pth")
    
    def load_models(self, path):
        for i, net in enumerate(self.networks):
            net.load_state_dict(torch.load(f"{path}/network_{i}.pth"))
        