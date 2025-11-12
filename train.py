import torch, numpy as np, gymnasium, argparse
from torch import nn as nn
from softactorcritic.agent import Agent
from softactorcritic.trainer import SACTrainer

# nov 11, wip trainer script for soft actor critic implementation

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_defualt_device(device)
np.set_printoptions(precision=3)

parser=argparse.ArgumentParser()
parser.add_argument("--nogui", action="store_true", help="Run without GUI")
parser.add_argument("--reset", action="store_true", help="Reset the environment")
args=parser.parse_args()

env=gymnasium.make("Humanoid-v5", render_mode=None if args.nogui else "human")
agent=Agent(state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_dims=[256,256],
            layer_size=256,
            activation=nn.ReLU,
            data_type=torch.float32)
trainer=SACTrainer(agent=agent,
                   gamma=0.99,
                   lr=3e-4,
                   alpha=0.2,
                   buffer_size=1000000,
                   batch_size=256,
                   tau=0.005,
                   data_type=torch.float32)

num_episodes=1000
