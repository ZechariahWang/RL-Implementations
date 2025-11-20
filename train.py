import torch, numpy as np, gymnasium, argparse
from torch import nn as nn
from softactorcritic.agent import Agent
from softactorcritic.trainer import SACTrainer

# Soft Actor-Critic (SAC) Training Script
# This script trains an agent using the SAC algorithm on gymnasium environments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(description="Train a Soft Actor-Critic agent")
parser.add_argument("--nogui", action="store_true", help="Run without GUI/rendering")
parser.add_argument("--reset", action="store_true", help="Reset the environment")
parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
parser.add_argument("--env", type=str, default="Humanoid-v5", help="Gymnasium environment name")
parser.add_argument("--save-path", type=str, default="./models", help="Path to save trained models")
args = parser.parse_args()

# Create the environment
env = gymnasium.make(args.env, render_mode=None if args.nogui else "human")

# Initialize the agent with network architecture
agent = Agent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    hidden_dims=[256, 256],
    layer_size=256,
    activation=nn.ReLU,
    data_type=torch.float32
)

# Initialize the trainer with hyperparameters
trainer = SACTrainer(
    agent=agent,
    gamma=0.99,  # Discount factor
    lr=3e-4,  # Learning rate
    alpha=0.2,  # Entropy coefficient
    buffer_size=1000000,  # Replay buffer size
    batch_size=256,  # Batch size for training
    tau=0.005,  # Target network update rate
    data_type=torch.float32
)

num_episodes = args.episodes

# Training loop
episode_rewards = []
for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    episode_reward = 0
    done = False
    truncated = False

    while not (done or truncated):
        # Get action from the agent
        with torch.no_grad():
            action = agent.get_action(state)

        # Step the environment
        action_np = action.cpu().numpy()[0]
        next_state, reward, done, truncated, _ = env.step(action_np)

        # Convert to tensors
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(float(done), dtype=torch.float32)

        # Store in replay buffer
        trainer.add_to_buffer(state, action, reward, next_state, done)

        # Train the agent
        agent.set_train()
        q1_loss, q2_loss, actor_loss, alpha_loss = trainer.train_step()
        agent.set_eval()

        episode_reward += reward.item()
        state = next_state

    episode_rewards.append(episode_reward)

    # Print progress every 10 episodes
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Current Reward: {episode_reward:.2f}")

    # Save models every 100 episodes
    if (episode + 1) % 100 == 0:
        trainer.save_trainer(args.save_path)
        print(f"Models saved to {args.save_path}")

print(f"\nTraining completed! Final average reward: {np.mean(episode_rewards[-10:]):.2f}")
trainer.save_trainer(args.save_path)
env.close()
