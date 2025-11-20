#!/usr/bin/env python3

import torch
import numpy as np
import gymnasium
import argparse
from torch import nn as nn
from softactorcritic.agent import Agent
from softactorcritic.trainer import SACTrainer


def train_agent(
    env_name="Humanoid-v5",
    num_episodes=1000,
    save_path="./models",
    render=False,
    batch_size=256,
    lr=3e-4,
    gamma=0.99,
    alpha=0.2,
    tau=0.005,
    buffer_size=1000000,
    hidden_dims=None,
):
    
    if hidden_dims is None:
        hidden_dims = [256, 256]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    np.set_printoptions(precision=3)

    print(f"Using device: {device}")
    print(f"Training on environment: {env_name}")
    print(f"Total episodes: {num_episodes}")
    if render:
        print("Rendering enabled (may not display depending on system configuration)")
    try:
        env = gymnasium.make(env_name, render_mode="human" if render else None)
    except Exception as e:
        print(f"Warning: Could not create environment with render_mode='human': {e}")
        print("Creating environment without rendering...")
        env = gymnasium.make(env_name, render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        layer_size=256,
        activation=nn.ReLU,
        data_type=torch.float32,
    )

    trainer = SACTrainer(
        agent=agent,
        gamma=gamma,
        lr=lr,
        alpha=alpha,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        data_type=torch.float32,
    )

    #training
    episode_rewards = []
    episode_steps = []

    print("\nStarting training...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        episode_reward = 0
        episode_step = 0
        done = False
        truncated = False

        while not (done or truncated):
            agent.set_eval()
            with torch.no_grad():
                action = agent.get_action(state)

            action_np = action.cpu().numpy()[0]
            next_state, reward, done, truncated, _ = env.step(action_np)

            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            done_tensor = torch.tensor(float(done), dtype=torch.float32)

            trainer.add_to_buffer(state, action, reward_tensor, next_state, done_tensor)

            agent.set_train()
            q1_loss, q2_loss, actor_loss, alpha_loss = trainer.train_step()

            episode_reward += reward
            episode_step += 1
            state = next_state

        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_steps = np.mean(episode_steps[-10:])
            print(
                f"Episode {episode + 1:4d}/{num_episodes} | "
                f"Avg Reward: {avg_reward:8.2f} | "
                f"Avg Steps: {avg_steps:6.1f} | "
                f"Episode Reward: {episode_reward:8.2f}"
            )

        # save checkpoints
        if (episode + 1) % 100 == 0:
            trainer.save_trainer(save_path)
            print(f"  -> Models saved to {save_path}")

    trainer.save_trainer(save_path)
    env.close()

    final_avg_reward = np.mean(episode_rewards[-10:])
    max_reward = max(episode_rewards)
    min_reward = min(episode_rewards)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Final 10-episode average reward: {final_avg_reward:.2f}")
    print(f"Best episode reward: {max_reward:.2f}")
    print(f"Worst episode reward: {min_reward:.2f}")
    print(f"Models saved to: {save_path}")
    print("=" * 60)

    return {
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "final_avg_reward": final_avg_reward,
        "max_reward": max_reward,
        "min_reward": min_reward,
    }


def evaluate_agent(model_path, env_name="Humanoid-v5", num_episodes=10, render=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    print(f"\nEvaluating agent from {model_path}")
    print(f"Environment: {env_name}")

    # Create environment
    env = gymnasium.make(env_name, render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        layer_size=256,
        activation=nn.ReLU,
        data_type=torch.float32,
    )

    agent.load_models(model_path)
    agent.set_eval()

    eval_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            with torch.no_grad():
                action = agent.get_action(state)

            action_np = action.cpu().numpy()[0]
            next_state, reward, done, truncated, _ = env.step(action_np)

            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            episode_reward += reward
            state = next_state

        eval_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes} | Reward: {episode_reward:.2f}")

    env.close()

    avg_eval_reward = np.mean(eval_rewards)
    print(f"\nAverage evaluation reward: {avg_eval_reward:.2f}")

    return {
        "eval_rewards": eval_rewards,
        "avg_reward": avg_eval_reward,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Soft Actor-Critic (SAC) Training and Evaluation"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    train_parser = subparsers.add_parser("train", help="Train a new agent")
    train_parser.add_argument(
        "--env", type=str, default="Humanoid-v5", help="Environment name"
    )
    train_parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    train_parser.add_argument(
        "--save-path", type=str, default="./models", help="Path to save models"
    )
    train_parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during training",
    )
    train_parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate"
    )
    train_parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor"
    )
    train_parser.add_argument(
        "--alpha", type=float, default=0.2, help="Entropy coefficient"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size"
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained agent")
    eval_parser.add_argument(
        "model_path", type=str, help="Path to the saved model"
    )
    eval_parser.add_argument(
        "--env", type=str, default="Humanoid-v5", help="Environment name"
    )
    eval_parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    eval_parser.add_argument(
        "--no-render",
        action="store_true",
        help="Do not render during evaluation",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_agent(
            env_name=args.env,
            num_episodes=args.episodes,
            save_path=args.save_path,
            render=args.render,
            lr=args.lr,
            gamma=args.gamma,
            alpha=args.alpha,
            batch_size=args.batch_size,
        )
    elif args.command == "evaluate":
        evaluate_agent(
            model_path=args.model_path,
            env_name=args.env,
            num_episodes=args.episodes,
            render=not args.no_render,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
