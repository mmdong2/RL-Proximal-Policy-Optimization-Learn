"""使用批量 REINFORCE 训练并评估 Gymnasium CartPole。"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """把 CartPole 的状态映射为离散动作的 logits。"""

    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        # CartPole 的状态有 4 个数，动作为左/右两个选项。
        # 两层 Tanh 隐藏层输出每个动作的 logits，再由 Categorical 转成概率分布。
        self.layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations)

def parse_args() -> argparse.Namespace:
    """读取训练轮数、批量大小、学习率和模型保存路径等设置。"""
    parser = argparse.ArgumentParser(
        description="Train a CartPole policy with batch REINFORCE."
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--batch-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("reinforce_cartpole.pt"),
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the evaluation episodes.",
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be greater than 0")
    if args.batch_episodes <= 0:
        parser.error("--batch-episodes must be greater than 0")
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be greater than 0")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than 0")
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be greater than 0")
    if not 0 <= args.gamma <= 1:
        parser.error("--gamma must be between 0 and 1")
    return args


def collect_episode(
    env: gym.Env,
    policy: PolicyNetwork,
    seed: int,
    gamma: float,
) -> tuple[list[np.ndarray], list[int], list[float], float]:
    observation, _ = env.reset(seed=seed)
    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    episode_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        state = np.asarray(observation, dtype=np.float32)
        with torch.no_grad():
            # 按当前策略分布采样动作；采集整批轨迹期间不更新策略参数。
            logits = policy(torch.as_tensor(state))
            action = int(Categorical(logits=logits).sample().item())

        states.append(state.copy())
        actions.append(action)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        episode_reward += float(reward)

    # 对应文档公式 (10)：G_t = sum_{k=t}^T gamma^k * r_k。
    # 从后往前累加，得到每一步对应的“整局折扣回报尾项”。
    # gamma 的指数仍使用整局时间 k，因此梯度里不再额外乘 gamma^t。
    returns = [0.0] * len(rewards)
    discounted_tail = 0.0
    for timestep in range(len(rewards) - 1, -1, -1):
        discounted_tail += (gamma**timestep) * rewards[timestep]
        returns[timestep] = discounted_tail

    return states, actions, returns, episode_reward


def train(
    args: argparse.Namespace,
    policy: PolicyNetwork,
    env: gym.Env,
) -> None:

    """先收集一批完整轨迹，再用整批数据更新一次策略。"""
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    episode_index = 0

    for iteration in range(1, args.iterations + 1):
        batch_states: list[np.ndarray] = []
        batch_actions: list[int] = []
        batch_returns: list[float] = []
        episode_rewards: list[float] = []

        # 这批数据都由同一个当前策略采集，直到整批采集完成才更新参数。
        for _ in range(args.batch_episodes):
            states, actions, returns, episode_reward = collect_episode(
                env=env,
                policy=policy,
                seed=args.seed + episode_index,
                gamma=args.gamma,
            )
            episode_index += 1
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)
            episode_rewards.append(episode_reward)

        states_tensor = torch.as_tensor(
            np.asarray(batch_states),
            dtype=torch.float32,
        )
        actions_tensor = torch.as_tensor(batch_actions, dtype=torch.long)
        returns_tensor = torch.as_tensor(batch_returns, dtype=torch.float32)

        # 重新计算采样动作在当前策略下的 log-probability，
        # 对应文档中的 ∇θ log πθ(a_t | s_t)。
        log_probabilities = Categorical(
            logits=policy(states_tensor)
        ).log_prob(actions_tensor)

        # 对应文档公式 (11) 的批量策略梯度估计。
        # 用负号把“梯度上升最大化回报”改写成优化器可最小化的损失；
        # 除以轨迹数 M，与公式中的批次平均一致。
        loss = -(log_probabilities * returns_tensor).sum() / args.batch_episodes
        optimizer.zero_grad()
        loss.backward()

        # 对应文档公式 (12)：完成整批梯度计算后更新一次参数。
        optimizer.step()

        if iteration == 1 or iteration % 10 == 0 or iteration == args.iterations:
            print(
                f"Iteration {iteration}/{args.iterations} | "
                f"mean training reward: {np.mean(episode_rewards):.1f} | "
                f"loss: {loss.item():.3f}"
            )


def evaluate(
    policy: PolicyNetwork,
    episodes: int,
    seed: int,
    render: bool,
) -> tuple[float, float]:
    """用确定性策略（选 logits 最大的动作）评估训练结果。"""
    env = gym.make(
        "CartPole-v1",
        render_mode="human" if render else None,
    )
    rewards: list[float] = []

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            terminated = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated):
                state = torch.as_tensor(
                    np.asarray(observation, dtype=np.float32)
                )
                with torch.no_grad():
                    # 评估时不再采样，直接选策略概率最高的动作。
                    action = int(policy(state).argmax().item())
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)

            rewards.append(episode_reward)
    finally:
        env.close()

    return float(np.mean(rewards)), float(np.std(rewards))


def main() -> None:
    args = parse_args()

    # 固定随机种子，便于复现实验。
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_env = gym.make("CartPole-v1")
    train_env.action_space.seed(args.seed)

    # 确认环境空间符合这个策略网络的输入/输出要求。
    if not isinstance(train_env.observation_space, gym.spaces.Box):
        raise TypeError("CartPole-v1 must provide a Box observation space")
    if not isinstance(train_env.action_space, gym.spaces.Discrete):
        raise TypeError("CartPole-v1 must provide a Discrete action space")

    policy = PolicyNetwork(
        observation_dim=train_env.observation_space.shape[0],
        action_dim=train_env.action_space.n,
        hidden_size=args.hidden_size,
    )

    try:
        train(args, policy, train_env)
    finally:
        train_env.close()

    # 保存网络参数和重建网络所需的基本配置。
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "observation_dim": policy.observation_dim,
            "action_dim": policy.action_dim,
            "hidden_size": args.hidden_size,
            "gamma": args.gamma,
            "seed": args.seed,
        },
        args.model_path,
    )
    print(f"Model saved to: {args.model_path}")

    mean_reward, std_reward = evaluate(
        policy=policy,
        episodes=args.eval_episodes,
        seed=args.seed + args.iterations * args.batch_episodes,
        render=args.render,
    )
    print(
        f"Evaluation reward over {args.eval_episodes} episodes: "
        f"{mean_reward:.1f} +/- {std_reward:.1f}"
    )


if __name__ == "__main__":
    main()
