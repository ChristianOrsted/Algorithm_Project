import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from weak_tie_module import WeakTieMAPPO_Critic


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, obs, avail_actions):
        """
        obs: (batch, obs_dim)
        avail_actions: (batch, act_dim) 1为可用, 0为不可用
        """
        logits = self.net(obs)
        # 屏蔽不可用动作：将其 logit 设为极大负数
        logits[avail_actions == 0] = -1e10
        return F.softmax(logits, dim=-1)


class WeakTieMAPPOAgent:
    def __init__(self, n_agents, obs_dim, act_dim, lr=5e-4, gamma=0.99, clip_param=0.2):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.clip_param = clip_param
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.critic = WeakTieMAPPO_Critic(n_agents, obs_dim, act_dim).to(self.device)

        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=lr)

    def select_action(self, obs, avail_actions):
        """采样动作"""
        obs_t = torch.FloatTensor(obs).to(self.device).unsqueeze(0)  # (1, n, obs)
        avail_t = torch.FloatTensor(avail_actions).to(self.device).unsqueeze(0)

        with torch.no_grad():
            # 批量处理所有智能体
            actions = []
            probs_list = []
            for i in range(self.n_agents):
                probs = self.actor(obs_t[:, i, :], avail_t[:, i, :])
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                actions.append(action.item())
                probs_list.append(probs.cpu().numpy()[0])

        return actions, np.array(probs_list)

    def update(self, buffer):
        """完整的 PPO 更新逻辑"""
        # 1. 数据准备
        obs = torch.FloatTensor(np.array(buffer['obs'])).to(self.device)  # (B, N, Obs)
        acts = torch.LongTensor(np.array(buffer['acts'])).to(self.device)  # (B, N)
        rewards = torch.FloatTensor(np.array(buffer['rewards'])).to(self.device)  # (B, N)
        dones = torch.FloatTensor(np.array(buffer['dones'])).to(self.device)
        avails = torch.FloatTensor(np.array(buffer['avails'])).to(self.device)  # (B, N, Act)
        masks = torch.FloatTensor(np.array(buffer['masks'])).to(self.device)
        keys = torch.LongTensor(np.array(buffer['keys'])).to(self.device)
        old_probs_buffer = torch.FloatTensor(np.array(buffer['probs'])).to(self.device)

        batch_size = obs.shape[0]
        acts_onehot = F.one_hot(acts, num_classes=self.act_dim).float()

        # 2. 计算回报 (Monte Carlo Return)
        # 注意：这里计算的是折扣回报 G_t，作为 Critic 的 Target
        returns = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(batch_size)):
            R = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R

        # 3. Critic 更新
        values = self.critic(obs, acts_onehot, masks, keys).squeeze(-1)
        critic_loss = F.mse_loss(values, returns.detach())

        # 4. Actor 更新 (计算 PPO Ratio)
        actor_loss = 0
        # 针对每个 Agent 分别计算 Loss 并求和 (或者 Batch 维度展开)
        for i in range(self.n_agents):
            # 获取当前策略的概率分布
            curr_probs = self.actor(obs[:, i, :], avails[:, i, :])
            dist = torch.distributions.Categorical(curr_probs)
            new_log_probs = dist.log_prob(acts[:, i])

            # 获取旧策略概率 (从 buffer 中恢复，重新计算 log_prob 以保证梯度切断)
            # 注意：这里直接使用 buffer 里的 probs 会更方便，但为了严谨使用 Categorical
            old_dist = torch.distributions.Categorical(old_probs_buffer[:, i, :])
            old_log_probs = old_dist.log_prob(acts[:, i])

            # PPO Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 优势函数 Advantage
            advantage = returns[:, i] - values[:, i].detach()
            # 标准化优势函数 (这对 PPO 稳定性很重要)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Clip Loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
            agent_loss = -torch.min(surr1, surr2).mean()
            actor_loss += agent_loss

        # 5. 反向传播
        total_loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.optimizer.step()

        return total_loss.item()