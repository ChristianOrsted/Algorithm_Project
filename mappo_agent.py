import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from weak_tie_module import WeakTieNet


# --- 工具类：Running Mean Std ---
class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


# --- 正交初始化 ---
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.GRUCell):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)


class WeakTieMAPPOAgent:
    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=64, lr=5e-4,
                 gamma=0.99, gae_lambda=0.95, clip_param=0.2,
                 ppo_epoch=10, mini_batch_size=8):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.mini_batch_size = mini_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 归一化层
        self.obs_norm = RunningMeanStd(shape=(obs_dim,))

        self.actor = WeakTieNet(obs_dim, hidden_dim, act_dim).to(self.device)
        self.critic = WeakTieNet(obs_dim + act_dim, hidden_dim, 1).to(self.device)

        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr * 5}
        ], eps=1e-5)

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.n_agents, self.hidden_dim).to(self.device)

    # ⭐️ 修改：增加 update_stats 参数，控制是否更新均值方差
    def normalize_obs(self, obs, update_stats=True):
        """对观测进行归一化"""
        flat_obs = obs.reshape(-1, self.obs_dim)
        if update_stats:
            self.obs_norm.update(flat_obs)
        norm_obs = (obs - self.obs_norm.mean) / (np.sqrt(self.obs_norm.var) + 1e-8)
        return norm_obs

    def select_action(self, obs, avail_actions, mask, key_idx, actor_hidden, deterministic=False):
        obs = np.array(obs)
        avail_actions = np.array(avail_actions)
        mask = np.array(mask)
        key_idx = np.array(key_idx)

        if obs.ndim == 2:
            obs = obs[None, ...]
            avail_actions = avail_actions[None, ...]
            mask = mask[None, ...]
            key_idx = key_idx[None, ...]

        # ⭐️ 玩游戏时：update_stats=True (收集统计信息)
        obs = self.normalize_obs(obs, update_stats=True)

        obs_t = torch.FloatTensor(obs).to(self.device)
        avail_t = torch.FloatTensor(avail_actions).to(self.device)
        mask_t = torch.FloatTensor(mask).to(self.device)
        key_t = torch.LongTensor(key_idx).to(self.device)

        with torch.no_grad():
            logits, new_hidden = self.actor(obs_t, obs_t, mask_t, key_t, actor_hidden)

            all_unavailable = (avail_t.sum(dim=-1, keepdim=True) == 0)
            if all_unavailable.any():
                avail_t = avail_t.clone()
                avail_t[all_unavailable.squeeze(-1), 0] = 1.0

            logits[avail_t == 0] = -1e10
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                actions = probs.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()

        return actions.cpu().numpy()[0], probs.cpu().numpy()[0], new_hidden

    def update_batch(self, buffer_list, entropy_coef=0.01):
        """
        基于 Batch (多个 Episodes) 进行 PPO 更新
        """
        batch_obs_list, batch_acts_list, batch_rews_list, batch_dones_list = [], [], [], []
        batch_avails_list, batch_masks_list, batch_keys_list, batch_old_probs_list = [], [], [], []
        batch_lens = []

        # --- 1. 数据收集与预处理 ---
        for episode_data in buffer_list:
            buf = episode_data[0]
            t_len = len(buf['obs'])

            obs = np.array(buf['obs']).reshape(t_len, self.n_agents, -1)
            acts = np.array(buf['acts']).reshape(t_len, self.n_agents)
            rews = np.array(buf['rewards']).reshape(t_len, self.n_agents, 1)
            dones = np.array(buf['dones']).reshape(t_len, self.n_agents, 1)
            avails = np.array(buf['avails']).reshape(t_len, self.n_agents, -1)
            masks = np.array(buf['masks']).reshape(t_len, self.n_agents, self.n_agents)
            keys = np.array(buf['keys']).reshape(t_len, 1)
            probs = np.array(buf['probs']).reshape(t_len, self.n_agents, -1)

            # ⭐️ 训练时：update_stats=False (不再更新统计量，保持稳定)
            obs = self.normalize_obs(obs, update_stats=False)

            batch_obs_list.append(torch.FloatTensor(obs))
            batch_acts_list.append(torch.LongTensor(acts))
            batch_rews_list.append(torch.FloatTensor(rews))
            batch_dones_list.append(torch.FloatTensor(dones))
            batch_avails_list.append(torch.FloatTensor(avails))
            batch_masks_list.append(torch.FloatTensor(masks))
            batch_keys_list.append(torch.LongTensor(keys))
            batch_old_probs_list.append(torch.FloatTensor(probs))
            batch_lens.append(t_len)

        # --- 2. 手动 Padding ---
        BatchSize = len(batch_lens)
        MaxTime = max(batch_lens)

        pad_obs = torch.zeros(BatchSize, MaxTime, self.n_agents, self.obs_dim).to(self.device)
        pad_acts = torch.zeros(BatchSize, MaxTime, self.n_agents, dtype=torch.long).to(self.device)
        pad_rews = torch.zeros(BatchSize, MaxTime, self.n_agents, 1).to(self.device)
        pad_dones = torch.zeros(BatchSize, MaxTime, self.n_agents, 1).to(self.device)
        pad_avails = torch.zeros(BatchSize, MaxTime, self.n_agents, self.act_dim).to(self.device)
        pad_masks = torch.zeros(BatchSize, MaxTime, self.n_agents, self.n_agents).to(self.device)
        pad_keys = torch.zeros(BatchSize, MaxTime, 1, dtype=torch.long).to(self.device)
        pad_old_probs = torch.zeros(BatchSize, MaxTime, self.n_agents, self.act_dim).to(self.device)

        valid_mask = torch.zeros(BatchSize, MaxTime).to(self.device)

        pad_avails[..., 0] = 1.0
        pad_old_probs[..., 0] = 1.0

        for i, t_len in enumerate(batch_lens):
            pad_obs[i, :t_len] = batch_obs_list[i].to(self.device)
            pad_acts[i, :t_len] = batch_acts_list[i].to(self.device)
            pad_rews[i, :t_len] = batch_rews_list[i].to(self.device)
            pad_dones[i, :t_len] = batch_dones_list[i].to(self.device)
            pad_avails[i, :t_len] = batch_avails_list[i].to(self.device)
            pad_masks[i, :t_len] = batch_masks_list[i].to(self.device)
            pad_keys[i, :t_len] = batch_keys_list[i].to(self.device)
            pad_old_probs[i, :t_len] = batch_old_probs_list[i].to(self.device)
            valid_mask[i, :t_len] = 1

        valid_mask_agent = valid_mask.unsqueeze(-1).expand(-1, -1, self.n_agents)

        # --- 3. GAE ---
        with torch.no_grad():
            values = []
            critic_hidden = self.init_hidden(BatchSize)
            acts_onehot = F.one_hot(pad_acts, num_classes=self.act_dim).float()
            critic_inputs = torch.cat([pad_obs, acts_onehot], dim=-1)

            for t in range(MaxTime):
                val, critic_hidden = self.critic(
                    critic_inputs[:, t], critic_inputs[:, t],
                    pad_masks[:, t], pad_keys[:, t], critic_hidden
                )
                values.append(val)

            values = torch.stack(values, dim=1).squeeze(-1)
            values_next = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)

            advantages = torch.zeros_like(values)
            deltas = pad_rews.squeeze(-1) + self.gamma * values_next * (1 - pad_dones.squeeze(-1)) - values
            gae = 0
            for t in reversed(range(MaxTime)):
                mask_t = valid_mask[:, t].unsqueeze(-1)
                gae = deltas[:, t] + self.gamma * self.gae_lambda * gae * mask_t
                advantages[:, t] = gae

            returns = advantages + values

        # --- 4. Update ---
        total_loss_log = 0
        indices = np.arange(BatchSize)

        for _ in range(self.ppo_epoch):
            np.random.shuffle(indices)

            for start_idx in range(0, BatchSize, self.mini_batch_size):
                mb_idx = indices[start_idx: start_idx + self.mini_batch_size]

                mb_obs = pad_obs[mb_idx]
                mb_acts = pad_acts[mb_idx]
                mb_avails = pad_avails[mb_idx]
                mb_masks = pad_masks[mb_idx]
                mb_keys = pad_keys[mb_idx]
                mb_old_probs = pad_old_probs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]
                mb_valid = valid_mask_agent[mb_idx]

                mb_acts_oh = F.one_hot(mb_acts, num_classes=self.act_dim).float()
                mb_critic_in = torch.cat([mb_obs, mb_acts_oh], dim=-1)

                curr_mb_size = len(mb_idx)
                actor_hidden = self.init_hidden(curr_mb_size)
                critic_hidden = self.init_hidden(curr_mb_size)

                new_log_probs_list = []
                new_values_list = []
                entropy_list = []

                for t in range(MaxTime):
                    logits, actor_hidden = self.actor(
                        mb_obs[:, t], mb_obs[:, t],
                        mb_masks[:, t], mb_keys[:, t], actor_hidden
                    )
                    logits[mb_avails[:, t] == 0] = -1e10
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)

                    new_log_probs_list.append(dist.log_prob(mb_acts[:, t]))
                    entropy_list.append(dist.entropy())

                    val, critic_hidden = self.critic(
                        mb_critic_in[:, t], mb_critic_in[:, t],
                        mb_masks[:, t], mb_keys[:, t], critic_hidden
                    )
                    new_values_list.append(val.squeeze(-1))

                new_log_probs = torch.stack(new_log_probs_list, dim=1)
                new_values = torch.stack(new_values_list, dim=1)
                entropy = torch.stack(entropy_list, dim=1)

                old_dist = torch.distributions.Categorical(mb_old_probs)
                old_log_probs = old_dist.log_prob(mb_acts)
                ratio = torch.exp(new_log_probs - old_log_probs)

                valid_adv = mb_adv[mb_valid == 1]
                if valid_adv.numel() > 1:
                    norm_adv = (mb_adv - valid_adv.mean()) / (valid_adv.std() + 1e-8)
                else:
                    norm_adv = mb_adv

                surr1 = ratio * norm_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * norm_adv

                valid_sum = mb_valid.sum() + 1e-8

                actor_loss = -(torch.min(surr1, surr2) * mb_valid).sum() / valid_sum
                critic_loss = (F.mse_loss(new_values, mb_ret, reduction='none') * mb_valid).sum() / valid_sum
                entropy_loss = (entropy * mb_valid).sum() / valid_sum

                total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
                self.optimizer.step()

                total_loss_log += total_loss.item()

        return total_loss_log / (self.ppo_epoch * (BatchSize / self.mini_batch_size))

    def save_model(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'obs_norm': {'mean': self.obs_norm.mean, 'var': self.obs_norm.var, 'count': self.obs_norm.count}
        }, path)

    def load_model(self, path):
        if os.path.exists(path):
            ckpt = torch.load(path)
            self.actor.load_state_dict(ckpt['actor'])
            self.critic.load_state_dict(ckpt['critic'])
            if 'obs_norm' in ckpt:
                self.obs_norm.mean = ckpt['obs_norm']['mean']
                self.obs_norm.var = ckpt['obs_norm']['var']
                self.obs_norm.count = ckpt['obs_norm']['count']
            print(f"✅ 模型已加载: {path}")