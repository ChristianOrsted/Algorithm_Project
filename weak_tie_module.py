import numpy as np
import torch
import torch.nn as nn
import networkx as nx


class WeakTieGraph:
    def __init__(self, n_agents, obs_range=15.0, alpha=0.3):
        self.n_agents = n_agents
        self.obs_range = obs_range
        self.alpha = alpha

    def compute_graph_info(self, agent_positions, alive_mask=None):
        """
        输入: agent_positions (n_agents, 2)
        输出: mask_beta (n, n), key_agent_idx (int)
        """
        n = self.n_agents
        G = nx.Graph()
        G.add_nodes_from(range(n))

        # 1. 建边 (Definition 1)
        for i in range(n):
            if alive_mask is not None and not alive_mask[i]: continue
            for j in range(i + 1, n):
                if alive_mask is not None and not alive_mask[j]: continue
                dist = np.linalg.norm(agent_positions[i] - agent_positions[j])
                if dist <= self.obs_range:
                    G.add_edge(i, j)

        # 2. 保证连通性 (Definition 2)
        active_nodes = [i for i in range(n) if (alive_mask is None or alive_mask[i])]
        if len(active_nodes) > 1 and not nx.is_connected(G.subgraph(active_nodes)):
            subgraphs = list(nx.connected_components(G.subgraph(active_nodes)))
            # 简单策略：串联各子图的第一个节点
            for k in range(len(subgraphs) - 1):
                u = list(subgraphs[k])[0]
                v = list(subgraphs[k + 1])[0]
                G.add_edge(u, v)

        # 3. 寻找 Key Agent (Definition 5)
        degrees = dict(G.degree())
        valid_degrees = {k: v for k, v in degrees.items() if (alive_mask is None or alive_mask[k])}
        key_agent_idx = max(valid_degrees, key=valid_degrees.get) if valid_degrees else 0

        # 4. 计算联系强度 H (Eq. 8)
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        H = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    H[i][j] = 1.0
                    continue
                # 死亡单位间无联系
                if alive_mask is not None and (not alive_mask[i] or not alive_mask[j]):
                    H[i][j] = 0.0
                    continue

                D_i = degrees[i]
                D_j = degrees[j]
                W_ij = path_lengths[i].get(j, 9999)  # 无穷大处理

                denominator = D_i + D_j + W_ij - 2
                strength = 1.0 / denominator if denominator > 0 else 0.0
                H[i][j] = strength

        # 5. 生成掩码 Beta (保留 弱联系 < alpha 和 自身)
        mask_beta = (H < self.alpha).astype(np.float32)
        np.fill_diagonal(mask_beta, 1.0)

        # 再次清理死亡单位的 mask
        if alive_mask is not None:
            dead_indices = np.where(alive_mask == 0)[0]
            mask_beta[dead_indices, :] = 0
            mask_beta[:, dead_indices] = 0

        return mask_beta, key_agent_idx


class WeakTieMAPPO_Critic(nn.Module):
    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=128):
        super(WeakTieMAPPO_Critic, self).__init__()
        self.n_agents = n_agents

        # 输入拼接：Local + Global(Weak) + KeyAgent
        feature_dim = obs_dim + act_dim
        input_dim = feature_dim + (n_agents * feature_dim) + feature_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, acts_onehot, mask_beta, key_agent_idx):
        batch_size = obs.shape[0]
        # (batch, n, feat)
        features = torch.cat([obs, acts_onehot], dim=-1)
        feat_dim = features.shape[-1]

        values = []
        for i in range(self.n_agents):
            # 1. 本地信息
            local_feat = features[:, i, :]

            # 2. 弱联系信息 (应用 Mask)
            mask = mask_beta[:, i, :].unsqueeze(-1)  # (batch, n, 1)
            weak_feats = features * mask
            weak_feats_flat = weak_feats.view(batch_size, -1)

            # 3. Key Agent 信息
            idx_exp = key_agent_idx.view(batch_size, 1, 1).expand(-1, 1, feat_dim)
            key_feat = torch.gather(features, 1, idx_exp).squeeze(1)

            combined = torch.cat([local_feat, weak_feats_flat, key_feat], dim=-1)
            v = self.net(combined)
            values.append(v)

        return torch.stack(values, dim=1)  # (batch, n, 1)