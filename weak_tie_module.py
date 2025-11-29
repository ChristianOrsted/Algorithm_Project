import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from scipy.spatial.distance import cdist


class WeakTieGraph:
    """
    [论文 3.3 节] 弱联系图构建模块
    """

    def __init__(self, n_agents, obs_range=15.0, alpha_quantile=0.3):
        self.n_agents = n_agents
        self.obs_range = obs_range
        self.alpha_quantile = alpha_quantile

    def compute_graph_info(self, agent_positions, alive_mask=None):
        n = self.n_agents
        G = nx.Graph()
        G.add_nodes_from(range(n))

        valid_indices = [i for i in range(n) if (alive_mask is None or alive_mask[i])]

        # 1. 基础建边 (Definition 1)
        if len(valid_indices) > 0:
            pos_valid = agent_positions[valid_indices]
            # 使用 scipy 加速距离计算
            dists = cdist(pos_valid, pos_valid)
            for i_idx, i in enumerate(valid_indices):
                for j_idx, j in enumerate(valid_indices):
                    if i < j and dists[i_idx, j_idx] <= self.obs_range:
                        G.add_edge(i, j)

        # 2. 保证连通性 (Definition 2 - 改进版：连接最近邻)
        if len(valid_indices) > 1:
            subgraph = G.subgraph(valid_indices)
            if not nx.is_connected(subgraph):
                comps = list(nx.connected_components(subgraph))
                for k in range(len(comps) - 1):
                    comp1 = list(comps[k])
                    comp2 = list(comps[k + 1])
                    pos1 = agent_positions[comp1]
                    pos2 = agent_positions[comp2]
                    dists = cdist(pos1, pos2)
                    min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                    G.add_edge(comp1[min_idx[0]], comp2[min_idx[1]])

        # 3. 寻找 Key Agent (Definition 5)
        degrees = dict(G.degree())
        valid_degrees = {k: v for k, v in degrees.items() if k in valid_indices}
        key_agent_idx = max(valid_degrees, key=valid_degrees.get) if valid_degrees else 0

        # 4. 计算联系强度 H (Eq. 8)
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        H = np.zeros((n, n))
        tie_values = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    H[i][j] = 1.0
                    continue
                if i not in valid_indices or j not in valid_indices:
                    H[i][j] = 0.0
                    continue

                D_i = degrees[i]
                D_j = degrees[j]
                W_ij = path_lengths[i].get(j, 9999)
                denominator = D_i + D_j + W_ij - 2
                strength = 1.0 / denominator if denominator > 0 else 0.0
                H[i][j] = strength
                tie_values.append(strength)

        # 5. 动态 Alpha 阈值 (论文核心)
        current_alpha = np.quantile(tie_values, self.alpha_quantile) if tie_values else 0.0

        # 6. 生成掩码
        mask_beta = (H <= current_alpha).astype(np.float32)
        np.fill_diagonal(mask_beta, 1.0)

        if alive_mask is not None:
            dead_indices = np.where(alive_mask == 0)[0]
            mask_beta[dead_indices, :] = 0
            mask_beta[:, dead_indices] = 0

        return mask_beta, key_agent_idx


class WeakTieFusionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WeakTieFusionLayer, self).__init__()
        # 输入: Local + Weak(Global) + Key
        self.fc = nn.Linear(input_dim * 3, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, local_feat, global_feat, mask_beta, key_agent_idx):
        batch_size, n_agents, feat_dim = local_feat.shape

        # 1. 弱联系特征聚合 (加权平均)
        mask = mask_beta.unsqueeze(-1)
        global_feat_exp = global_feat.unsqueeze(0).expand(n_agents, -1, -1, -1).permute(1, 0, 2, 3)
        weak_feats = global_feat_exp * mask
        weak_feat_agg = (weak_feats.sum(dim=2) / (mask.sum(dim=2) + 1e-6))

        # 2. Key Agent 特征
        idx_exp = key_agent_idx.view(batch_size, 1, 1).expand(-1, 1, feat_dim)
        key_feat = torch.gather(global_feat, 1, idx_exp).expand(-1, n_agents, -1)

        # 3. 拼接与映射
        combined = torch.cat([local_feat, weak_feat_agg, key_feat], dim=-1)
        out = self.fc(combined)
        return self.layer_norm(out)


class WeakTieNet(nn.Module):
    """
    带有 GRU 和 LayerNorm 的 SOTA 网络结构
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WeakTieNet, self).__init__()
        self.fusion = WeakTieFusionLayer(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, local_x, global_x, mask, key_idx, hidden_state):
        # 1. 融合
        x = self.fusion(local_x, global_x, mask, key_idx)
        x = self.relu(x)

        # 2. GRU 记忆
        B, N, F = x.shape
        x_flat = x.reshape(-1, F)
        h_flat = hidden_state.reshape(-1, F)
        h_new = self.gru(x_flat, h_flat)

        # 3. 输出
        out = self.fc_out(h_new)
        return out.reshape(B, N, -1), h_new.reshape(B, N, -1)