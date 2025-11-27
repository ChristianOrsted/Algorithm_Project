from pettingzoo.mpe import simple_spread_v3
import numpy as np
import torch
import time
from weak_tie_module import WeakTieGraph, WeakTieMAPPO_Critic

# --- 配置参数 ---
N_AGENTS = 3
MAX_CYCLES = 25
OBS_RANGE = 0.5  # 论文中的观测范围定义
ALPHA = 0.3  # 弱联系阈值


def get_agent_positions(env):
    """从 MPE 环境中获取真实的智能体位置 (x, y)"""
    # MPE 的 parallel_env 包装器下，通常可以通过 unwrapped 访问底层属性
    # 这里为了通用性，尝试访问底层 agents 对象
    try:
        # 这里的 env.agents 是 agent 名字列表
        # 我们需要访问 env.unwrapped.agents 对象列表来获取物理状态
        positions = []
        for agent in env.unwrapped.agents:
            positions.append(agent.state.p_pos)
        return np.array(positions)
    except:
        # 如果无法直接访问，生成随机位置仅作演示（防止代码报错卡住）
        return np.random.rand(N_AGENTS, 2)


def main():
    # 1. 初始化环境
    env = simple_spread_v3.parallel_env(
        render_mode="human",
        max_cycles=MAX_CYCLES,
        continuous_actions=False,
        N=N_AGENTS
    )
    observations, infos = env.reset()

    # 获取维度信息
    sample_agent = env.agents[0]
    obs_dim = env.observation_space(sample_agent).shape[0]
    act_dim = env.action_space(sample_agent).n

    print(f"环境初始化完成: {N_AGENTS} Agents")
    print(f"Obs Dim: {obs_dim}, Act Dim: {act_dim}")

    # 2. 初始化核心算法模块
    wt_graph = WeakTieGraph(n_agents=N_AGENTS, obs_range=OBS_RANGE, alpha=ALPHA)
    wt_critic = WeakTieMAPPO_Critic(n_agents=N_AGENTS, obs_dim=obs_dim, act_dim=act_dim)

    print("\n>>> 开始弱联系算法运行演示 <<<\n")

    step = 0
    while env.agents:
        # --- 模拟动作选择 (这里用随机动作，实际应替换为 Actor 网络) ---
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        # 环境步进
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # --- 核心环节：获取位置并构建图 ---
        positions = get_agent_positions(env)

        # 计算联系强度矩阵、主导智能体、弱联系掩码
        H, key_agent_idx, mask_beta = wt_graph.compute_tie_strength_matrix(positions)

        # --- 核心环节：准备数据输入网络 ---
        # 将数据转换为 Tensor 并增加 batch 维度
        obs_tensor = torch.tensor(np.array([observations[a] for a in env.agents]), dtype=torch.float32).unsqueeze(0)

        # 将离散动作转换为 One-hot 编码供 Critic 使用
        act_indices = np.array([actions[a] for a in env.agents])
        act_onehot = np.eye(act_dim)[act_indices]
        act_tensor = torch.tensor(act_onehot, dtype=torch.float32).unsqueeze(0)

        mask_beta_tensor = torch.tensor(mask_beta, dtype=torch.float32).unsqueeze(0)
        key_idx_tensor = torch.tensor([key_agent_idx], dtype=torch.long).unsqueeze(0)

        # --- 核心环节：Critic 前向传播 (计算 V 值) ---
        values = wt_critic(obs_tensor, act_tensor, mask_beta_tensor, key_idx_tensor)

        # 打印信息
        print(f"Step {step}:")
        print(f"  - Key Agent Index: {key_agent_idx}")
        print(f"  - Weak Tie Mask (Agent 0视角): {mask_beta[0].astype(int)}")  # 1代表保留(弱联系)，0代表丢弃(强联系)
        print(f"  - Critic Values: {values.detach().numpy().flatten().round(3)}")

        observations = next_obs
        step += 1
        time.sleep(0.1)  # 慢放以便观察

        if all(terminations.values()) or all(truncations.values()):
            break

    env.close()
    print("\n✅ 演示结束")


if __name__ == "__main__":
    main()