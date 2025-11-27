from weak_tie_env import WeakTieStarCraft2Env
import numpy as np
import time
from weak_tie_module import WeakTieGraph
from mappo_agent import WeakTieMAPPOAgent

# --- 配置参数 ---
MAP_NAME = "1c3s5z"  # 使用论文演示的战术地图
N_EPISODES = 5000  # 训练回合数
BATCH_SIZE = 32  # 更新批次大小
OBS_RANGE = 25.0  # 观测半径
ALPHA = 0.3  # 弱联系阈值

# [关键修改] 动作延迟时间（秒）
# 设置为 0.3 秒，您可以根据需要调整：0.5 会更慢，0.1 会稍微快一点
STEP_DELAY = 0.5


def main():
    # 1. 初始化自定义环境
    try:
        # 添加可视化参数 window_size_x 等确保窗口弹出
        # 注意：StarCraft2Env 默认会尝试连接客户端，如果没有弹出窗口，
        # 请确保星际争霸II客户端已正常运行或允许自动启动。
        env = WeakTieStarCraft2Env(
            map_name=MAP_NAME,
            difficulty="3",
            window_size_x=1920,
            window_size_y=1200
        )
    except Exception as e:
        print(f"环境启动失败，请检查地图文件: {e}")
        return

    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    obs_dim = env_info["obs_shape"]

    print(f"环境启动: {MAP_NAME} | Agents: {n_agents}")
    print(f"当前运行速度: 每步延迟 {STEP_DELAY} 秒")

    # 2. 初始化模块
    wt_graph = WeakTieGraph(n_agents, obs_range=OBS_RANGE, alpha=ALPHA)
    agent = WeakTieMAPPOAgent(n_agents, obs_dim, n_actions)

    buffer = {'obs': [], 'acts': [], 'rewards': [], 'dones': [],
              'avails': [], 'probs': [], 'masks': [], 'keys': []}

    for episode in range(N_EPISODES):
        obs, state = env.reset()
        terminated = False
        episode_reward = 0
        step_counter = 0

        while not terminated:
            avail_actions = env.get_avail_actions()

            # --- [核心步骤] 获取绝对坐标 ---
            positions = env.get_all_unit_positions()

            # 生成存活掩码
            alive_mask = np.any(positions != 0, axis=1)

            # --- [核心步骤] 图计算 ---
            mask_beta, key_agent_idx = wt_graph.compute_graph_info(positions, alive_mask)

            # --- 动作选择 ---
            actions, probs = agent.select_action(obs, avail_actions)

            # --- 环境步进 ---
            reward, terminated, info = env.step(actions)
            next_obs = env.get_obs()

            # --- [关键修改] 减慢运行速度 ---
            if STEP_DELAY > 0:
                time.sleep(STEP_DELAY)

            # --- 存储数据 ---
            buffer['obs'].append(obs)
            buffer['acts'].append(actions)
            buffer['rewards'].append([reward] * n_agents)
            buffer['dones'].append([float(terminated)] * n_agents)
            buffer['avails'].append(avail_actions)
            buffer['probs'].append(probs)
            buffer['masks'].append(mask_beta)
            buffer['keys'].append([key_agent_idx])

            obs = next_obs
            episode_reward += reward
            step_counter += 1

        print(f"Episode {episode + 1} | Steps: {step_counter} | Reward: {episode_reward:.2f}")

        # --- 模型更新 ---
        if len(buffer['obs']) >= BATCH_SIZE * 20:
            loss = agent.update(buffer)
            print(f">>> Updated | Loss: {loss:.4f}")
            for k in buffer: buffer[k] = []

    env.close()


if __name__ == "__main__":
    main()