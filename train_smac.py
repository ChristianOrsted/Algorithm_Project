from weak_tie_env import WeakTieStarCraft2Env
import numpy as np
import time
import os
from weak_tie_module import WeakTieGraph
from mappo_agent import WeakTieMAPPOAgent

# ==============================================================================
# 📜 论文复现配置 (Batch Training 版)
# ==============================================================================
MAP_NAME = "2s3z"
N_EPISODES = 50000
BATCH_SIZE = 32
PPO_EPOCH = 15
MINI_BATCH_SIZE = 8

OBS_RANGE = 15.0
EVAL_INTERVAL = 500
EVAL_EPISODES = 20
MODEL_PATH = "best_model.pt"
STEP_DELAY = 0.0

ENTROPY_START = 0.2
ENTROPY_END = 0.05
ENTROPY_DECAY_EPISODES = 100000

if MAP_NAME in ["1c3s5z", "50m", "10m_vs_11m"]:
    HIDDEN_DIM = 256
    LR = 0.0003
    print(f">>> [困难地图] {MAP_NAME} -> 应用论文参数: Hidden={HIDDEN_DIM}, LR={LR}")
else:
    HIDDEN_DIM = 128
    LR = 0.0005
    print(f">>> [常规地图] {MAP_NAME} -> 应用参数: Hidden={HIDDEN_DIM}, LR={LR}")


def get_current_entropy(episode):
    if episode > ENTROPY_DECAY_EPISODES:
        return ENTROPY_END
    frac = 1.0 - (episode / ENTROPY_DECAY_EPISODES)
    return ENTROPY_END + frac * (ENTROPY_START - ENTROPY_END)


def run_episode(env, agent, wt_graph, train_mode=True):
    obs, state = env.reset()
    terminated = False

    episode_reward = 0
    raw_episode_reward = 0

    actor_hidden = agent.init_hidden(batch_size=1)

    episode_buffer = {'obs': [], 'acts': [], 'rewards': [], 'dones': [],
                      'avails': [], 'probs': [], 'masks': [], 'keys': []}

    while not terminated:
        avail_actions = env.get_avail_actions()
        positions = env.get_all_unit_positions()
        alive_mask = np.any(positions != 0, axis=1)

        mask_beta, key_agent_idx = wt_graph.compute_graph_info(positions, alive_mask)

        actions, probs, next_hidden = agent.select_action(
            obs, avail_actions, mask_beta, key_agent_idx, actor_hidden,
            deterministic=(not train_mode)
        )

        reward, terminated, info = env.step(actions)
        next_obs = env.get_obs()

        shaped_reward = reward / 5.0

        if train_mode:
            episode_buffer['obs'].append([obs])
            episode_buffer['acts'].append([actions])
            episode_buffer['rewards'].append([[shaped_reward] * len(actions)])
            episode_buffer['dones'].append([[float(terminated)] * len(actions)])
            episode_buffer['avails'].append([avail_actions])
            episode_buffer['probs'].append([probs])
            episode_buffer['masks'].append([mask_beta])
            episode_buffer['keys'].append([key_agent_idx])

        obs = next_obs
        actor_hidden = next_hidden
        episode_reward += shaped_reward
        raw_episode_reward += reward

    is_win = info.get('battle_won', False)
    next_data = None

    return episode_reward, raw_episode_reward, is_win, episode_buffer, next_data


def main():
    try:
        env = WeakTieStarCraft2Env(map_name=MAP_NAME, difficulty="2", window_size_x=800, window_size_y=600)
    except Exception as e:
        print(f"❌ 环境启动失败: {e}")
        return

    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    obs_dim = env_info["obs_shape"]

    print(f"--- 🚀 Batch Training 启动: {MAP_NAME} ---")
    print(f"--- Batch Size: {BATCH_SIZE}, Epochs: {PPO_EPOCH} ---")

    wt_graph = WeakTieGraph(n_agents, obs_range=OBS_RANGE, alpha_quantile=0.3)

    agent = WeakTieMAPPOAgent(n_agents, obs_dim, n_actions,
                              hidden_dim=HIDDEN_DIM, lr=LR,
                              ppo_epoch=PPO_EPOCH, mini_batch_size=MINI_BATCH_SIZE)

    if os.path.exists(MODEL_PATH):
        print("⚠️ 发现旧模型，删除以重新训练...")
        try:
            os.remove(MODEL_PATH)
        except:
            pass

    best_win_rate = 0.0
    total_wins = 0
    recent_raw_rewards = []

    batch_buffer = []

    for episode in range(1, N_EPISODES + 1):
        curr_entropy = get_current_entropy(episode)

        _, raw_reward, is_win, buffer, next_data = run_episode(env, agent, wt_graph, train_mode=True)

        batch_buffer.append((buffer, next_data))

        if is_win: total_wins += 1
        recent_raw_rewards.append(raw_reward)

        if len(batch_buffer) >= BATCH_SIZE:
            loss = agent.update_batch(batch_buffer, entropy_coef=curr_entropy)
            batch_buffer = []
            print(f"🔄 Ep {episode} | Update! Loss: {loss:.4f} | Ent: {curr_entropy:.3f}")

        res_str = "WIN 🚩" if is_win else "LOSE"
        if episode % 10 == 0:
            print(f"Ep {episode} | RawRew: {raw_reward:.2f} | {res_str} | Wins: {total_wins}")

        if episode % 200 == 0:
            avg_rew = np.mean(recent_raw_rewards) if recent_raw_rewards else 0
            status = "✅ 提升中" if avg_rew > 5.0 else "⏳ 探索中"
            print(f"\n==============================================")
            print(f"📈 [趋势] 第 {episode - 199}~{episode} 轮 ({status})")
            print(f"   平均得分: {avg_rew:.2f}")
            print(f"   当前胜场: {total_wins}")
            print(f"==============================================\n")
            recent_raw_rewards = []

        if episode % EVAL_INTERVAL == 0:
            print(f">>> 🔍 正在评估 ({EVAL_EPISODES}局)...")
            eval_wins = 0
            for _ in range(EVAL_EPISODES):
                _, _, win, _, _ = run_episode(env, agent, wt_graph, train_mode=False)
                if win: eval_wins += 1

            win_rate = eval_wins / EVAL_EPISODES
            print(f">>> 📊 真实胜率: {win_rate * 100:.1f}%")

            if win_rate >= best_win_rate and win_rate > 0:
                best_win_rate = win_rate
                agent.save_model(MODEL_PATH)
                print(f">>> 💾 模型已保存 (胜率 {best_win_rate:.1%})")

    env.close()


if __name__ == "__main__":
    main()