from weak_tie_env import WeakTieStarCraft2Env
import numpy as np
import time
import os
import sys
import glob
from datetime import datetime
from weak_tie_module import WeakTieGraph
from mappo_agent import WeakTieMAPPOAgent
import torch

# ==============================================================================
# 论文复现配置 (自主选择模型版)
# ==============================================================================
MAP_NAME = "1c3s5z"
N_EPISODES = 100000
BATCH_SIZE = 32
MINI_BATCH_SIZE = 32
PPO_EPOCH = 10

OBS_RANGE = 15.0
EVAL_INTERVAL = 500
EVAL_EPISODES = 20
MODEL_PATH = "best_model.pt"

# --- 断点续训配置 ---
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 1000  # 每 1000 轮覆盖更新一次存档

# [新功能] 选择你的恢复策略
# 可选值:
#   "ckpt"  -> 强制加载 ckpt_latest.pt (通常是最近的进度)
#   "best"  -> 强制加载 best_model.pt (如果你觉得之前的模型更好)
#   "latest"-> 自动比较两者，谁的轮数大加载谁
#   "none"  -> 强制从头开始 (Ep 1)
RESUME_SOURCE = "latest"

# 提速优化
GRAPH_UPDATE_INTERVAL = 3
STEP_DELAY = 0.0

# 参数
ENTROPY_START = 0.01
ENTROPY_END = 0.01
ENTROPY_DECAY_EPISODES = 1

if MAP_NAME in ["1c3s5z", "50m", "10m_vs_11m"]:
    HIDDEN_DIM = 256
    LR = 0.0003
else:
    HIDDEN_DIM = 128
    LR = 0.0005


# ==============================================================================
# 日志系统
# ==============================================================================
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_logger(log_dir='log'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'training_log_{MAP_NAME}.txt')
    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger
    print(f"\n{'=' * 60}")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")
    return log_file


def get_current_entropy(episode):
    if episode > ENTROPY_DECAY_EPISODES:
        return ENTROPY_END
    frac = 1.0 - (episode / ENTROPY_DECAY_EPISODES)
    return max(ENTROPY_END, ENTROPY_END + frac * (ENTROPY_START - ENTROPY_END))


def peek_model_episode(path, device):
    """只读取模型文件中的 episode 信息，不加载参数"""
    if not os.path.exists(path):
        return None
    try:
        # map_location 避免显存不足
        ckpt = torch.load(path, map_location=device)
        return ckpt.get('episode', 0)
    except Exception as e:
        print(f"警告: 无法读取文件 {path}: {e}")
        return None


def run_episode(env, agent, wt_graph, train_mode=True):
    obs, state = env.reset()
    terminated = False
    episode_reward = 0
    raw_episode_reward = 0
    actor_hidden = agent.init_hidden(batch_size=1)

    episode_buffer = {'obs': [], 'acts': [], 'rewards': [], 'dones': [],
                      'avails': [], 'probs': [], 'masks': [], 'keys': []}

    step_count = 0
    last_mask_beta = None
    last_key_agent_idx = None

    while not terminated:
        avail_actions = env.get_avail_actions()
        positions = env.get_all_unit_positions()
        alive_mask = np.any(positions != 0, axis=1)

        if step_count % GRAPH_UPDATE_INTERVAL == 0:
            mask_beta, key_agent_idx = wt_graph.compute_graph_info(positions, alive_mask)
            last_mask_beta = mask_beta
            last_key_agent_idx = key_agent_idx
        else:
            mask_beta = last_mask_beta
            key_agent_idx = last_key_agent_idx

        step_count += 1

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
    return episode_reward, raw_episode_reward, is_win, episode_buffer, None


def main():
    setup_logger()

    print(f"地图: {MAP_NAME} | 目标回合: {N_EPISODES}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ===== 【核心修改】关闭渲染，启用无头模式 =====
    # 设置环境变量禁用 SC2 渲染
    os.environ["SDL_VIDEODRIVER"] = "dummy"  # 禁用视频驱动
    
    try:
        env = WeakTieStarCraft2Env(
            map_name=MAP_NAME, 
            difficulty="1",
            # 【关键参数】禁用可视化窗口
            window_size_x=640,  # 保持默认尺寸，但不会渲染
            window_size_y=480
        )
        print("✓ 环境已启动（无渲染模式，性能优化已开启）")
    except Exception as e:
        print(f"环境启动失败: {e}")
        # 如果环境变量方式不起作用，尝试传递 disable_fog 参数
        try:
            env = WeakTieStarCraft2Env(
                map_name=MAP_NAME,
                difficulty="1",
                window_size_x=640,
                window_size_y=480,
                disable_fog=False  # 某些版本支持此参数
            )
            print("✓ 环境已启动（备选无渲染模式）")
        except Exception as e2:
            print(f"备选方案也失败: {e2}")
            return

    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    obs_dim = env_info["obs_shape"]

    wt_graph = WeakTieGraph(n_agents, obs_range=OBS_RANGE, alpha_quantile=0.3)
    agent = WeakTieMAPPOAgent(n_agents, obs_dim, n_actions,
                              hidden_dim=HIDDEN_DIM, lr=LR,
                              ppo_epoch=PPO_EPOCH, mini_batch_size=MINI_BATCH_SIZE)

    # ==========================================================================
    # 智能模型加载逻辑 + 自动创建文件夹
    # ==========================================================================
    # 确保 checkpoints 文件夹存在
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"已创建文件夹: {CHECKPOINT_DIR}")
    
    # 确保 best_model.pt 所在的目录存在
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"已创建文件夹: {model_dir}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, "ckpt_latest.pt")
    best_path = MODEL_PATH

    # 1. 侦察：看看现在硬盘里有哪些存档
    ckpt_ep = peek_model_episode(ckpt_path, device)
    best_ep = peek_model_episode(best_path, device)

    print(f"\n存档状态扫描:")
    print(f"[Ckpt]  自动存档: {'不存在' if ckpt_ep is None else f'Ep {ckpt_ep}'}")
    print(f"[Best]  最佳模型: {'不存在' if best_ep is None else f'Ep {best_ep}'}")

    start_episode = 0
    target_file = None

    # 2. 决策：根据 RESUME_SOURCE 决定加载谁
    if RESUME_SOURCE == "ckpt":
        if ckpt_ep is not None:
            target_file = ckpt_path
            print(f"策略: 强制加载 Ckpt")
        else:
            print(f"策略要求加载 Ckpt 但文件不存在，将尝试 Best 或重头开始。")
            if best_ep is not None: target_file = best_path

    elif RESUME_SOURCE == "best":
        if best_ep is not None:
            target_file = best_path
            print(f"策略: 强制加载 Best Model")
        else:
            print(f"策略要求加载 Best 但文件不存在，将尝试 Ckpt。")
            if ckpt_ep is not None: target_file = ckpt_path

    elif RESUME_SOURCE == "latest":
        print(f"策略: 自动选择轮数最新的模型")
        ep_c = ckpt_ep if ckpt_ep is not None else -1
        ep_b = best_ep if best_ep is not None else -1

        if ep_c > ep_b:
            target_file = ckpt_path
        elif ep_b > -1:
            target_file = best_path

    # 3. 执行加载
    if target_file:
        print(f"最终决定加载: {target_file}")
        start_episode = agent.load_model(target_file)
    else:
        print(f"未找到可用模型或策略设为 none，从头开始训练。")

    if start_episode >= N_EPISODES:
        print("训练目标已达成，无需继续训练。")
        env.close()
        return

    # ==========================================================================

    best_win_rate = 0.0
    total_wins = 0
    recent_raw_rewards = []
    batch_buffer = []

    training_start_time = time.time()

    print(f"\n正式开始训练 (从 Ep {start_episode + 1} 到 {N_EPISODES})...\n")

    for episode in range(start_episode + 1, N_EPISODES + 1):
        curr_entropy = get_current_entropy(episode)

        _, raw_reward, is_win, buffer, _ = run_episode(env, agent, wt_graph, train_mode=True)

        batch_buffer.append((buffer, None))
        if is_win: total_wins += 1
        recent_raw_rewards.append(raw_reward)

        if len(batch_buffer) >= BATCH_SIZE:
            loss = agent.update_batch(batch_buffer, entropy_coef=curr_entropy)
            batch_buffer = []
            print(f"Ep {episode} | Loss: {loss:.4f} | Ent: {curr_entropy:.3f}")

        if episode % 10 == 0:
            res_str = "WIN" if is_win else "LOSE"
            elapsed_time = time.time() - training_start_time
            print(
                f"Ep {episode} | RawRew: {raw_reward:.2f} | {res_str} | Wins: {total_wins} | Time: {elapsed_time / 60:.1f}m")

        if episode % 200 == 0:
            avg_rew = np.mean(recent_raw_rewards) if recent_raw_rewards else 0
            current_session_episodes = episode - start_episode
            win_rate = total_wins / current_session_episodes * 100 if current_session_episodes > 0 else 0

            print(f"\n=== [趋势] Ep {episode} ===")
            print(f"平均得分: {avg_rew:.2f}")
            print(f"当前运行胜场: {total_wins}/{current_session_episodes} ({win_rate:.2f}%)")
            print(f"==========================\n")
            recent_raw_rewards = []

        if episode % EVAL_INTERVAL == 0:
            print(f">>> 评估 ({EVAL_EPISODES}局)...")
            eval_wins = 0
            eval_rewards = []
            for _ in range(EVAL_EPISODES):
                _, raw_rew, win, _, _ = run_episode(env, agent, wt_graph, train_mode=False)
                if win: eval_wins += 1
                eval_rewards.append(raw_rew)

            curr_win_rate = eval_wins / EVAL_EPISODES
            avg_eval_reward = np.mean(eval_rewards)
            print(f">>> 评估胜率: {curr_win_rate * 100:.1f}% | 平均得分: {avg_eval_reward:.2f}")

            # [修改] 只有当胜率【严格大于】历史最佳时才更新，防止同分覆盖
            # 如果你希望最新的同分模型覆盖旧的，改回 >= 即可
            if curr_win_rate > best_win_rate:
                best_win_rate = curr_win_rate
                agent.save_model(MODEL_PATH, episode)
                print(f">>> 最佳模型已更新 (胜率 {best_win_rate:.1%} @ Ep {episode})")
            elif curr_win_rate == best_win_rate and best_win_rate > 0:
                print(f">>> 胜率持平 ({best_win_rate:.1%})，保留原 Best Model (Ep {best_ep})")

        if episode % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, "ckpt_latest.pt")
            agent.save_model(ckpt_path, episode)
            print(f">>> 安全存档已更新: {ckpt_path}")

    env.close()
    print("训练结束！")


if __name__ == "__main__":
    main()
