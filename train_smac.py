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
# 🔥 优化后的训练配置 - 修复"送死路线"问题
# ==============================================================================
MAP_NAME = "1c3s5z"
N_EPISODES = 15000

# 【修改1】调整 Batch 和 PPO 参数，提升训练稳定性
BATCH_SIZE = 32
MINI_BATCH_SIZE = 16      # 从 32 降低到 16，增加更新频率
PPO_EPOCH = 15            # 从 10 提升到 15，充分利用经验

OBS_RANGE = 15.0
EVAL_INTERVAL = 500       # 评估间隔
EVAL_EPISODES = 50        # 【修改2】从 20 提升到 50，减少运气因素
MODEL_PATH = "best_model.pt"

# --- 断点续训配置 ---
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 1000

# 【修改3】恢复策略改为 "none"，清除错误经验重新训练
# 可选值:
#   "ckpt"  -> 强制加载 ckpt_latest.pt
#   "best"  -> 强制加载 best_model.pt
#   "latest"-> 自动比较两者，谁的轮数大加载谁
#   "none"  -> 强制从头开始（推荐用于修复错误策略）
RESUME_SOURCE = "none"

# 提速优化
GRAPH_UPDATE_INTERVAL = 3
STEP_DELAY = 0.0

# 【修改4】熵系数 - 恢复探索能力（核心修复）
ENTROPY_START = 0.05      # 从 0.01 提升到 0.05
ENTROPY_END = 0.001       # 从 0.01 降低到 0.001
ENTROPY_DECAY_EPISODES = 3000  # 从 1 延长到 3000，让探索贯穿前半训练

# 【修改5】学习率降低，防止遗忘过快
if MAP_NAME in ["1c3s5z", "50m", "10m_vs_11m"]:
    HIDDEN_DIM = 256
    LR = 0.0001           # 从 0.0003 或 0.0005 降低
else:
    HIDDEN_DIM = 128
    LR = 0.0001


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
    """渐进式熵衰减"""
    if episode > ENTROPY_DECAY_EPISODES:
        return ENTROPY_END
    frac = 1.0 - (episode / ENTROPY_DECAY_EPISODES)
    return max(ENTROPY_END, ENTROPY_END + frac * (ENTROPY_START - ENTROPY_END))


def peek_model_episode(path, device):
    """只读取模型文件中的 episode 信息，不加载参数"""
    if not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location=device)
        return ckpt.get('episode', 0)
    except Exception as e:
        print(f"警告: 无法读取文件 {path}: {e}")
        return None


def run_episode(env, agent, wt_graph, train_mode=True, episode_num=0):
    """
    【修改6】奖励塑形优化 - 增强信号强度
    """
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

        # 【修改6】改进奖励塑形
        # 原代码: shaped_reward = reward / 5.0  # 过度削弱信号
        # 新策略: 早期死亡惩罚 + 轻度缩放
        if episode_reward == 0 and step_count < 50 and terminated and not info.get('battle_won', False):
            # 早期死亡（送死）严重惩罚
            shaped_reward = reward - 0.3
        else:
            # 正常战斗：保持原始奖励（或轻度缩放）
            shaped_reward = reward  # 直接使用原始奖励，让信号更强
            # 如果觉得波动太大，可以改为: shaped_reward = reward / 2.0

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

    # ===== 关闭渲染，启用无头模式 =====
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    try:
        env = WeakTieStarCraft2Env(
            map_name=MAP_NAME, 
            difficulty="1",  # 最低难度，便于初期学习
            window_size_x=640,
            window_size_y=480
        )
        print("环境已启动（无渲染模式，性能优化已开启）")
    except Exception as e:
        print(f"环境启动失败: {e}")
        try:
            env = WeakTieStarCraft2Env(
                map_name=MAP_NAME,
                difficulty="1",
                window_size_x=640,
                window_size_y=480,
                disable_fog=False
            )
            print("环境已启动（备选无渲染模式）")
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
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"已创建文件夹: {CHECKPOINT_DIR}")
    
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"已创建文件夹: {model_dir}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, "ckpt_latest.pt")
    best_path = MODEL_PATH

    ckpt_ep = peek_model_episode(ckpt_path, device)
    best_ep = peek_model_episode(best_path, device)

    print(f"\n存档状态扫描:")
    print(f"[Ckpt]  自动存档: {'不存在' if ckpt_ep is None else f'Ep {ckpt_ep}'}")
    print(f"[Best]  最佳模型: {'不存在' if best_ep is None else f'Ep {best_ep}'}")

    start_episode = 0
    target_file = None

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

    # 【修改7】多指标评估：同时跟踪胜率和平均得分
    best_win_rate = 0.0
    best_avg_reward = -999.0  # 新增：防止低质量模型被保存
    total_wins = 0
    recent_raw_rewards = []
    batch_buffer = []

    training_start_time = time.time()

    print(f"\n正式开始训练 (从 Ep {start_episode + 1} 到 {N_EPISODES})")
    print(f"配置: LR={LR}, Entropy={ENTROPY_START}→{ENTROPY_END}, PPO_Epoch={PPO_EPOCH}\n")

    for episode in range(start_episode + 1, N_EPISODES + 1):
        curr_entropy = get_current_entropy(episode)

        _, raw_reward, is_win, buffer, _ = run_episode(env, agent, wt_graph, train_mode=True, episode_num=episode)

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

            print(f"\n=== [趋势报告] Ep {episode} ===")
            print(f"平均得分: {avg_rew:.2f}")
            print(f"当前运行胜场: {total_wins}/{current_session_episodes} ({win_rate:.2f}%)")
            print(f"探索系数: {curr_entropy:.4f}")
            print(f"============================\n")
            recent_raw_rewards = []

        # 【修改8】改进评估和保存逻辑
        if episode % EVAL_INTERVAL == 0:
            print(f"\n>>> 评估模式 ({EVAL_EPISODES}局)...")
            eval_wins = 0
            eval_rewards = []
            for _ in range(EVAL_EPISODES):
                _, raw_rew, win, _, _ = run_episode(env, agent, wt_graph, train_mode=False)
                if win: eval_wins += 1
                eval_rewards.append(raw_rew)

            curr_win_rate = eval_wins / EVAL_EPISODES
            avg_eval_reward = np.mean(eval_rewards)
            print(f">>> 评估胜率: {curr_win_rate * 100:.1f}% | 平均得分: {avg_eval_reward:.2f}")

            # 多指标评估：优先胜率，其次得分
            should_save = False
            if curr_win_rate > best_win_rate:
                should_save = True
                save_reason = f"胜率提升 {best_win_rate:.1%} → {curr_win_rate:.1%}"
            elif curr_win_rate == best_win_rate and curr_win_rate > 0:
                if avg_eval_reward > best_avg_reward:
                    should_save = True
                    save_reason = f"胜率持平但得分提升 {best_avg_reward:.2f} → {avg_eval_reward:.2f}"
                else:
                    print(f">>> 胜率持平但得分未提升，保留原模型")
            
            if should_save:
                best_win_rate = curr_win_rate
                best_avg_reward = avg_eval_reward
                agent.save_model(MODEL_PATH, episode)
                print(f">>> 最佳模型已更新 @ Ep {episode}")
                print(f"    原因: {save_reason}")

        if episode % CHECKPOINT_INTERVAL == 0:
            ckpt_save_path = os.path.join(CHECKPOINT_DIR, "ckpt_latest.pt")
            agent.save_model(ckpt_save_path, episode)
            print(f">>> 安全存档已更新: {ckpt_save_path}")

    env.close()
    print("\n训练完成！")
    print(f"最终最佳胜率: {best_win_rate:.1%} (得分 {best_avg_reward:.2f})")


if __name__ == "__main__":
    main()
