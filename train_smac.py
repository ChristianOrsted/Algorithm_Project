from weak_tie_env import WeakTieStarCraft2Env
import numpy as np
import time
import os
import sys
from datetime import datetime
from weak_tie_module import WeakTieGraph
from mappo_agent import WeakTieMAPPOAgent

# ==============================================================================
# 论文复现配置 (Batch Training 版)
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


# ==============================================================================
# 📝 日志系统
# ==============================================================================
class Logger:
    """同时输出到控制台和文件的日志类"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 实时写入文件
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_next_train_number(log_dir='log'):
    """自动获取下一个训练编号"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        return 1
    
    existing_logs = [f for f in os.listdir(log_dir) if f.startswith('train') and f.endswith('.txt')]
    
    if not existing_logs:
        return 1
    
    # 提取所有编号
    numbers = []
    for log_file in existing_logs:
        try:
            # 从 train1.txt 中提取 1
            num = int(log_file.replace('train', '').replace('.txt', ''))
            numbers.append(num)
        except:
            continue
    
    if numbers:
        return max(numbers) + 1
    else:
        return 1


def setup_logger(log_dir='log'):
    """设置日志系统"""
    # 确保目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    train_num = get_next_train_number(log_dir)
    log_file = os.path.join(log_dir, f'train{train_num}.txt')
    
    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger
    
    print("="*60)
    print(f"训练开始 - 第 {train_num} 次训练")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print()
    
    return train_num


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
    # 设置日志系统
    train_num = setup_logger()
    
    # 记录配置信息
    print(f"地图名称: {MAP_NAME}")
    print(f"训练回合数: {N_EPISODES}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"PPO轮数: {PPO_EPOCH}")
    print(f"隐藏层维度: {HIDDEN_DIM}")
    print(f"学习率: {LR}")
    print(f"评估间隔: {EVAL_INTERVAL}")
    print()
    
    try:
        env = WeakTieStarCraft2Env(map_name=MAP_NAME, difficulty="2", window_size_x=800, window_size_y=600)
    except Exception as e:
        print(f"环境启动失败: {e}")
        return

    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    obs_dim = env_info["obs_shape"]

    print(f"--- Batch Training 启动: {MAP_NAME} ---")
    print(f"--- Batch Size: {BATCH_SIZE}, Epochs: {PPO_EPOCH} ---")
    print(f"--- 智能体数量: {n_agents}, 观测维度: {obs_dim}, 动作数: {n_actions} ---")
    print()

    wt_graph = WeakTieGraph(n_agents, obs_range=OBS_RANGE, alpha_quantile=0.3)

    agent = WeakTieMAPPOAgent(n_agents, obs_dim, n_actions,
                              hidden_dim=HIDDEN_DIM, lr=LR,
                              ppo_epoch=PPO_EPOCH, mini_batch_size=MINI_BATCH_SIZE)

    if os.path.exists(MODEL_PATH):
        print("发现旧模型，删除以重新训练...")
        try:
            os.remove(MODEL_PATH)
        except:
            pass

    best_win_rate = 0.0
    total_wins = 0
    recent_raw_rewards = []

    batch_buffer = []
    
    training_start_time = time.time()

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

        res_str = "WIN" if is_win else "LOSE"
        if episode % 10 == 0:
            elapsed_time = time.time() - training_start_time
            print(f"Ep {episode} | RawRew: {raw_reward:.2f} | {res_str} | Wins: {total_wins} | Time: {elapsed_time/60:.1f}min")

        if episode % 200 == 0:
            avg_rew = np.mean(recent_raw_rewards) if recent_raw_rewards else 0
            status = "提升中" if avg_rew > 5.0 else "探索中"
            print(f"\n==============================================")
            print(f"[趋势] 第 {episode - 199}~{episode} 轮 ({status})")
            print(f"平均得分: {avg_rew:.2f}")
            print(f"当前胜场: {total_wins}")
            print(f"总胜率: {total_wins/episode*100:.2f}%")
            print(f"==============================================\n")
            recent_raw_rewards = []

        if episode % EVAL_INTERVAL == 0:
            print(f">>> 正在评估 ({EVAL_EPISODES}局)...")
            eval_wins = 0
            eval_rewards = []
            for _ in range(EVAL_EPISODES):
                _, raw_rew, win, _, _ = run_episode(env, agent, wt_graph, train_mode=False)
                if win: eval_wins += 1
                eval_rewards.append(raw_rew)

            win_rate = eval_wins / EVAL_EPISODES
            avg_eval_reward = np.mean(eval_rewards)
            print(f">>> 真实胜率: {win_rate * 100:.1f}% | 平均得分: {avg_eval_reward:.2f}")

            if win_rate >= best_win_rate and win_rate > 0:
                best_win_rate = win_rate
                agent.save_model(MODEL_PATH)
                print(f">>> 模型已保存 (胜率 {best_win_rate:.1%})")

    env.close()
    
    total_time = time.time() - training_start_time
    print("\n" + "="*60)
    print("训练完成!")
    print(f"最终统计:")
    print(f"总回合数: {N_EPISODES}")
    print(f"总胜场数: {total_wins}")
    print(f"最终胜率: {total_wins/N_EPISODES*100:.2f}%")
    print(f"最佳评估胜率: {best_win_rate*100:.1f}%")
    print(f"总训练时间: {total_time/3600:.2f} 小时")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
