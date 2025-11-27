from smac.env import StarCraft2Env
import numpy as np
import time

# 创建环境时添加可视化参数
env = StarCraft2Env(
    map_name="2s3z",
    window_size_x=1920,      # 窗口宽度
    window_size_y=1200,      # 窗口高度
    replay_dir="replays",    # 保存回放
    replay_prefix="3m_test", # 回放前缀
    step_mul=16,              # 每步的游戏帧数（越大越快，建议8-16）
    difficulty="7"           # 难度
)

env_info = env.get_env_info()
n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]

print(f"地图: 3m")
print(f"智能体数量: {n_agents}")
print(f"动作数量: {n_actions}")

env.reset()

terminated = False
episode_reward = 0
step_count = 0

while not terminated:
    avail_actions = env.get_avail_actions()
    
    actions = []
    for agent_id in range(n_agents):
        avail = np.atleast_1d(avail_actions[agent_id])
        available_action_ids = np.nonzero(avail)[0]
        
        if len(available_action_ids) > 0:
            action = np.random.choice(available_action_ids)
        else:
            action = 0
        
        actions.append(action)
    
    reward, terminated, info = env.step(actions)
    episode_reward += reward
    step_count += 1
    
    # 添加延迟，让你能看清楚
    time.sleep(1)
    
    if step_count % 20 == 0:
        print(f"Step {step_count}, Reward: {reward:.2f}")
    
    if step_count > 1000:
        break

print(f"\n✅ Episode 结束")
print(f"总奖励: {episode_reward:.2f}")
print(f"总步数: {step_count}")

env.close()
input("\n按Enter键退出...")