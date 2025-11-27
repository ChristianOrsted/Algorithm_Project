from pettingzoo.mpe import simple_spread_v3
import time

# 创建环境（带可视化）
env = simple_spread_v3.parallel_env(
    render_mode="human",
    max_cycles=500,
    continuous_actions=False  # 使用离散动作
)

observations, infos = env.reset()
print(f"🎮 智能体: {env.agents}")
print(f"📊 观测空间: {env.observation_space('agent_0')}")
print(f"🎯 动作空间: {env.action_space('agent_0')}")

episode_rewards = {agent: 0 for agent in env.agents}

for step in range(500):
    # 所有智能体随机动作
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # 累计奖励
    for agent in env.agents:
        episode_rewards[agent] += rewards[agent]
    
    # 打印进度
    if step % 50 == 0:
        avg_reward = sum(rewards.values()) / len(rewards)
        print(f"⏱️  Step {step}, Avg Reward: {avg_reward:.3f}")
    
    # 检查是否结束
    if all(terminations.values()) or all(truncations.values()):
        print(f"🏁 Episode 结束于 Step {step}")
        break
    
    time.sleep(0.03)  # 控制播放速度

print(f"\n✅ 最终奖励: {episode_rewards}")
env.close()