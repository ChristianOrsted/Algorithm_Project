import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ["SC2PATH"] = "D:/Program Files (x86)/StarCraft II"

from weak_tie_env import WeakTieStarCraft2Env
from mappo_agent import WeakTieMAPPOAgent
from weak_tie_module import WeakTieGraph
import torch
import numpy as np

def evaluate_model(model_path, n_episodes=20):
    """评估指定模型"""
    try:
        # 创建环境
        env = WeakTieStarCraft2Env(map_name="1c3s5z")
        env_info = env.get_env_info()
        
        # 获取维度信息
        n_agents = env_info["n_agents"]
        obs_dim = env_info["obs_shape"]
        n_actions = env_info["n_actions"]
        
        # ✅ 修正：hidden_dim 改为 256
        agent = WeakTieMAPPOAgent(
            n_agents=n_agents,
            obs_dim=obs_dim,
            act_dim=n_actions,
            hidden_dim=256,  # 🔧 和训练时保持一致
            lr=0.0003,
            gamma=0.99,
            gae_lambda=0.95,
            clip_param=0.2,
            ppo_epoch=10,
            mini_batch_size=8
        )
        
        # 加载模型
        agent.load_model(model_path)
        
        # ✅ 修正：参数名改为 obs_range
        weak_tie_graph = WeakTieGraph(
            n_agents=n_agents, 
            obs_range=15.0,      # 🔧 修改参数名
            alpha_quantile=0.3   # 🔧 添加必需参数
        )
        
        # 评估指标
        wins = 0
        total_reward = 0
        
        for ep in range(n_episodes):
            env.reset()
            episode_reward = 0
            terminated = False
            
            # 初始化隐藏状态
            actor_hidden = agent.init_hidden(batch_size=1)
            
            step_count = 0
            while not terminated:
                # 获取环境信息
                obs = env.get_obs()
                avail_actions = env.get_avail_actions()
                positions = env.get_all_unit_positions()
                
                # 存活掩码
                alive_mask = np.array([1 if env.agents[i].health > 0 else 0 
                       for i in range(n_agents)])
                
                # 计算弱联系图信息
                mask_beta, key_agent_idx = weak_tie_graph.compute_graph_info(
                    positions, alive_mask
                )
                
                # 选择动作
                actions, probs, actor_hidden = agent.select_action(
                    obs=obs,
                    avail_actions=avail_actions,
                    mask=mask_beta,
                    key_idx=key_agent_idx,
                    actor_hidden=actor_hidden,
                    deterministic=True
                )
                
                # 执行动作
                reward, terminated, info = env.step(actions)
                episode_reward += reward
                step_count += 1
                
                # 防止无限循环
                if step_count > 200:
                    break
            
            # 统计结果
            if env.win_counted:
                wins += 1
            total_reward += episode_reward
            
            print(f"Episode {ep+1}/{n_episodes}: "
                  f"{'Win' if env.win_counted else 'Loss'} | "
                  f"Reward: {episode_reward:.2f}")
        
        env.close()
        
        win_rate = wins / n_episodes
        avg_reward = total_reward / n_episodes
        
        return win_rate, avg_reward
    
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0  # 🔧 确保始终返回两个值


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🎮 开始评估模型性能...")
    print("=" * 60)
    
    # 评估 Best Model (Ep 1500)
    print("\n📊 评估 Best Model (Ep 1500)...")
    print("-" * 60)
    best_win_rate, best_avg_reward = evaluate_model("best_model.pt", n_episodes=20)
    print(f"\n✅ Best Model 结果:")
    print(f"   胜率: {best_win_rate*100:.1f}% ({int(best_win_rate*20)}/20)")
    print(f"   平均得分: {best_avg_reward:.2f}")
    
    # 评估 Latest Checkpoint (Ep 5000)
    print("\n" + "=" * 60)
    print("📊 评估 Latest Checkpoint (Ep 5000)...")
    print("-" * 60)
    latest_win_rate, latest_avg_reward = evaluate_model(
        "checkpoints/ckpt_latest.pt", n_episodes=20
    )
    print(f"\n✅ Latest Checkpoint 结果:")
    print(f"   胜率: {latest_win_rate*100:.1f}% ({int(latest_win_rate*20)}/20)")
    print(f"   平均得分: {latest_avg_reward:.2f}")
    
    # 对比结果
    print("\n" + "=" * 60)
    print("📈 对比结果:")
    print("-" * 60)
    
    if best_win_rate > latest_win_rate:
        print(f"🏆 Best Model (Ep 1500) 胜率更高")
        print(f"   优势: {(best_win_rate - latest_win_rate)*100:.1f}%")
        print(f"\n💡 建议: 在 train_smac.py 中设置 RESUME_SOURCE='best'")
    elif latest_win_rate > best_win_rate:
        print(f"🏆 Latest Checkpoint (Ep 5000) 胜率更高")
        print(f"   优势: {(latest_win_rate - best_win_rate)*100:.1f}%")
        print(f"\n💡 建议: 继续使用 RESUME_SOURCE='latest'")
    else:
        print(f"⚖️ 两个模型胜率相同")
        print(f"\n💡 建议: 考虑重新训练（可能陷入局部最优）")
    
    if best_avg_reward > latest_avg_reward:
        print(f"\n📊 平均得分: Best Model 更高 ({best_avg_reward:.2f} vs {latest_avg_reward:.2f})")
    elif latest_avg_reward > best_avg_reward:
        print(f"\n📊 平均得分: Latest Checkpoint 更高 ({latest_avg_reward:.2f} vs {best_avg_reward:.2f})")
    
    print("\n" + "=" * 60)