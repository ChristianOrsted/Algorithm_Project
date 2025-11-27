from smac.env import StarCraft2Env
import numpy as np

class WeakTieStarCraft2Env(StarCraft2Env):
    """
    继承自 SMAC 环境，增加获取单位绝对坐标的接口。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_tags = []

    def reset(self):
        """重置环境并记录所有智能体的 Tag，保证ID顺序一致"""
        obs, state = super().reset()
        # 记录初始时的所有单位Tag，确保后续索引对齐
        self.agent_tags = sorted(list(self.agents.keys()))
        return obs, state

    def get_all_unit_positions(self):
        """
        [核心功能] 获取所有智能体的绝对坐标 (x, y)。
        即使智能体死亡，也返回 [0, 0] 以保持数组形状 (n_agents, 2)。
        """
        positions = []
        for tag in self.agent_tags:
            if tag in self.agents:
                unit = self.agents[tag]
                # 直接访问 PySC2 Unit 对象的 pos 属性
                positions.append([unit.pos.x, unit.pos.y])
            else:
                # 死亡单位坐标归零
                positions.append([0.0, 0.0])
        return np.array(positions)