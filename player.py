import random
import numpy as np
import torch
import torch.nn.functional as F

class Player:
    """玩家基类，定义统一接口"""
    def __init__(self, player_id):
        self.player_id = player_id  # 0或1
        
    def serve(self, obs) -> list:
        """生成发球动作，返回 (shot_type, landing_pos, height)"""
        raise NotImplementedError

    def generate_shot(self, obs) -> list:
        """生成击球动作"""
        raise NotImplementedError

    def result(self, obs):
        """根据状态和动作生成结果（出界/下网/成功）"""
        raise NotImplementedError
    
    def hit(self, obs):
        """根据状态和动作预测是否接到球"""
        raise NotImplementedError


class SamplePlayer(Player):
    def __init__(self, player_id):
        self.player_id = player_id
        
    def serve(self, obs):
        return [0, 4, 1]

    def generate_shot(self, obs):
        shot_type = random.randint(1, 9)
        target = random.randint(1, 9)
        return [shot_type, target, 2]
    
    def result(self, obs):
        return random.choices(
            ['成功', '出界', '下网'],
            weights=[80, 10, 10],
            k=1
        )[0]

    def hit(self, obs):
        return random.randint(0, 4) > 0


class DLPlayer(Player):
    def __init__(self, player_id, result_model, defense_model, act_model):
        """
        result_model: 结果模型（用于判断击球结果）
        defense_model: 防守模型（用于判断是否接到对手击球）
        act_model: 决策模型（用于生成击球动作）
        """
        super().__init__(player_id)
        self.result_model = result_model
        self.defense_model = defense_model
        self.act_model = act_model
        self.result_model.eval()
        self.defense_model.eval()
        self.act_model.eval()

    def serve(self, obs):
        action = self.act_model.predict(obs)
        if action[0] != 0:
            action[0] = 0  # 保证类型为发球
        if action[1] % 3 == 2:
            action[1] -= 1  # 保证区域合法
        return action

    def generate_shot(self, obs):
        action = self.act_model.predict(obs)
        if action[0] == 0:
            action[0] = 6  # 保证类型不为发球

        if action[0] == 2 and action[1] in [0, 1, 2]:
            action[1] += 3
        if action[0] in [3, 7, 8] and action[1] in [6, 7, 8]:
            action[1] -= 3
        if action[0] in [1, 2] and action[2] in [0, 1]:
            action[2] += 1
        return action

    def result(self, obs):
        return self.result_model.predict(obs)

    def hit(self, obs):
        return self.defense_model.predict(obs)