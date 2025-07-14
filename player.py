import random
import numpy as np

class Player:
    """玩家基类，定义统一接口"""
    def __init__(self, player_id):
        self.player_id = player_id  # 0或1
        
    def serve(self, obs) -> tuple:
        """生成发球动作，返回 (shot_type, landing_pos, height)"""
        raise NotImplementedError

    def generate_shot(self, obs) -> tuple:
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
        
    def serve(self, obs) -> tuple:
        """生成发球动作，需返回 (shot_type, landing_pos, height)"""
        if obs[-1] == 0:
            # 玩家0发球到对方场地右侧（坐标[4,2]）
            return (0, (4, 2), 1)  # (shot_type, landing_pos, height)
        else:
            # 玩家1发球到对方场地左侧（坐标[1,2]）
            return (0, (1, 2), 1)

    def generate_shot(self, obs):
        shot_type = random.randint(0, 9)
        target = (random.uniform(3,5), random.uniform(0,2)) if obs[-1] == 0 else (random.uniform(0,2), random.uniform(0,2))
        return (shot_type, target, 2)
    
    def result(self, obs):
        return random.choices(
            ['成功', '出界', '下网'],
            weights=[60, 20, 20],
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

    def serve(self, obs) -> tuple:
        ty, landing_pos, hit_height = self.act_model.predict(obs)
        if self.player_id == 0:
            landing_pos = np.clip(landing_pos, [2.6, 1.1], [4.9, 1.9])
        else:
            landing_pos = np.clip(landing_pos, [0.1, 0.1], [2.4, 0.9])

        return (ty, landing_pos, hit_height)

    def generate_shot(self, obs) -> tuple:
        ty, landing_pos, hit_height = self.act_model.predict(obs)
        if self.player_id == 0:
            landing_pos = np.clip(landing_pos, [2.6, 0.1], [4.9, 1.9])
        else:
            landing_pos = np.clip(landing_pos, [0.1, 0.1], [2.4, 1.9])

        return (ty, landing_pos, hit_height)

    def result(self, obs):
        return self.result_model.predict(obs)

    def hit(self, obs):
        return self.defense_model.predict(obs)