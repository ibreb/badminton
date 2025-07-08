import random

class Player:
    """玩家基类，定义统一接口"""
    def __init__(self, player_id):
        self.player_id = player_id  # 0或1
        
    def serve(self, obs) -> tuple:
        """生成发球动作，需返回 (shot_type, landing_pos, height)"""
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



class SamplePlayer:
    def __init__(self, player_id):
        self.player_id = player_id
        self.shot_count = 0  # 击球计数器
        
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
    
    def result(self, obs, action):
        return random.choices(
            ["success", "出界", "下网"],
            weights=[60, 20, 20],
            k=1
        )[0]

    def hit(self, obs, action):
        return random.randint(0, 4) > 0