# player.py
import random
class Player:
    """玩家基类，定义统一接口"""
    def __init__(self, player_id):
        self.player_id = player_id  # 0或1
        self.shot_count = 0  # 击球计数器
        
    def serve(self, env_state) -> tuple:
        """生成发球动作，需返回 (shot_type, landing_pos, height)"""
        if env_state[-1] == 0:
            # 玩家0发球到对方场地右侧（坐标[4,2]）
            return (0, (4, 2), 1)  # (shot_type, landing_pos, height)
        else:
            # 玩家1发球到对方场地左侧（坐标[1,2]）
            return (0, (1, 2), 1)
        

    def generate_shot(self, env_state):
        """击球动作生成，前两次强制成功，第三次概率性失败"""
        self.shot_count += 1
        
        # 前两次强制成功（动作类型1或5）
        if self.shot_count <= 2:
            return self._safe_shot(env_state)
        
        # 第三次开始按概率触发失败
        return self._random_failure_shot(env_state)
    
    def _safe_shot(self, env_state):
        """安全击球（类型1或5）"""
        shot_type = random.choice([1, 5])  # 长球/平球
        target = (4, 1) if env_state[-1] == 0 else (2, 1)  # 合法落点
        return (shot_type, target, 2)  # 高度2避免挂网

    def _random_failure_shot(self, env_state):
        """随机失败类型：40%出界，30%挂网，30%对手致胜"""
        failure_type = random.choices(
            [2, 3, 4],       # 对应挂网/对手致胜/出界
            weights=[40, 30, 30],
            k=1
        )[0]
        
        
        # 根据失败类型生成动作
        if failure_type == 2:    # 挂网
            return (2, (2.5, 1), -1)  # 推球 + 高度 0
        elif failure_type == 3: # 对手致胜
            if env_state[-1]==0:
                return (3, (5, 2), -1)
            else:
                return (3, (0, 0), -1)
        else:                   # 出界
            return (4, (6, 3), 2) if env_state[-1] == 0 else (4, (-1, 2), 2)

