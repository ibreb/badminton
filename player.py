import random
import numpy as np
import torch
import torch.nn.functional as F

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
        return (0, 4, 1)
        """生成发球动作，需返回 (shot_type, landing_pos, height)"""
        # if obs[-1] == 0:
        #     # 玩家0发球到对方场地右侧（坐标[4,2]）
        #     return (0, (4, 2), 1)  # (shot_type, landing_pos, height)
        # else:
        #     # 玩家1发球到对方场地左侧（坐标[1,2]）
        #     return (0, (1, 2), 1)

    def generate_shot(self, obs):
        shot_type = random.randint(0, 9)
        # target = (random.uniform(3,5), random.uniform(0,2)) if obs[-1] == 0 else (random.uniform(0,2), random.uniform(0,2))
        target = random.randint(1, 9)
        return (shot_type, target, 2)
    
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

    def serve(self, obs) -> tuple:
        ty, landing_pos, hit_height = self.act_model.predict(obs)
        # if self.player_id == 0:
        #     landing_pos = (4, 2)  # np.clip(landing_pos, [2.6, 1.1], [4.9, 1.9])
        # else:
        #     landing_pos = (1, 0)  # np.clip(landing_pos, [0.1, 0.1], [2.4, 0.9])

        return (ty, 1, hit_height)

    def generate_shot(self, obs) -> tuple:
        ty, landing_pos_idx, hit_height = self.act_model.predict(obs)
        
        # if self.player_id == 0:
        #     landing_pos = [
        #         (2.9166666666666665, 0.3333333333333333),
        #         (2.9166666666666665, 1.0),
        #         (2.9166666666666665, 1.6666666666666667),
        #         (3.75, 0.3333333333333333),
        #         (3.75, 1.0),
        #         (3.75, 1.6666666666666667),
        #         (4.583333333333334, 0.3333333333333333),
        #         (4.583333333333334, 1.0),
        #         (4.583333333333334, 1.6666666666666667)
        #     ][landing_pos_idx]
        # else:
        #     landing_pos = [
        #         (0.4166666666666667, 0.3333333333333333),
        #         (0.4166666666666667, 1.0),
        #         (0.4166666666666667, 1.6666666666666667),
        #         (1.25, 0.3333333333333333),
        #         (1.25, 1.0),
        #         (1.25, 1.6666666666666667),
        #         (2.0833333333333335, 0.3333333333333333),
        #         (2.0833333333333335, 1.0),
        #         (2.0833333333333335, 1.6666666666666667)
        #     ][landing_pos_idx]
        return (ty, landing_pos_idx, hit_height)

    def result(self, obs):
        return self.result_model.predict(obs)

    def hit(self, obs):
        return self.defense_model.predict(obs)

class GreedyPlayer(DLPlayer):
    def __init__(self, player_id, result_model, defense_model, act_model):
        super().__init__(player_id, result_model, defense_model, act_model)

    def generate_shot(self, state) -> tuple:
        max_prob = 0
        for ty in range(10):
            for landing_pos_idx in range(9):
                for hit_height in range(3):
                    if self.player_id == 0:
                        landing_pos = [
                            (2.9166666666666665, 0.3333333333333333),
                            (2.9166666666666665, 1.0),
                            (2.9166666666666665, 1.6666666666666667),
                            (3.75, 0.3333333333333333),
                            (3.75, 1.0),
                            (3.75, 1.6666666666666667),
                            (4.583333333333334, 0.3333333333333333),
                            (4.583333333333334, 1.0),
                            (4.583333333333334, 1.6666666666666667)
                        ][landing_pos_idx]
                    else:
                        landing_pos = [
                            (0.4166666666666667, 0.3333333333333333),
                            (0.4166666666666667, 1.0),
                            (0.4166666666666667, 1.6666666666666667),
                            (1.25, 0.3333333333333333),
                            (1.25, 1.0),
                            (1.25, 1.6666666666666667),
                            (2.0833333333333335, 0.3333333333333333),
                            (2.0833333333333335, 1.0),
                            (2.0833333333333335, 1.6666666666666667)
                        ][landing_pos_idx]
                    action = (ty, landing_pos, hit_height)

                    obs = state.copy()
                    obs.append(action[0])
                    obs.append(action[1][0])
                    obs.append(action[1][1])
                    obs.append(action[2])

                    obs = torch.FloatTensor(obs).unsqueeze(0)

                    with torch.no_grad():
                        logits = self.result_model(obs)
                        probs = F.softmax(logits, dim=1)
                        prob = probs[0][0].item()

                    if prob > max_prob:
                        max_prob = prob
                        final_action = action
        # print('max_prob:', max_prob)
        # print('action:', final_action)
        return final_action