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

    def generate_shot(self, obs):
        shot_type = random.randint(1, 9)
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
        self.load()

    def load(self):
        self.result_model.load_state_dict(torch.load('result_model.pth'))
        self.result_model.eval()

        self.defense_model.load_state_dict(torch.load('defense_model.pth'))
        self.defense_model.eval()

        self.act_model.load_state_dict(torch.load('act_model.pth'))
        self.act_model.eval()

    def serve(self, obs) -> tuple:
        ty, landing_pos, hit_height = self.act_model.predict(obs)
        return (0, landing_pos + 1, hit_height + 1)

    def generate_shot(self, obs) -> tuple:
        ty, landing_pos, hit_height = self.act_model.predict(obs)
        return (ty, landing_pos + 1, hit_height + 1)

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
                    action = (ty, landing_pos_idx, hit_height)

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