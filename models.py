import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import config

class ResultModel(nn.Module):
    def __init__(self, num_classes=3):
        """
        结果模型：判断当前动作是否成功（成功/出界/下网）
        
        参数：
        - obs_dim: 观测向量的维度
        - num_classes: 输出类别数量
        """
        obs_dim=config.OBS_DIM_RESULT_MODEL
        super(ResultModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, obs):
        """
        输入：当前观测 obs
        输出：'成功', '出界', '下网'
        """
        obs = torch.FloatTensor(obs).unsqueeze(0)  # (1, input_dim)
        with torch.no_grad():
            logits = self(obs)
            probs = F.softmax(logits, dim=1)
            idx = torch.multinomial(probs, num_samples=1).item()
            # print('result probs: ', probs)
            # assert probs[0][0] > probs[0][1] and probs[0][0] > probs[0][2]
        return ['成功', '出界', '下网'][idx]


class DefenseModel(nn.Module):
    def __init__(self):
        """
        防守模型：判断是否能接到对手的击球
        
        参数：
        - obs_dim: 观测向量的维度
        """
        obs_dim=config.OBS_DIM_DEFENSE_MODEL
        super(DefenseModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, obs):
        """
        输入：当前观测 obs
        输出：是否能接到球（True/False）
        """
        obs = torch.FloatTensor(obs).unsqueeze(0)  # (1, input_dim)
        with torch.no_grad():
            prob = self(obs).item()
        # print('hit prob = ', prob)
        # assert prob > 0.5
        return random.random() < prob


class ActModel(nn.Module):
    def __init__(self, shot_types=10, landing_pos_n=9, height_levels=3):
        """
        决策模型：根据观测和对手动作生成击球动作
        
        参数：
        - obs_dim: 观测向量的维度
        - shot_types: 击球类型数量
        - height_levels: 击球高度级数
        """
        obs_dim=config.OBS_DIM_ACT_MODEL
        super(ActModel, self).__init__()
        self.shot_types = shot_types
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.shot_type_head = nn.Linear(128, shot_types)
        self.landing_pos_head = nn.Linear(128 + shot_types, landing_pos_n)
        self.height_head = nn.Linear(128 + shot_types, height_levels)
        if 'backhand' in config.FEATURES:
           self.backhand_head = nn.Linear(128 + shot_types, 2)
        if 'aroundhead' in config.FEATURES:
            self.aroundhead_head = nn.Linear(128 + shot_types, 2)

    def forward(self, x, shot_type=None):
        features = self.net(x)  # (batch, 128)
        shot_type_logits = self.shot_type_head(features)

        if self.training:
            if shot_type is None:
                raise ValueError("shot_type must be provided during training")
            shot_onehot = F.one_hot(shot_type, num_classes=self.shot_types).float()
        else:
            _, shot_argmax = torch.max(shot_type_logits, dim=1)
            shot_onehot = F.one_hot(shot_argmax, num_classes=self.shot_types).float()

        combined = torch.cat([features, shot_onehot], dim=1)  # (batch, 128 + shot_types)
        landing_logits = self.landing_pos_head(combined)
        height_logits = self.height_head(combined)

        result = [shot_type_logits, landing_logits, height_logits]

        if 'backhand' in config.FEATURES:
            backhand_logits = self.backhand_head(combined)
            result.append(backhand_logits)
        if 'aroundhead' in config.FEATURES:
            aroundhead_logits = self.aroundhead_head(combined)
            result.append(aroundhead_logits)

        return result

    def predict(self, obs, return_logprobs=False):
        obs = torch.FloatTensor(obs).unsqueeze(0)  # (1, obs_dim)
        with torch.no_grad():
            features = self.net(obs)  # (1, 128)

            shot_type_logits = self.shot_type_head(features)
            shot_probs = F.softmax(shot_type_logits, dim=1)
            # print('shot_probs:', shot_probs)
            shot_type = torch.multinomial(shot_probs, num_samples=1)

            shot_tensor = torch.tensor([shot_type], dtype=torch.long)
            shot_onehot = F.one_hot(shot_tensor, num_classes=self.shot_types).float()  # (1, shot_types)

            combined = torch.cat([features, shot_onehot], dim=1)  # (1, 128 + shot_types)

            landing_logits = self.landing_pos_head(combined)
            landing_probs = F.softmax(landing_logits, dim=1)
            landing_pos = torch.multinomial(landing_probs, num_samples=1)
            # print('landing_probs:', landing_probs)

            height_logits = self.height_head(combined)
            height_probs = F.softmax(height_logits, dim=1)
            height = torch.multinomial(height_probs, num_samples=1)

            action = [shot_type.item(), landing_pos.item(), height.item()]

            if 'backhand' in config.FEATURES:
                backhand_logits = self.backhand_head(combined)
                backhand_probs = F.softmax(backhand_logits, dim=1)
                action.append(torch.multinomial(backhand_probs, num_samples=1).item())

            if 'aroundhead' in config.FEATURES:
                aroundhead_logits = self.aroundhead_head(combined)
                aroundhead_probs = F.softmax(aroundhead_logits, dim=1)
                action.append(torch.multinomial(aroundhead_probs, num_samples=1).item())

        if not return_logprobs:
            return action

        shot_dist = Categorical(shot_probs)
        landing_dist = Categorical(landing_probs)
        height_dist = Categorical(height_probs)
        if 'backhand' in config.FEATURES:
            backhand_dist = Categorical(backhand_probs)
        if 'aroundhead' in config.FEATURES:
            aroundhead_dist = Categorical(aroundhead_probs)

        logprobs = [
            shot_dist.log_prob(shot_type).item(),
            landing_dist.log_prob(landing_pos).item(),
            height_dist.log_prob(height).item()
        ]
        if 'backhand' in config.FEATURES:
            logprobs.append(backhand_dist.log_prob(backhand).item())
        if 'aroundhead' in config.FEATURES:
            logprobs.append(aroundhead_dist.log_prob(aroundhead).item())

        return action, logprobs