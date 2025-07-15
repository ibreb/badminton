import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResultModel(nn.Module):
    def __init__(self, obs_dim=12, num_classes=3):
        """
        结果模型：判断当前动作是否成功（成功/出界/下网）
        
        参数：
        - obs_dim: 观测向量的维度
        - num_classes: 输出类别数量
        """
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
        return ['成功', '出界', '下网'][idx]


class DefenseModel(nn.Module):
    def __init__(self, obs_dim=12):
        """
        防守模型：判断是否能接到对手的击球
        
        参数：
        - obs_dim: 观测向量的维度
        """
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
        return random.random() < prob


class ActModel(nn.Module):
    def __init__(self, obs_dim=8, shot_types=10, landing_pos_n=9, height_levels=3):
        """
        决策模型：根据观测和对手动作生成击球动作
        
        参数：
        - obs_dim: 观测向量的维度
        - shot_types: 击球类型数量
        - height_levels: 击球高度级数
        """
        super(ActModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU()
        )
        self.shot_type_head = nn.Linear(128, shot_types)
        self.landing_pos_head = nn.Linear(128, landing_pos_n)
        self.height_head = nn.Linear(128, height_levels)

    def forward(self, x):
        features = self.net(x)
        shot_type_logits = self.shot_type_head(features)
        landing_pos_logits = self.landing_pos_head(features)
        height_logits = self.height_head(features)
        return shot_type_logits, landing_pos_logits, height_logits

    def predict(self, obs):
        """
        输入：当前观测 obs
        输出：(shot_type, landing_pos, height)
        """
        obs = torch.FloatTensor(obs).unsqueeze(0)  # (1, input_dim)

        with torch.no_grad():
            shot_type_logits, landing_pos_logits, height_logits = self(obs)

            shot_type_probs = F.softmax(shot_type_logits, dim=1)
            shot_type = torch.multinomial(shot_type_probs, num_samples=1).item()

            landing_pos_probs = F.softmax(landing_pos_logits, dim=1)
            landing_pos_idx = torch.multinomial(landing_pos_probs, num_samples=1).item()

            height_probs = F.softmax(height_logits, dim=1)
            height = torch.multinomial(height_probs, num_samples=1).item()


        return (shot_type, landing_pos_idx, height)