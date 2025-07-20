import csv
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from env import Env
from player import Player
from models import ResultModel, DefenseModel, ActModel

class RecordPlayer(Player):
    def __init__(self):
        self.obs_list = []
        self.labels = []

        self.hit_obs = []
        self.hit_res = []

        self.act_obs = []
        self.shot_types = []
        self.landing_positions = []
        self.heights = []

    def load(self, file_path):
        ACTIONS = ['发球', '扣杀', '高远球', '网前吊球', '吊球', '平抽球', '挑球', '扑球', '挡网', '切球']
        actions = []
        winner = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.end_reason = rows[-1].get('win_reason') + rows[-1].get('lose_reason')

            if self.end_reason == '擦网落地':
                self.end_reason = '下网'
            if self.end_reason == '落地':
                self.end_reason = '击球落地'
            if self.end_reason == '其他原因':
                print('对局结果：其他原因')
                self.end_reason = '击球落地'
            if self.end_reason not in ['下网', '击球落地', '出界']:
                return False

            for i, row in enumerate(rows[:-1]):
                if not row.get('player') or not row['player'].strip():
                    continue
                
                ty = ACTIONS.index(row['type'])
                
                landing_area = int(rows[i + 1]['landing_area'])
                
                height = int(float(row['ball_height']))

                if height not in [1, 2, 3]:
                    return False

                if landing_area == -1:
                    landing_area = random.randint(7, 9)
                
                actions.append((ty, landing_area, height))

            winner = rows[-1].get('getpoint_player', '').strip()

        self.idx = 0
        self.n = len(actions)
        self.actions = actions
        self.winner = winner
        return True

    def result(self, obs):
        if self.idx == self.n and self.end_reason in ['出界', '下网']:
            res = self.end_reason
        else:
            res = '成功'

        self.obs_list.append(obs)
        self.labels.append(['成功', '出界', '下网'].index(res))
        return res

    def hit(self, obs):
        self.hit_obs.append(obs)
        self.hit_res.append(int(self.idx < self.n))
        return self.idx < self.n

    def serve(self, obs):
        action = self.actions[self.idx]
        self.debug[action[1] - 1] += 1
        self.act_obs.append(obs)
        self.shot_types.append(action[0])
        self.landing_positions.append(action[1] - 1)
        self.heights.append(action[2] - 1)
        self.idx += 1
        return action

    def generate_shot(self, obs):
        action = self.actions[self.idx]
        if action[2] == 0:
            action = (action[0], action[1], 1)
        # print(action)
        if action[1] > 0:
            self.act_obs.append(obs)
            self.shot_types.append(action[0])
            self.landing_positions.append(action[1] - 1)
            self.heights.append(action[2] - 1)
        self.idx += 1
        return action

if __name__ == "__main__":

    path = Path('羽毛球关键帧')
    replayPlayer = RecordPlayer()
    for csv_file in path.rglob('*.csv'):
        # print(csv_file)
        if not replayPlayer.load(csv_file):
            print('skipping ', csv_file)
            continue
        env = Env(replayPlayer, replayPlayer)
        state = env.reset(0)
        env.run_episode()

    model = ResultModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    obs_tensor = torch.tensor(replayPlayer.obs_list, dtype=torch.float32)
    labels_tensor = torch.tensor(replayPlayer.labels, dtype=torch.long)

    dataset = TensorDataset(obs_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 10

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'result_model.pth')




    model = DefenseModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    obs_tensor = torch.tensor(replayPlayer.hit_obs, dtype=torch.float32)
    labels_tensor = torch.tensor(replayPlayer.hit_res, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(obs_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 10

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'defense_model.pth')

    obs_tensor = torch.tensor(replayPlayer.act_obs, dtype=torch.float32)
    shot_tensor = torch.tensor(replayPlayer.shot_types, dtype=torch.long)
    land_tensor = torch.tensor(replayPlayer.landing_positions, dtype=torch.long)
    height_tensor = torch.tensor(replayPlayer.heights, dtype=torch.long)

    act_dataset = TensorDataset(obs_tensor, shot_tensor, land_tensor, height_tensor)
    act_loader = DataLoader(act_dataset, batch_size=32, shuffle=True)

    act_model = ActModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(act_model.parameters(), lr=0.001)

    for epoch in range(10):
        for obs, shot, land, height in act_loader:
            shot_logits, land_logits, height_logits = act_model(obs, shot)
            loss = criterion(shot_logits, shot) + criterion(land_logits, land) + criterion(height_logits, height)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    torch.save(act_model.state_dict(), 'act_model.pth')

    print(replayPlayer.debug)