import csv
import random
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split

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
        self.players = []
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
                # print('对局结果：其他原因')
                self.end_reason = '击球落地'
            if self.end_reason not in ['下网', '击球落地', '出界']:
                return False

            for i, row in enumerate(rows[:-1]):
                if not row.get('player') or not row['player'].strip():
                    continue
                
                ty = ACTIONS.index(row['type'])
                
                landing_area = int(rows[i]['player_location_area'])
                
                height = int(float(row['ball_height']))

                if height not in [1, 2, 3]:
                    return False

                if landing_area == -1:
                    landing_area = random.randint(7, 9)

                player = str(row['player'])
                self.players.append(player)
                
                actions.append((ty, landing_area, height))
                # if row['type'] == '扣杀':
                #     print(height)

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

        if self.players[self.idx - 1] == target_player:
            self.obs_list.append(obs)
            self.labels.append(['成功', '出界', '下网'].index(res))
        return res

    def hit(self, obs):
        if self.players[self.idx - 1] != target_player:
            self.hit_obs.append(obs)
            self.hit_res.append(int(self.idx < self.n))
        return self.idx < self.n

    def serve(self, obs):
        action = self.actions[self.idx]
        if self.players[self.idx] == target_player:
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
            if self.players[self.idx] == target_player:
                self.act_obs.append(obs)
                self.shot_types.append(action[0])
                self.landing_positions.append(action[1] - 1)
                self.heights.append(action[2] - 1)
        self.idx += 1
        return action

# 损失曲线绘制函数
def plot_losses(train_losses, test_losses, title, y_max=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(title)
    if y_max is not None:
        plt.ylim(top=y_max)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)


    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()


if __name__ == "__main__":
    target_player = '李宗伟'

    path = Path('羽毛球关键帧')
    replayPlayer = RecordPlayer()
    for csv_file in path.rglob('*.csv'):
        if not replayPlayer.load(csv_file):
            print('skipping ', csv_file)
            continue
        env = Env(replayPlayer, replayPlayer)
        state = env.reset(0)
        env.run_episode()


    # ResultModel训练
    model = ResultModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    
    obs_tensor = torch.tensor(replayPlayer.obs_list, dtype=torch.float32)
    labels_tensor = torch.tensor(replayPlayer.labels, dtype=torch.long)
    
    dataset = TensorDataset(obs_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    num_epochs = 50
    train_losses, test_losses = [], []
    
    for epoch in range(num_epochs + 1):
        # 训练阶段

        if epoch > 0:
            model.train()
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

        train_loss = 0.0
        with torch.no_grad():
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                train_loss += loss.item() * inputs.size(0)
        
        # 记录损失
        train_losses.append(train_loss / len(train_loader.dataset))
        test_losses.append(test_loss / len(test_loader.dataset))
        
        print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
    
    # 保存模型和绘图
    torch.save(model.state_dict(), f'result_model_{target_player}.pth')
    plot_losses(train_losses, test_losses, 'Result_Model_Loss', y_max=0.4)

    # DefenseModel训练
    model = DefenseModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    
    obs_tensor = torch.tensor(replayPlayer.hit_obs, dtype=torch.float32)
    labels_tensor = torch.tensor(replayPlayer.hit_res, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(obs_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    num_epochs = 50
    train_losses, test_losses = [], []
    
    for epoch in range(num_epochs + 1):
        # 训练阶段
        if epoch > 0:
            model.train()
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # print(outputs)
                # print(targets)
                # quit()
                test_loss += loss.item() * inputs.size(0)

        train_loss = 0.0
        with torch.no_grad():
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                train_loss += loss.item() * inputs.size(0)
        
        # 记录损失
        train_losses.append(train_loss / len(train_loader.dataset))
        test_losses.append(test_loss / len(test_loader.dataset))
        
        print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
    
    # 保存模型和绘图
    torch.save(model.state_dict(), f'defense_model_{target_player}.pth')
    plot_losses(train_losses, test_losses, 'Defense_Model_Loss', y_max=0.3)

    # ActModel训练
    model = ActModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    
    # print(replayPlayer.act_obs)
    obs_tensor = torch.tensor(replayPlayer.act_obs, dtype=torch.float32)
    shot_tensor = torch.tensor(replayPlayer.shot_types, dtype=torch.long)
    land_tensor = torch.tensor(replayPlayer.landing_positions, dtype=torch.long)
    height_tensor = torch.tensor(replayPlayer.heights, dtype=torch.long)
    
    dataset = TensorDataset(obs_tensor, shot_tensor, land_tensor, height_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    num_epochs = 50
    train_losses, test_losses = [], []
    
    for epoch in range(num_epochs + 1):
        # 训练阶段
        if epoch > 0:
            model.train()
            epoch_loss = 0.0
            for obs, shot, land, height in train_loader:
                shot_logits, land_logits, height_logits = model(obs, shot)
                loss = criterion(shot_logits, shot) + criterion(land_logits, land) + criterion(height_logits, height)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * obs.size(0)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for obs, shot, land, height in test_loader:
                shot_logits, land_logits, height_logits = model(obs, shot)
                loss = criterion(shot_logits, shot) + criterion(land_logits, land) + criterion(height_logits, height)
                test_loss += loss.item() * obs.size(0)

        train_loss = 0.0
        with torch.no_grad():
            for obs, shot, land, height in train_loader:
                shot_logits, land_logits, height_logits = model(obs, shot)
                loss = criterion(shot_logits, shot) + criterion(land_logits, land) + criterion(height_logits, height)
                train_loss += loss.item() * obs.size(0)
        
        # 记录损失
        train_losses.append(train_loss / len(train_loader.dataset))
        test_losses.append(test_loss / len(test_loader.dataset))
        
        print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
    
    # 测试评估
    model.eval()
    correct_top1 = 0
    correct_top2 = 0
    correct_top3 = 0
    correct_top4 = 0
    total = 0
    with torch.no_grad():
        for obs, shot, land, height in test_loader:
            shot_logits, land_logits, _ = model(obs, shot)

            topk_shot = torch.topk(land_logits, k=4, dim=1).indices

            # Top-1 Accuracy
            top1_preds = topk_shot[:, 0]
            correct_top1 += (land == top1_preds).sum().item()

            # Top-2 Accuracy
            top2_preds = topk_shot[:, :2]
            correct_top2 += (land.unsqueeze(1) == top2_preds).any(dim=1).sum().item()

            # Top-3 Accuracy
            top3_preds = topk_shot[:, :3]
            correct_top3 += (land.unsqueeze(1) == top3_preds).any(dim=1).sum().item()

            # Top-4 Accuracy
            correct_top4 += (land.unsqueeze(1) == topk_shot).any(dim=1).sum().item()
            total += obs.size(0)
    
    print(f'ActModel Test Accuracy: Top1={correct_top1 / total:.4f}, '
      f'Top2={correct_top2 / total:.4f}, '
      f'Top3={correct_top3 / total:.4f},'
      f'Top4={correct_top4 / total:.4f}'
      )
    
    # 保存模型和绘图
    torch.save(model.state_dict(), f'act_model_{target_player}.pth')
    plot_losses(train_losses, test_losses, 'Act_Model_Loss')