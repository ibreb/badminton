import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import json
import random
import matplotlib.pyplot as plt


import config
from env import Env
from player import Player, DLPlayer
from models import ResultModel, DefenseModel, ActModel

class CriticModel(nn.Module):
    def __init__(self):
        obs_dim = config.OBS_DIM_ACT_MODEL
        super(CriticModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

class PPOPlayer(DLPlayer):
    def __init__(self, player_id, result_model, defense_model, act_model, lr=1e-4, gamma=0.9, eps_clip=0.1, K_epochs=4, lam=0.95):
        super().__init__(player_id, result_model, defense_model, act_model)
        self.optimizer = optim.Adam(self.act_model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.critic_model = CriticModel()
        self.lam = lam  # GAE Lambda
        self.memory = []

    def serve(self, obs) -> tuple:
        return self._act(obs)

    def generate_shot(self, obs) -> tuple:
        return self._act(obs)

    def _act(self, obs):
        model = self.act_model
        action, logprobs = model.predict(obs, return_logprobs=True)
        self.current_action_logprobs = logprobs
        return action

    def store_transition(self, state, action, logprobs, reward, done):
        self.memory.append((state, action, logprobs, reward, done))

    def clear_memory(self):
        self.memory = []

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values.tolist() + [0]
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i + 1]
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32)

    def update(self):
        if len(self.memory) == 0:
            return

        states, actions, logprobs_old, rewards, dones = zip(*self.memory)
        
        old_states = torch.FloatTensor(np.array(states))
        old_actions = torch.LongTensor(actions)
        old_logprobs = torch.tensor(logprobs_old, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        with torch.no_grad():
            old_values = self.critic_model(old_states)

        advantages = self.compute_gae(rewards, old_values, dones)
        returns = advantages + old_values

        for _ in range(self.K_epochs):
            values = self.critic_model(old_states)
            
            if 'aroundhead' in config.FEATURES:
                shot_logits, landing_logits, height_logits, backhand_logits, aroundhead_logits = self.act_model(old_states, old_actions[:, 0])
            elif 'backhand' in config.FEATURES:
                shot_logits, landing_logits, height_logits, backhand_logits = self.act_model(old_states, old_actions[:, 0])
            else:
                shot_logits, landing_logits, height_logits = self.act_model(old_states, old_actions[:, 0])

            shot_probs = F.softmax(shot_logits, dim=1)
            landing_probs = F.softmax(landing_logits, dim=1)
            height_probs = F.softmax(height_logits, dim=1)


            # print('!', old_states, shot_logits, shot_probs)
            shot_dist = Categorical(shot_probs)
            landing_dist = Categorical(landing_probs)
            height_dist = Categorical(height_probs)

            logprobs_shot = shot_dist.log_prob(old_actions[:, 0])
            logprobs_landing = landing_dist.log_prob(old_actions[:, 1])
            logprobs_height = height_dist.log_prob(old_actions[:, 2])

            logprobs = logprobs_shot + logprobs_landing + logprobs_height

            if 'backhand' in config.FEATURES:
                backhand_probs = F.softmax(backhand_logits, dim=1)
                backhand_dist = Categorical(backhand_probs)
                logprobs_backhand = backhand_dist.log_prob(old_actions[:, 3])
                logprobs = logprobs + logprobs_backhand

            if 'aroundhead' in config.FEATURES:
                aroundhead_probs = F.softmax(aroundhead_logits, dim=1)
                aroundhead_dist = Categorical(aroundhead_probs)
                logprobs_aroundhead = aroundhead_dist.log_prob(old_actions[:, 4])
                logprobs = logprobs + logprobs_aroundhead

            ratios = torch.exp(logprobs - old_logprobs.sum(dim=1))

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # print(f'({advantages}  {actor_loss})')

            critic_loss = F.mse_loss(values, returns)

            loss = actor_loss + 0.5 * critic_loss

            print('loss:', loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.clear_memory()


def train_ppo_player(env: Env, ppo_player: PPOPlayer, opponent: Player, episodes=1000):
    rewards = []

    for ep in range(episodes):
        serve_player = random.randint(0, 1)
        state = env.reset(serve_player=serve_player)
        done = False
        ppo_player.clear_memory()

        total_reward = 0

        obs = env._get_act_model_obs()
        if env.current_player == 0:
            action = ppo_player.serve(obs)
            logprobs = ppo_player.current_action_logprobs
        else:
            action = opponent.serve(obs)
            logprobs = None

        while not done:
            next_state, reward, done, info = env.step(action)

            if env.current_player == 1:
                ppo_player.store_transition(obs, action, logprobs, reward, done)
                total_reward += reward

            if done:
                break

            obs = env._get_act_model_obs()
            if env.current_player == 0:
                action = ppo_player.generate_shot(obs)
                logprobs = ppo_player.current_action_logprobs
            else:
                action = opponent.generate_shot(obs)
                logprobs = None

        ppo_player.update()
        rewards.append(total_reward)

        if ep % 100 == 0:
            print(f"Episode {ep}, Average Reward: {np.mean(rewards[-100:])}")

    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    plt.plot(smooth(rewards, 100))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('PPO Training Reward')
    plt.show()


if __name__ == '__main__':

    with open('badminton_configs.json', 'r', encoding='utf-8') as f:
        configs = json.load(f)

    config.load_config(configs['w3_f3'])

    result_model = ResultModel()
    defense_model = DefenseModel()
    act_model = ActModel()

    result_model.load_state_dict(torch.load('result_model_w3_f3_李宗伟.pth'))
    defense_model.load_state_dict(torch.load('defense_model_w3_f3_李宗伟.pth'))
    act_model.load_state_dict(torch.load('act_model_w3_f3_李宗伟.pth'))

    player0 = PPOPlayer(0, result_model, defense_model, act_model)

    result_model = ResultModel()
    defense_model = DefenseModel()
    act_model = ActModel()

    result_model.load_state_dict(torch.load('result_model_w3_f3_林丹.pth'))
    defense_model.load_state_dict(torch.load('defense_model_w3_f3_林丹.pth'))
    act_model.load_state_dict(torch.load('act_model_w3_f3_林丹.pth'))

    player1 = DLPlayer(1, result_model, defense_model, act_model)

    env = Env(player0, player1)

    train_ppo_player(env, player0, player1, episodes=2000)

    torch.save(player0.act_model.state_dict(), f'act_model_李宗伟_ppo_1.pth')