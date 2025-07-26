import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from utils import *
from env import Env
from player import Player, DLPlayer
from models import ResultModel, DefenseModel, ActModel

class PPOPlayer(DLPlayer):
    def __init__(self, player_id, result_model, defense_model, act_model, lr=1e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        super().__init__(player_id, result_model, defense_model, act_model)
        self.optimizer = optim.Adam(self.act_model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.memory = []

    def serve(self, obs) -> tuple:
        return self._act(obs)

    def generate_shot(self, obs) -> tuple:
        return self._act(obs)

    def _act(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0)  # (1, obs_dim)
        model = self.act_model
        with torch.no_grad():
            features = model.net(obs)  # (1, 128)

            shot_type_logits = model.shot_type_head(features)
            shot_probs = F.softmax(shot_type_logits, dim=1)
            # print('shot_probs:', shot_probs)
            shot_type = torch.multinomial(shot_probs, num_samples=1)

            shot_tensor = torch.tensor([shot_type], dtype=torch.long)
            shot_onehot = F.one_hot(shot_tensor, num_classes=model.shot_types).float()  # (1, shot_types)

            combined = torch.cat([features, shot_onehot], dim=1)  # (1, 128 + shot_types)

            landing_logits = model.landing_pos_head(combined)
            landing_probs = F.softmax(landing_logits, dim=1)
            landing_pos = torch.multinomial(landing_probs, num_samples=1)

            height_logits = model.height_head(combined)
            height_probs = F.softmax(height_logits, dim=1)
            height = torch.multinomial(height_probs, num_samples=1)

            backhand_logits = model.backhand_head(combined)
            backhand_probs = F.softmax(backhand_logits, dim=1)
            backhand = torch.multinomial(backhand_probs, num_samples=1)

            aroundhead_logits = model.aroundhead_head(combined)
            aroundhead_probs = F.softmax(aroundhead_logits, dim=1)
            aroundhead = torch.multinomial(aroundhead_probs, num_samples=1)

            shot_dist = Categorical(shot_probs)
            landing_dist = Categorical(landing_probs)
            height_dist = Categorical(height_probs)
            backhand_dist = Categorical(backhand_probs)
            aroundhead_dist = Categorical(aroundhead_probs)

            self.current_action_logprobs = (
                shot_dist.log_prob(shot_type).item(),
                landing_dist.log_prob(landing_pos).item(),
                height_dist.log_prob(height).item(),
                backhand_dist.log_prob(backhand).item(),
                aroundhead_dist.log_prob(aroundhead).item()
            )
            
        return (shot_type.item(), landing_pos.item() + 1, height.item() + 1, backhand.item(), aroundhead.item())

    def store_transition(self, state, action, logprobs, reward, done):
        self.memory.append((state, action, logprobs, reward, done))

    def clear_memory(self):
        self.memory = []

    def update(self):
        if len(self.memory) == 0:
            return

        states, actions, logprobs_old, rewards, dones = zip(*self.memory)

        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        old_states = torch.FloatTensor(np.array(states))
        old_actions = torch.LongTensor(actions)
        old_logprobs = torch.tensor(logprobs_old, dtype=torch.float32)

        for _ in range(self.K_epochs):
            shot_logits, landing_logits, height_logits, backhand_logits, aroundhead_logits = self.act_model(old_states)

            shot_probs = F.softmax(shot_logits, dim=1)
            landing_probs = F.softmax(landing_logits, dim=1)
            height_probs = F.softmax(height_logits, dim=1)
            backhand_probs = F.softmax(backhand_logits, dim=1)
            aroundhead_probs = F.softmax(aroundhead_logits, dim=1)

            shot_dist = Categorical(shot_probs)
            landing_dist = Categorical(landing_probs)
            height_dist = Categorical(height_probs)
            backhand_dist = Categorical(backhand_probs)
            aroundhead_dist = Categorical(aroundhead_probs)

            logprobs_shot = shot_dist.log_prob(old_actions[:, 0])
            logprobs_landing = landing_dist.log_prob(old_actions[:, 1] - 1)
            logprobs_height = height_dist.log_prob(old_actions[:, 2] - 1)
            logprobs_backhand = backhand_dist.log_prob(old_actions[:, 3])
            logprobs_aroundhead = aroundhead_dist.log_prob(old_actions[:, 4])

            logprobs = logprobs_shot + logprobs_landing + logprobs_height + logprobs_backhand + logprobs_aroundhead
            ratios = torch.exp(logprobs - old_logprobs.sum(dim=1))

            advantages = returns

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.clear_memory()


def train_ppo_player(env: Env, ppo_player: PPOPlayer, opponent: Player, episodes=1000):
    for ep in range(episodes):
        state = env.reset(serve_player=0)
        done = False
        ppo_player.clear_memory()

        obs = env._get_act_model_obs()
        action = ppo_player.serve(obs)
        logprobs = ppo_player.current_action_logprobs

        while not done:
            next_state, reward, done, info = env.step(action)
            ppo_player.store_transition(obs, action, logprobs, reward, done)

            if done:
                break

            obs = env._get_act_model_obs()
            action = ppo_player.generate_shot(obs)
            logprobs = ppo_player.current_action_logprobs

        ppo_player.update()

        if ep % 100 == 0:
            print(f"Episode {ep}, Score: {env.score}")


if __name__ == '__main__':
    act_model = ActModel()

    result_model = ResultModel()
    defense_model = DefenseModel()
    act_model = ActModel()

    result_model.load_state_dict(torch.load('result_model_林丹.pth'))
    defense_model.load_state_dict(torch.load('defense_model_林丹.pth'))
    act_model.load_state_dict(torch.load('act_model_林丹.pth'))

    ppo_player = PPOPlayer(0, result_model, defense_model, act_model)

    result_model = ResultModel()
    defense_model = DefenseModel()
    act_model = ActModel()

    result_model.load_state_dict(torch.load('result_model_李宗伟.pth'))
    defense_model.load_state_dict(torch.load('defense_model_李宗伟.pth'))
    act_model.load_state_dict(torch.load('act_model_李宗伟.pth'))

    fixed_opponent = DLPlayer(1, result_model, defense_model, act_model)

    env = Env(ppo_player, fixed_opponent)

    train_ppo_player(env, ppo_player, fixed_opponent, episodes=2000)

    torch.save(act_model.state_dict(), f'act_model_林丹_ppo.pth')