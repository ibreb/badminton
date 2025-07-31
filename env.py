import numpy as np
from collections import deque
import csv

import config
from player import Player
from visualize import Visualizer

class Env:

    def __init__(self, player0: Player, player1: Player, winning_score=21):
        self.players = [player0, player1]
        self.current_player = 0

        self.score = [0, 0]
        self.winning_score = winning_score  # 比赛结束的分数

        self.history = []
        self.log = []

        self.scored_player = 0

    def reset(self, serve_player=0):
        self.current_player = serve_player

        self.center = 4

        player0_pos = self.center
        player1_pos = self.center
        ball_pos = self.center
        ball_height = 0

        self.state = [player0_pos, player1_pos, ball_pos, ball_height, serve_player]

        self.window_size = config.WINDOW_SIZE
        self.window = deque(maxlen=self.window_size)
        for i in range(self.window_size):
            action = [10, self.center, 2]
            while len(action) < len(config.FEATURES):
                action.append(0)
            self.window.append(action)

        return self.state

    def step(self, action):
        ty = config.ACTIONS[action[0]]
        landing_pos = action[1]
        hit_height = action[2]

        self.window.append(action)

        assert 0 <= hit_height <= 2
        assert -1 <= landing_pos <= 8
        player_id = self.current_player

        opponent_id = 1 - player_id

        reward = 0
        done = False

        info = {
            'failure_reason': '成功',
            'losing_player': -1,
            'action': action,
            'current_player': player_id
        }

        obs = self._get_result_model_obs(action)
        result = self.players[player_id].result(obs)
        if result != '成功':  # 出界或下网
            reward = -1
            done = True

            self.scored_player = opponent_id
            self.score[opponent_id] += 1
            info['failure_reason'] = result
            info['losing_player'] = player_id

        else:
            obs = self._get_hit_model_obs(action)
            hit = self.players[opponent_id].hit(obs)
            if not hit:
                reward = 1
                done = True

                self.scored_player = player_id
                self.score[player_id] += 1
                info['failure_reason'] = '击球落地'
                result = '击球落地'
                info['losing_player'] = opponent_id

        if player_id == 0:
            player_positions = [self.center, landing_pos]
        else:
            player_positions = [landing_pos, self.center]

        ball_pos = landing_pos
        ball_height = hit_height

        next_state = [
            *player_positions, ball_pos, ball_height, opponent_id
        ]

        self.history.append({
            'state': self.state,
            'action': action,
            'current_player': player_id,
            'next_state': next_state,
            'reward': reward,
            'score_player0': self.score[0],
            'score_player1': self.score[1],
            'failure_reason': result,
            'losing_player': info['losing_player']
        })

        self.log.append({
            'Score_A' : self.score[0],
            'score_B' : self.score[1],
            'player': self.current_player,
            'type': ty,
            'ball_height': ball_height,
            'landing_area': landing_pos,
            'result': result,
            'losing_player': info['losing_player']
        })

        self.state = next_state
        self.current_player = opponent_id

        return next_state, reward, done, info

    def _one_hot(self, x, n):
        V = []
        for i in range(n):
            V.append(i == x)
        return V

    def _get_state(self):
        state = self.state.copy()
        if self.current_player == 1:
            state[0], state[1] = state[1], state[0]
        return state[:4]

    def _get_result_model_obs(self, action):
        obs = []
        for i in range(1, self.window_size + 1):
            for j, feature in enumerate(config.FEATURES):
                if i == 1 and feature == 'landing_area':
                    continue
                obs += self._one_hot(self.window[-i][j], config.FEATURE_SIZES[j])
        return obs
    
    def _get_hit_model_obs(self, action):
        obs = []
        for i in range(1, self.window_size + 1):
            for j, feature in enumerate(config.FEATURES):
                obs += self._one_hot(self.window[-i][j], config.FEATURE_SIZES[j])
        return obs

    def _get_act_model_obs(self):
        obs = []
        for i in range(1, self.window_size + 1):
            for j, feature in enumerate(config.FEATURES):
                obs += self._one_hot(self.window[-i][j], config.FEATURE_SIZES[j])
        return obs

    def render(self):
        """
        渲染击打过程
        """
        print(self.history)
        Visualizer.animate(self.history)

    def run_episode(self, serve_player=0):
        """
        运行完整一分
        """
        state = self.reset(serve_player)
        action = self.players[self.current_player].serve(self._get_act_model_obs())
        done = False
        while not done:
            obs, reward, done, _ = self.step(action)
            if done:
                break
            action = self.players[self.current_player].generate_shot(self._get_act_model_obs())

    def _check_game_over(self):
        """
        检查是否有一方达到胜利条件
        """
        return (self.score[0] >= self.winning_score or self.score[1] >= self.winning_score) and \
           abs(self.score[0] - self.score[1]) >= 2

    def run_match(self):
        """
        运行完整一个对局
        """
        self.score = [0, 0]
        self.scored_player = 0
        while not self._check_game_over():
            self.run_episode(self.scored_player)

    def run_n_matches(self, n):
        scores = [0, 0]
        for i in range(n):
            self.run_match()
            winner = 0 if self.score[0] > self.score[1] else 1
            scores[winner] += 1
        print(scores)
    
    def save_to_csv(self):
        with open('game_history.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.log[0].keys())
            writer.writeheader()
            writer.writerows(self.log)