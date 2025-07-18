import numpy as np
from player import Player
from visualize import Visualizer

class Env:

    def __init__(self, player0: Player, player1: Player, winning_score=10):
        self.players = [player0, player1]
        self.current_player = 0
        self.ACTIONS = ['发球', '扣杀', '高远球', '网前球', '吊球', '平抽球', '挑球', '扑球', '挡网', '切球']

        self.score = [0, 0]
        self.winning_score = winning_score  # 比赛结束的分数

        self.history = []

        self.scored_player = 0

    def reset(self, serve_player=0):
        self.current_player = serve_player

        player0_pos = 5
        player1_pos = 5
        ball_pos = 5
        ball_height = 0

        self.state = [player0_pos, player1_pos, ball_pos, ball_height, serve_player]

        return self.state

    def step(self, action):
        _, landing_pos, hit_height = action
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

            # if result == '下网':
            #     landing_pos = (2.5, 1)
            # if result == '出界':
            #     landing_pos = (5, 3) if player_id == 0 else (0, -1)

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
            player_positions = [5, landing_pos]
        else:
            player_positions = [landing_pos, 5]

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

        self.state = next_state
        self.current_player = opponent_id

        return next_state, reward, done, info

    def _get_result_model_obs(self, action):
        obs = self.state + list(action)
        return obs
    
    def _get_hit_model_obs(self, action):
        obs = self.state + list(action)
        return obs

    def _get_act_model_obs(self):
        return self.state

    def render(self):
        """
        渲染击打过程
        """
        Visualizer.animate(self.history, self.ACTIONS)


    def run_episode(self, serve_player=0):
        """
        运行完整一分
        """
        state = self.reset(serve_player)
        action = self.players[self.current_player].serve(state)
        done = False
        while not done:
            obs, reward, done, _ = self.step(action)
            action = self.players[self.current_player].generate_shot(obs)

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
        