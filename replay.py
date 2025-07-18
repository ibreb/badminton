import csv

from env import Env
from player import Player
import random

class RecordPlayer(Player):
    def __init__(self, file_path):
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
            assert self.end_reason in ['下网', '击球落地', '出界']

            for i, row in enumerate(rows[:-1]):
                if not row.get('player') or not row['player'].strip():
                    continue
                
                ty = ACTIONS.index(row['type'])
                
                landing_area = int(rows[i + 1]['landing_area'])
                
                height = int(float(row['ball_height']))

                if landing_area == -1:
                    landing_area = random.randint(7, 9)
                
                actions.append((ty, landing_area, height))

            winner = rows[-1].get('getpoint_player', '').strip()

        self.idx = 0
        self.n = len(actions)
        self.actions = actions
        self.winner = winner


    def result(self, obs):
        if self.idx == self.n and self.end_reason in ['出界', '下网']:
            return self.end_reason
        return '成功'

    def hit(self, obs):
        if self.idx == self.n:
            return False
        return True

    def serve(self, obs):
        action = self.actions[self.idx]
        self.idx += 1
        return action

    def generate_shot(self, obs):
        action = self.actions[self.idx]
        self.idx += 1
        return action

if __name__ == '__main__':

    replayPlayer = RecordPlayer('foo.csv')

    env = Env(replayPlayer, replayPlayer)
    state = env.reset(0)
    env.run_episode()
    env.render()