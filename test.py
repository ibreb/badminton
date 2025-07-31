import torch
import json

import config
from env import Env
from player import SamplePlayer, DLPlayer
from models import ResultModel, DefenseModel, ActModel

def main():
    with open('badminton_configs.json', 'r', encoding='utf-8') as f:
        configs = json.load(f)

    config.load_config(configs['w3_f3'])
    result_model = ResultModel()
    defense_model = DefenseModel()
    act_model = ActModel()

    result_model.load_state_dict(torch.load('result_model_w3_f3_林丹.pth'))
    defense_model.load_state_dict(torch.load('defense_model_w3_f3_林丹.pth'))
    act_model.load_state_dict(torch.load('act_model_w3_f3_林丹.pth'))

    player0 = DLPlayer(0, result_model, defense_model, act_model)

    result_model = ResultModel()
    defense_model = DefenseModel()
    act_model = ActModel()

    result_model.load_state_dict(torch.load('result_model_w3_f3_李宗伟.pth'))
    defense_model.load_state_dict(torch.load('defense_model_w3_f3_李宗伟.pth'))
    act_model.load_state_dict(torch.load('act_model_李宗伟_ppo.pth'))

    player1 = DLPlayer(1, result_model, defense_model, act_model)

    env = Env(player0, player1)
    env.run_match()
    env.save_to_csv()
    env.render()

    # env = Env(player0, player1)
    # env.run_n_matches(500)

    # env = Env(player1, player0)
    # env.run_n_matches(500)

if __name__ == '__main__':
    main()