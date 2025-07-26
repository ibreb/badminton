import torch

from env import Env
from player import SamplePlayer, DLPlayer, GreedyPlayer
from models import ResultModel, DefenseModel, ActModel

def main():
    result_model = ResultModel()
    defense_model = DefenseModel()
    act_model = ActModel()

    result_model.load_state_dict(torch.load('result_model_林丹.pth'))
    defense_model.load_state_dict(torch.load('defense_model_林丹.pth'))
    act_model.load_state_dict(torch.load('act_model_林丹_ppo.pth'))

    player0 = DLPlayer(0, result_model, defense_model, act_model)

    result_model = ResultModel()
    defense_model = DefenseModel()
    act_model = ActModel()

    result_model.load_state_dict(torch.load('result_model_李宗伟.pth'))
    defense_model.load_state_dict(torch.load('defense_model_李宗伟.pth'))
    act_model.load_state_dict(torch.load('act_model_李宗伟.pth'))

    player1 = DLPlayer(1, result_model, defense_model, act_model)

    env = Env(player0, player1)
    env.run_match()
    env.render()


if __name__ == '__main__':
    main()