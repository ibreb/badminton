from env import Env
from player import SamplePlayer, DLPlayer, GreedyPlayer
from models import ResultModel, DefenseModel, ActModel

def main():
    result_model = ResultModel()
    defense_model = DefenseModel()
    act_model = ActModel()

    player0 = DLPlayer(0, result_model, defense_model, act_model)

    result_model = ResultModel()
    defense_model = DefenseModel()
    act_model = ActModel()

    player1 = DLPlayer(1, result_model, defense_model, act_model)

    env = Env(player0, player1)
    env.run_match()
    env.render()


    # player0 = DLPlayer(0, result_model, defense_model, act_model)
    # player1 = GreedyPlayer(1, result_model, defense_model, act_model)
    # env = Env(player0, player1)
    # env.run_n_matches(100)

    # player0 = GreedyPlayer(0, result_model, defense_model, act_model)
    # player1 = DLPlayer(1, result_model, defense_model, act_model)
    # env = Env(player0, player1)
    # env.run_n_matches(100)
    

if __name__ == '__main__':
    main()
