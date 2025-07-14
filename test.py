from env import Env
from player import DLPlayer
from models import ResultModel, DefenseModel, ActModel

def main():
    result_model = ResultModel()
    defense_model = DefenseModel()
    act_model = ActModel()
    player0 = DLPlayer(0, result_model, defense_model, act_model)
    player1 = DLPlayer(1, result_model, defense_model, act_model)
    env = Env(player0, player1)
    env.run_match()
    env.render()

if __name__ == "__main__":
    main()