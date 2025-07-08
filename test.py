from env import Env
from player import SamplePlayer

def main():
    player0 = SamplePlayer(0)
    player1 = SamplePlayer(1)
    env = Env(player0, player1)
    env.run_match()
    env.render()

if __name__ == "__main__":
    main()