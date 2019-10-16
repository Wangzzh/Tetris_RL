from tetris.tetris import Tetris

T = Tetris(render=True)

T.start()
T.print_state()
while True:
    action = int(input("Action: "))
    if action < 0:
        T.start()
    else:
        T.step(action)
    T.print_state()
