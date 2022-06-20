import pyray as rl
from importlib import reload
import traceback as tb
import copy

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "game")
rl.set_target_fps(60)

import game  # noqa
state = game.state
error = None

while not rl.window_should_close():
    rl.begin_drawing()
    rl.clear_background(rl.BLACK)

    if rl.is_key_released(rl.KEY_R):
        print('reload, with state reset')
        error = None
        reload(game)
        state = game.state

    if rl.is_key_released(rl.KEY_F5):
        print('reload')
        error = None
        reload(game)

    if error is None:
        try:
            game.step_game(state)
        except Exception as e:
            error = e
    else:
        rl.draw_text(''.join(tb.format_exception(None, error, error.__traceback__)), 10, 40, 10, rl.RED)

    rl.draw_fps(10, 10)
    rl.end_drawing()

rl.close_window()
