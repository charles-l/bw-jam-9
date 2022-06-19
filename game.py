import pyray as rl
import heapq
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Tuple, List, Optional, Dict, TypeVar, NewType, Union
from types import SimpleNamespace as Namespace
import math


T = TypeVar("T")


def unwrap_optional(x: Optional[T]) -> T:
    assert x is not None
    return x


@dataclass(slots=True)
class V2:
    x: float
    y: float

    def set(self, other: 'V2'):
        self.x = other.x
        self.y = other.y

    def __eq__(self, other):
        return self.x == other[0] and self.y == other[1]

    def __lt__(self, other):
        return (self[0], self[1]) < (other[0], other[1])

    def __getitem__(self, i: int):
        return (self.x, self.y)[i]

    def __add__(self, other: "VecType"):
        return V2(self.x + other[0], self.y + other[1])

    def __sub__(self, other: "VecType"):
        return V2(self.x - other[0], self.y - other[1])

    def __mul__(self, scalar: float):
        return V2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float):
        return V2(self.x / scalar, self.y / scalar)

    def __floordiv__(self, scalar: int):
        return V2(self.x // scalar, self.y // scalar)

    def __mod__(self, scalar: float):
        return V2(self.x % scalar, self.y % scalar)

    def __hash__(self):
        return hash((self.x, self.y))

    def floor(self) -> "V2i":
        return V2i(V2(int(self.x), int(self.y)))

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    def copy(self) -> "V2":
        return V2(self.x, self.y)

    def clamp(self, lower: float, upper: float) -> "V2":
        return V2(min(max(lower, self.x), upper), min(max(lower, self.y), upper))

    def as_rvec(self):
        return rl.Vector2(self.x, self.y)


V2i = NewType("V2i", V2)
VecType = Union[V2i, V2]


@dataclass
class SpriteSheet:
    texture: rl.Texture
    frames: int
    origin: rl.Vector2
    anims: Dict[str, List[int]]

    @property
    def width(self):
        return self.texture.width / self.frames

    def rect(self, frame_i: int) -> rl.Rectangle:
        return rl.Rectangle(self.width * frame_i, 0, self.width, self.texture.height)

    def draw(self, pos: V2):
        self.draw_frame(pos, 0, False)

    def draw_frame(self, pos: V2, i: int, fliph: bool):
        r = self.rect(int(i))
        if fliph:
            r.width = -r.width
        rl.draw_texture_pro(
            self.texture,
            r,
            rl.Rectangle(
                pos.x,
                pos.y,
                self.width,
                self.texture.height,
            ),
            self.origin,
            0,
            rl.WHITE,
        )


@dataclass
class Sprites:
    sprites: Dict[str, SpriteSheet] = field(default_factory=dict)

    def load(self, filename, nframes=1, origin=rl.Vector2(0, 0), anims={}):
        name = filename.split(".")[0]
        self.sprites[name] = SpriteSheet(rl.load_texture(filename), nframes, origin, anims)
        assert (
            self.sprites[name].texture.width % nframes == 0
        ), "not divisible by nframes"
        return name

    def __getattr__(self, name) -> SpriteSheet:
        return self.sprites[name]


@dataclass
class Knight:
    path: List[V2i]
    path_i: int
    pos: V2


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
GRID_COUNT = 24
GRID_SIZE = 16

if __name__ == "__main__":
    rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "game")


# TODO: look at updating makeStructHelper to allow kwargs along with positional args.
camera = rl.Camera2D(
    (
        SCREEN_WIDTH / 2 - GRID_COUNT * GRID_SIZE,
        SCREEN_HEIGHT / 2 - GRID_COUNT * GRID_SIZE,
    ),  # offset
    (0, 0),  # target
    0,  # rotation
    2,  # zoom
)

state = Namespace(
    player=Namespace(
        pos=V2(0, 0),
        next_pos=V2(0, 0),
        flip=False,
    ),
    knights=[],
)

LEVEL1 = """\
.......-................
.......-................
.......-................
.......-...w............
.......-...w............
.......-...w............
.......-...w............
.......-................
.......-----------......
.................-......
.................-......
.................-......
.................-......
.................-......
.................-......
.................-......
.................-......
.................-......
.................-......
.................-......
.................-......
.................-......
.................-......
.................-......\
""".split(
    "\n"
)

cur_level = LEVEL1


def tile_free(state, tile):
    return (
        cur_level[tile.y][tile.x] in (".", "-")
        and tile != state.player.pos
        and tile != state.player.next_pos
        and all(k.path[k.path_i] != tile for k in state.knights)
    )


sprites = Sprites()
sprites.load("knight.png", 2)
sprites.load("hero.png", 6, origin=rl.Vector2(0, 0), anims={'walk': [4, 5], 'idle': [0, 1], 'bonk': [0, 2, 3]})


def find_path(level):
    def find_start():
        for y, row in enumerate(level):
            for x, v in enumerate(row):
                if v == "-":
                    return V2(x, y)
        assert False

    cur = find_start()
    path = [cur]
    visited = {cur}
    while cur.y < len(level) - 1:
        for d in [V2(-1, 0), V2(1, 0), V2(0, -1), V2(0, 1)]:
            n = cur + d
            if (
                0 <= n.x < len(level[0])
                and 0 <= n.y < len(level)
                and level[n.y][n.x] == "-"
                and n not in visited
            ):
                path.append(n)
                visited.add(n)
                cur = n
                break
    return path


path = find_path(cur_level)
state.knights.append(Knight(path, 0, path[0]))
state.knights.append(Knight(path, 0, path[0] - V2(0, 10)))


def tween(pos: V2, target: V2, time: float):
    start = rl.get_time()
    orig_val = pos.copy()
    while rl.get_time() < start + time:
        t = (rl.get_time() - start) / time
        pos.set(orig_val + (target - orig_val) * t)
        yield

    pos.set(target)


def wait(time: float):
    start = rl.get_time()
    while rl.get_time() < start + time:
        yield (rl.get_time() - start) / time


input_tween = None


def step_game(state):
    t = 0
    global input_tween

    rl.begin_mode_2d(camera)

    move: Optional[V2] = None
    if rl.is_key_down(rl.KEY_LEFT):
        move = V2(-1, 0)
    if rl.is_key_down(rl.KEY_RIGHT):
        move = V2(1, 0)
    if rl.is_key_down(rl.KEY_UP):
        move = V2(0, -1)
    if rl.is_key_down(rl.KEY_DOWN):
        move = V2(0, 1)

    if input_tween is None:
        if rl.is_key_pressed(rl.KEY_SPACE):
            input_tween = wait(0.2)
        if move:
            state.player.flip = move.x < 0 if abs(move.x) > 0 else state.player.flip
            next_pos = (state.player.pos + move).clamp(0, GRID_COUNT - 1).floor()

            if next_pos != state.player.pos and tile_free(state, next_pos):
                input_tween = tween(state.player.pos, next_pos, 0.2)
                state.player.next_pos = next_pos
    else:
        try:
            t = next(input_tween)
        except StopIteration:
            input_tween = None

    if input_tween is not None:
        if input_tween.__name__ == 'tween':
            sprites.hero.draw_frame(
                state.player.pos * GRID_SIZE,
                sprites.hero.anims['walk'][int((rl.get_time() % 0.2) / 0.1)],
                state.player.flip,
            )
        elif input_tween.__name__ == 'wait':

            print(int(t * len(sprites.hero.anims['bonk'])))
            sprites.hero.draw_frame(
                state.player.pos * GRID_SIZE,
                sprites.hero.anims['bonk'][int(t * len(sprites.hero.anims['bonk']))],
                state.player.flip,
            )
    else:
        sprites.hero.draw_frame(state.player.pos * GRID_SIZE,
                                sprites.hero.anims['idle'][int((rl.get_time() % 5) > 4.9)],
                                state.player.flip)

    for i in range(GRID_COUNT):
        # rl.draw_line(0, i * GRID_SIZE, GRID_COUNT * GRID_SIZE, i * GRID_SIZE, rl.WHITE)
        # rl.draw_line(i * GRID_SIZE, 0, i * GRID_SIZE, GRID_COUNT * GRID_SIZE, rl.WHITE)

        for j, c in enumerate(cur_level[i]):
            if c == "w":
                rl.draw_rectangle(
                    j * GRID_SIZE, i * GRID_SIZE, GRID_SIZE, GRID_SIZE, rl.WHITE
                )
            if c == "-":
                line_width = 1
                if cur_level[i - 1][j] == "-":
                    rl.draw_rectangle(
                        j * GRID_SIZE + GRID_SIZE // 2 - line_width // 2,
                        (i - 1) * GRID_SIZE + GRID_SIZE // 2,
                        line_width,
                        line_width,
                        rl.WHITE,
                    )
                if cur_level[i][j - 1] == "-":
                    rl.draw_rectangle(
                        (j - 1) * GRID_SIZE + GRID_SIZE // 2,
                        i * GRID_SIZE + GRID_SIZE // 2 - line_width // 2,
                        line_width,
                        line_width,
                        rl.WHITE,
                    )

    for i, knight in enumerate(state.knights):
        goal = knight.path[knight.path_i] * GRID_SIZE
        diff = goal - knight.pos
        l = diff.length()
        if l < 0.7:
            knight.pos = goal
            if not knight.path_i + 1 < len(knight.path):
                del state.knights[i]
            else:
                next_goal = knight.path[knight.path_i + 1]
                if tile_free(state, next_goal):
                    knight.path_i += 1
        else:
            knight.pos += (diff / l) * 0.6
        sprites.knight.draw_frame(knight.pos, int((rl.get_time() % 0.2) / 0.1), False)

    # rl.draw_line(
    #     0,
    #     GRID_COUNT * GRID_SIZE,
    #     GRID_COUNT * GRID_SIZE,
    #     GRID_COUNT * GRID_SIZE,
    #     rl.WHITE,
    # )
    # rl.draw_line(
    #     GRID_COUNT * GRID_SIZE,
    #     0,
    #     GRID_COUNT * GRID_SIZE,
    #     GRID_COUNT * GRID_SIZE,
    #     rl.WHITE,
    # )

    rl.end_mode_2d()


if __name__ == "__main__":
    rl.set_target_fps(60)
    while not rl.window_should_close():
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)
        step_game(state)
        rl.end_drawing()
    rl.close_window()
