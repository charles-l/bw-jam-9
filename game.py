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
            rl.Vector2(0, 0),
            0,
            rl.WHITE,
        )


@dataclass
class Sprites:
    sprites: Dict[str, SpriteSheet] = field(default_factory=dict)

    def load(self, filename, nframes=1):
        name = filename.split(".")[0]
        self.sprites[name] = SpriteSheet(rl.load_texture(filename), nframes)
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
    player=Namespace(pos=V2(0, 0)),
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
    return cur_level[tile.y][tile.x] in ('.', '-') and tile != state.player.pos and all(k.path[k.path_i] != tile for k in state.knights)

sprites = Sprites()
sprites.load("knight.png", 2)


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

def step_game(state):
    rl.begin_mode_2d(camera)

    next_pos = state.player.pos.copy()
    if rl.is_key_pressed(rl.KEY_LEFT):
        next_pos.x -= 1
    if rl.is_key_pressed(rl.KEY_RIGHT):
        next_pos.x += 1
    if rl.is_key_pressed(rl.KEY_UP):
        next_pos.y -= 1
    if rl.is_key_pressed(rl.KEY_DOWN):
        next_pos.y += 1

    next_pos = next_pos.clamp(0, GRID_COUNT - 1)

    if tile_free(state, next_pos):
        state.player.pos = next_pos

    rl.draw_rectangle(
        state.player.pos.x * GRID_SIZE,
        state.player.pos.y * GRID_SIZE,
        GRID_SIZE,
        GRID_SIZE,
        rl.WHITE,
    )

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
