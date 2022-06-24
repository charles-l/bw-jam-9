import pyray as rl
import functools
from dataclasses import dataclass, field
from typing import (
    Tuple,
    List,
    Optional,
    Dict,
    TypeVar,
    NewType,
    Union,
    Generator,
    Any,
    cast,
)
from types import SimpleNamespace as Namespace
from perlin_noise import PerlinNoise
import math
import itertools

noise = PerlinNoise()

T = TypeVar("T")


def unwrap_optional(x: Optional[T]) -> T:
    assert x is not None
    return x


def namedcoro(name: Optional[str] = None):
    def inner(f):
        @functools.wraps(f)
        def g(*args, **kwargs):
            return NamedCoro(name, f(*args, **kwargs))

        return g

    return inner


class NamedCoro:
    def __init__(
        self,
        name: Optional[str],
        *generators: List[Generator[None, None, None] | "NamedCoro"]
    ):
        self.generators = list(generators)
        self._name = name

    @property
    def name(self):
        if self._name is not None:
            return self._name
        elif self.generators:
            if isinstance(self.generators[0], NamedCoro):
                return self.generators[0].name
            elif hasattr(self.generators[0], "__name__"):
                return self.generators[0].__name__
            else:
                assert False
        else:
            return None

    def __next__(self):
        try:
            return next(self.generators[0])
        except StopIteration:
            if len(self.generators) > 1:
                self.generators.pop(0)
                return self.__next__()
            else:
                raise


@dataclass(slots=True)
class V2:
    x: float
    y: float

    def set(self, other: "V2"):
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
        tex = rl.load_texture(filename)
        if origin.x < 0:
            origin.x = (tex.width // nframes) + origin.x
        if origin.y < 0:
            origin.y = (tex.height) + origin.y
        self.sprites[name] = SpriteSheet(tex, nframes, origin, anims)
        assert (
            self.sprites[name].texture.width % nframes == 0
        ), "not divisible by nframes"
        return name

    def __getattr__(self, name) -> SpriteSheet:
        return self.sprites[name]


class Knight:
    __slots__ = ('path_i', 'pos', 'flipped', 'coro')
    def __init__(self, path_i: int, pos: V2, flipped: bool = False):
        self.path_i = path_i
        self.pos = pos
        self.flipped = flipped
        self.coro = knight_brain(self)

    def rect(self):
        return rl.Rectangle(
            self.pos.x, self.pos.y, sprites.knight.width, sprites.knight.texture.height
        )


def knight_brain(self):
    while True:
        goal = state.path[self.path_i] * GRID_SIZE
        diff = goal - self.pos
        l = diff.length()
        if l < 0.7:
            self.pos = goal
            if not self.path_i + 1 < len(state.path):
                return "delete"
            else:
                next_goal = state.path[self.path_i + 1]
                if tile_free(state, next_goal):
                    self.flipped = (next_goal * GRID_SIZE - self.pos).x < 0
                    self.path_i += 1
                else:
                    if not any(
                        (k.pos / GRID_SIZE).floor() == next_goal for k in state.knights
                    ):
                        yield from wait(0.1, C("slice", frame=0))
                        yield from wait(0.1, C("slice", frame=1))
                        hit(next_goal)
                        yield from wait(0.1, C("slice", frame=2))
        else:
            self.pos += (diff / l) * 0.6
        yield "move"


@dataclass(slots=True)
class Dart:
    pos: V2
    vel: V2


@dataclass(slots=True)
class Wall:
    pass


@dataclass(slots=True)
class BreakWall:
    health: int = 10

    def hit(self):
        self.health -= 1


@dataclass(slots=True)
class Pillar:
    _pos: V2
    coro: Optional[Generator[None, None, None]] = None
    flipped: bool = False

    def fire(self):
        last_time = rl.get_time()
        while True:
            yield "fire"
            if rl.get_time() - last_time > 1:
                d = -1 if self.flipped else 1
                state.darts.append(
                    Dart(self._pos * GRID_SIZE + V2(14 * d, 5), V2(2 * d, 0))
                )
                last_time = rl.get_time()

    def stun(self):
        yield from wait(0.2, "shake")
        yield from wait(3.0, "disable")
        yield from wait(0.2, "telegraph")

    def hit(self):
        self.coro = self.stun()

    def update(self):
        if self.coro is None:
            self.coro = self.fire()

        try:
            return next(self.coro)
        except StopIteration:
            self.coro = None

        if self.coro is None:
            self.coro = self.fire()


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
GRID_COUNT = 24
GRID_SIZE = 16

if __name__ == "__main__":
    rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "game")

FONT = rl.load_font("TimesNewPixel.fnt", 16)

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
        shake_amount=0,
    ),
    knights=[],
    darts=[],
    path=[],
    cur_level=0,
)

LEVEL1 = """\
.......-...w............
.......-...w............
.......-...w............
.......-...w............
...>...-...w............
.......-...w............
.......-...w............
...>...-...w............
.......-...w............
.......-...w............
.......-................
.......-................
...p...-----------......
.................-......
.................-......
...........w.....-......
...>.......w.....-......
...........w.....-......
...........w.....-......
...........w.....--.....
...........w......-.....
...........w.....--.....
...........w.....-......
...........w.....-......\
""".split(
    "\n"
)

LEVEL2 = """\
.......-................
...>...-................
.......-................
.......-...<............
.......-................
.......-................
.......-...<............
.......-................
.......-................
...>...-................
.......-................
.......-................
...p...-................
.......-................
.......-................
.......-................
.......-................
....www*www.............
....w..-..w.............
....w..-..w.............
....w..-..w.............
....w..-..w.............
....w..-..w.............
....w..-..w.............\
""".split(
    "\n"
)

Tile = Union[Wall, BreakWall, Pillar]

foreground_layer: Dict[Tuple[int, int], Tile] = {}


def load_level(level):
    print("load level", state)
    foreground_layer.clear()
    for y, row in enumerate(level):
        for x, c in enumerate(row):
            if c == "*":
                foreground_layer[(x, y)] = BreakWall()
            if c == "w":
                foreground_layer[(x, y)] = Wall()
            elif c == ">":
                foreground_layer[(x, y)] = Pillar(V2(x, y), flipped=False)
            elif c == "<":
                foreground_layer[(x, y)] = Pillar(V2(x, y), flipped=True)
            elif c == "p":
                state.player.pos = V2(x, y)

    state.path = find_path(level)


def level_1():
    load_level(LEVEL1)
    for x in wait(4):
        yield

    state.knights.append(Knight(0, state.path[0] * GRID_SIZE))
    state.knights.append(Knight(0, state.path[0] * GRID_SIZE + V2(0, 20)))
    state.knights.append(Knight(0, state.path[0] * GRID_SIZE + V2(0, 40)))
    state.knights.append(Knight(0, state.path[0] * GRID_SIZE + V2(0, 60)))
    state.knights.append(Knight(0, state.path[0] * GRID_SIZE + V2(0, 80)))

    while state.knights:
        yield


def level_2():
    load_level(LEVEL2)
    for x in wait(2):
        yield

    state.knights.append(Knight(0, state.path[0] * GRID_SIZE))
    state.knights.append(Knight(0, state.path[0] * GRID_SIZE + V2(0, 20)))
    state.knights.append(Knight(0, state.path[0] * GRID_SIZE + V2(0, 40)))
    state.knights.append(Knight(0, state.path[0] * GRID_SIZE + V2(0, 60)))
    state.knights.append(Knight(0, state.path[0] * GRID_SIZE + V2(0, 80)))

    while state.knights:
        yield


def tile_free(state, tile):
    return (
        (tile.x, tile.y) not in foreground_layer
        and tile != state.player.pos
        and tile != state.player.next_pos
        and all(state.path[k.path_i] != tile for k in state.knights)
    )


sprites = Sprites()
sprites.load("knight.png", 4, anims={"slice": [0, 2, 3]})
sprites.load(
    "hero.png",
    6,
    origin=rl.Vector2(0, 0),
    anims={"walk": [4, 5], "idle": [0, 1], "bonk": [0, 2, 3]},
)
sprites.load(
    "pillar.png",
    4,
    origin=rl.Vector2(0, -GRID_SIZE),
    anims={"lit": [0, 1, 2], "snuffed": [3]},
)
sprites.load("breakwall.png", 4)


def find_path(level):
    def find_start():
        for y, row in reversed(list(enumerate(level))):
            for x, v in enumerate(row):
                if v == "-":
                    return V2(x, y)
        assert False

    cur = find_start()
    path = [cur]
    visited = {cur}
    steps = 0
    while cur.y > 0 and steps < 128:
        for d in [V2(-1, 0), V2(1, 0), V2(0, -1), V2(0, 1)]:
            n = cur + d
            if (
                0 <= n.x < len(level[0])
                and 0 <= n.y < len(level)
                and level[n.y][n.x] in "-*"
                and n not in visited
            ):
                path.append(n)
                visited.add(n)
                cur = n
                break
        steps += 1
    return path


def tween(pos: V2, target: V2, time: float, y: str = "tween"):
    start = rl.get_time()
    orig_val = pos.copy()
    while rl.get_time() < start + time:
        t = (rl.get_time() - start) / time
        pos.set(orig_val + (target - orig_val) * t)
        yield y

    pos.set(target)


def wait(time: float, y: str = "wait"):
    start = rl.get_time()
    while rl.get_time() < start + time:
        yield y


def final_screen():
    global state
    state.path.clear()
    foreground_layer.clear()
    state.knights.clear()
    while True:
        yield


input_tween = None

level_coros = [level_1(), level_2(), final_screen()]


class ContextStr:
    def __init__(self, s: str, **kwargs: Dict[str, Any]):
        self.str = s
        self.ctx = kwargs

    def __eq__(self, other):
        if isinstance(other, str):
            return self.str == other

    def __getitem__(self, k):
        return self.ctx[k]

C = ContextStr


def hit(pos: V2):
    if state.player.pos.floor() == pos:
        state.player.shake_amount = 4

    p = (pos[0], pos[1])
    if p in foreground_layer and hasattr(foreground_layer[p], "hit"):
        foreground_layer[p].hit()


def step_game(state):
    global input_tween

    try:
        next(level_coros[state.cur_level])
    except StopIteration:
        state.cur_level += 1

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
            input_tween = itertools.chain(
                wait(0.1, C("bonk", frame=0)),
                wait(0.1, C("bonk", frame=1)),
                wait(0.1, C("bonk", frame=2)),
            )
        if move:
            state.player.flip = move.x < 0 if abs(move.x) > 0 else state.player.flip
            next_pos = (state.player.pos + move).clamp(0, GRID_COUNT - 1).floor()

            if next_pos != state.player.pos and tile_free(state, next_pos):
                input_tween = tween(state.player.pos, next_pos, 0.2, "move")
                state.player.next_pos = next_pos

    if input_tween is not None:
        try:
            input_state = next(input_tween)
        except StopIteration:
            input_tween = None

    state.player.shake_amount = max(state.player.shake_amount - 0.2, 0)
    shake_vec = (
        V2(
            noise(state.player.pos.x + rl.get_time() * 10),
            noise(state.player.pos.y + rl.get_time() * 10 + 4),
        )
        * state.player.shake_amount
    )

    # draw after all updates
    if input_tween is not None:
        assert input_state is not None
        if input_state == "move":
            sprites.hero.draw_frame(
                state.player.pos * GRID_SIZE + shake_vec,
                sprites.hero.anims["walk"][int((rl.get_time() % 0.2) / 0.1)],
                state.player.flip,
            )
        elif input_state == "bonk":
            sprites.hero.draw_frame(
                state.player.pos * GRID_SIZE,
                sprites.hero.anims["bonk"][input_state['frame']],
                state.player.flip,
            )
            if input_state['frame'] > 1:
                bonked_tile_pos = V2(
                    state.player.pos.x + (-1 if state.player.flip else 1),
                    state.player.pos.y,
                )
                if bonked_tile_pos in foreground_layer:
                    hit(bonked_tile_pos)
        else:
            assert False, input_state
    else:
        sprites.hero.draw_frame(
            state.player.pos * GRID_SIZE + shake_vec,
            sprites.hero.anims["idle"][int((rl.get_time() % 5) > 4.9)],
            state.player.flip,
        )

    tiles_to_del = []
    for pos, tile in foreground_layer.items():
        if isinstance(tile, Wall):
            rl.draw_rectangle(
                pos[0] * GRID_SIZE, pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE, rl.WHITE
            )
        if isinstance(tile, BreakWall):
            sprites.breakwall.draw_frame(V2(pos[0] * GRID_SIZE, pos[1] * GRID_SIZE), int((1 - (tile.health / 10)) * 4), False)
            if tile.health <= 0:
                tiles_to_del.append(pos)
        elif isinstance(tile, Pillar):
            tile_state = tile.update()
            if tile_state == "shake":
                shake_vec = (
                    V2(noise(rl.get_time() * 10), noise(rl.get_time() * 10 + 4)) * 4
                )
                sprites.pillar.draw_frame(
                    V2(*pos) * GRID_SIZE + shake_vec,
                    sprites.pillar.anims["snuffed"][0],
                    tile.flipped,
                )
            elif tile_state == "telegraph":
                sprites.pillar.draw_frame(
                    V2(*pos) * GRID_SIZE + V2(0, -2),
                    sprites.pillar.anims["snuffed"][0],
                    tile.flipped,
                )
            elif tile_state == "disable":
                sprites.pillar.draw_frame(
                    V2(*pos) * GRID_SIZE,
                    sprites.pillar.anims["snuffed"][0],
                    tile.flipped,
                )
            else:
                sprites.pillar.draw_frame(
                    V2(*pos) * GRID_SIZE,
                    sprites.pillar.anims["lit"][int((rl.get_time() % 0.3) / 0.1)],
                    tile.flipped,
                )

    for p in tiles_to_del:
        del foreground_layer[p]

    for p in state.path:
        line_width = 1
        rl.draw_rectangle(
            p[0] * GRID_SIZE + GRID_SIZE // 2 - line_width // 2,
            p[1] * GRID_SIZE + GRID_SIZE // 2 - line_width // 2,
            line_width,
            line_width,
            rl.WHITE,
        )

    for dart_i, dart in enumerate(state.darts):
        dart.pos += dart.vel
        dart_rect = rl.Rectangle(dart.pos.x, dart.pos.y, 5, 1)
        rl.draw_rectangle_rec(dart_rect, rl.WHITE)
        destroy_dart = False
        for knight_i, knight in enumerate(state.knights):
            if rl.check_collision_recs(dart_rect, knight.rect()):
                del state.knights[knight_i]
                destroy_dart = True
                break

        if (
            0 <= dart.pos.x < GRID_SIZE * GRID_COUNT
            and 0 <= dart.pos.y < GRID_SIZE * GRID_COUNT
        ):
            destroy_dart |= (
                int(dart.pos.x / GRID_SIZE),
                int(dart.pos.y / GRID_SIZE),
            ) in foreground_layer
            hit_player = rl.check_collision_recs(
                dart_rect,
                rl.Rectangle(
                    state.player.pos.x * GRID_SIZE,
                    state.player.pos.y * GRID_SIZE,
                    sprites.hero.width,
                    sprites.hero.texture.height,
                ),
            )
            if hit_player:
                state.player.shake_amount = 4
            destroy_dart |= hit_player
        else:
            destroy_dart = True

        if destroy_dart:
            del state.darts[dart_i]

    for i, knight in enumerate(state.knights):
        kstate = "move"
        try:
            kstate = next(knight.coro)
        except StopIteration as e:
            if e.value == "delete":
                del state.knights[i]

        if kstate == "move":
            sprites.knight.draw_frame(
                knight.pos, int((rl.get_time() % 0.2) / 0.1), knight.flipped
            )
        elif kstate == "slice":
            sprites.knight.draw_frame(
                knight.pos,
                sprites.knight.anims["slice"][kstate["frame"]],
                knight.flipped,
            )
        else:
            assert False

    rl.end_mode_2d()

    # draw UI

    text = ""
    if text:
        v = rl.measure_text_ex(FONT, text, 32, 1)
        text_box_width = int(v.x) + 10
        text_box_height = int(v.y)
        top_left = V2(
            SCREEN_WIDTH // 2 - text_box_width // 2,
            SCREEN_HEIGHT - text_box_height - 20,
        )
        rl.draw_rectangle(
            *top_left - V2(2, 2), text_box_width + 4, text_box_height + 4, rl.WHITE
        )
        rl.draw_rectangle(*top_left, text_box_width, text_box_height, rl.BLACK)
        rl.draw_text_ex(FONT, text, rl.Vector2(*(top_left + V2(4, 2))), 32, 1, rl.WHITE)


if __name__ == "__main__":
    rl.set_target_fps(60)
    while not rl.window_should_close():
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)
        step_game(state)
        rl.end_drawing()
    rl.close_window()
