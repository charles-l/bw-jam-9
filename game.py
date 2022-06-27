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
import sys
import copy

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
        *generators: List[Generator[None, None, None] | "NamedCoro"],
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

    def round(self):
        return V2i(
            V2(
                int(
                    math.floor(self.x + 0.5) if self.x >= 0 else math.ceil(self.x - 0.5)
                ),
                int(
                    math.floor(self.y + 0.5) if self.y >= 0 else math.ceil(self.y - 0.5)
                ),
            )
        )

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    def copy(self) -> "V2":
        return V2(self.x, self.y)

    def clamp(self, lower: float, upper: float) -> "V2":
        return V2(min(max(lower, self.x), upper), min(max(lower, self.y), upper))

    def as_rvec(self):
        return rl.Vector2(self.x, self.y)

    def norm(self):
        l = self.length()
        if l:
            return self / l
        else:
            return V2(0, 0)

    def astuple(self):
        return (self.x, self.y)


class V2Pause(V2):
    pass


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


class DeadKnight:
    pass


class Planks:
    pass


class Knight:
    __slots__ = ("path_i", "pos", "flipped", "coro")

    def __init__(self, path_i: int, pos: V2, flipped: bool = False):
        self.path_i = path_i
        self.pos = pos
        self.flipped = flipped
        self.coro = knight_brain(self)

    def rect(self):
        return rl.Rectangle(
            self.pos.x, self.pos.y, sprites.knight.width, sprites.knight.texture.height
        )

def knight_die(self):
    yield "die"
    return "die"


def knight_brain(self):
    while True:
        goal = state.path[self.path_i] * GRID_SIZE
        diff = goal - self.pos
        l = diff.length()
        if l < 0.7:
            self.pos = goal
            t = (self.pos // GRID_SIZE).floor().astuple()
            if t in foreground_layer and isinstance(foreground_layer[t], Spikes):
                return "die"
            if not self.path_i + 1 < len(state.path):
                return "pass"
            else:
                next_goal = state.path[self.path_i + 1]
                self.flipped = ((next_goal * GRID_SIZE) - self.pos).x < 0
                if tile_free(state, next_goal):
                    self.path_i += 1
                else:
                    if not any(
                        (k.pos / GRID_SIZE).floor() == next_goal
                        for k in state.knights
                    ):
                        yield from wait(0.1, C("slice", frame=0))
                        yield from wait(0.1, C("slice", frame=1))
                        hit(next_goal, (next_goal - self.pos).norm())
                        yield from wait(0.1, C("slice", frame=2))
        else:
            self.pos += (diff / l) * 0.6
        yield "move"


@dataclass(slots=True)
class Lermin:
    pos: V2
    coro: Any = None
    vulnerable: bool = False

    def drop_bombs(self):
        while True:
            yield from wait(4)
            launch_bomb((self.pos / GRID_SIZE).round(), V2(0, 1), 2)
            launch_bomb((self.pos / GRID_SIZE).round(), V2(1, 1), 1)
            launch_bomb((self.pos / GRID_SIZE).round(), V2(-1, 1), 1)


    def hit(self, d):
        if self.vulnerable:
            explode_sprite(self.pos.floor(), delay=0.1)
            explode_sprite(self.pos.floor() + V2(0.1, 0.3), delay=0.4)
            explode_sprite(self.pos.floor() - V2(-0.3, -0.1), delay=0.8)
            state.boss = None


@dataclass(slots=True)
class Dart:
    pos: V2
    vel: V2


class Bomb:
    pos: V2
    move_coro: Optional[Generator[None, None, None]]
    anim_coro: Any

    def __init__(
        self, pos: V2, move_coro: Optional[Generator[None, None, None]], live=False
    ):
        self.pos = pos
        self.move_coro = move_coro
        if live:
            self.anim_coro = self.anim()
        else:
            self.anim_coro = None

    def anim(self):
        for i in range(sprites.bomb.frames):
            yield from wait(0.2, C("anim", frame=i))

        ll = ((self.pos // GRID_SIZE).round() - V2(1, 1)) * GRID_SIZE

        self.explode()
        explode_sprite(self.pos, shockwave=True)

    def hit(self, d):
        goal = find_open_tile((self.pos // GRID_SIZE).round(), d, 2)
        if tile_free(state, goal):
            self.move_coro = tween(self.pos, goal * GRID_SIZE, 0.5)
        if not self.anim_coro:
            self.anim_coro = self.anim()

    def explode(self):
        for d in [
            V2(-1, -1),
            V2(0, -1),
            V2(1, -1),
            V2(-1, 0),
            V2(0, 0),
            V2(1, 0),
            V2(-1, 1),
            V2(0, 1),
            V2(1, 1),
        ]:
            hit((((self.pos / GRID_SIZE) + d)).floor(), d, destroy=True)


@dataclass
class StaticSprite:
    sprite: Any


class Wall:
    pass


class Spikes:
    pass


@dataclass(slots=True)
class BreakWall:
    health: int = 10

    def hit(self, d):
        self.health -= 1


def find_open_tile(start_pos, direction, max_dist):
    assert max_dist > 0
    # BUG: BORKED if we throw it off the edge
    goal = start_pos + direction
    if tile_free(state, goal):
        for i in range(max_dist - 1):
            if not tile_free(state, goal + direction):
                break
            goal += direction
    return goal


def launch_bomb(pos: V2, d: V2, max_dist):
    goal = find_open_tile(pos, d, max_dist)

    b = Bomb(pos * GRID_SIZE, None, live=True)
    b.move_coro = tween(
        b.pos, goal * GRID_SIZE, (goal - pos).length() / max_dist
    )
    state.bombs.append(b)


class Cannon:
    def __init__(self, pos, fire_dist=6, flipped=False):
        self.pos = pos
        self.coro = self.fire()
        self.fire_dist = fire_dist
        self.flipped = flipped

    def fire(self):
        while True:
            yield from wait(4, C("anim", frame=0))
            d = -1 if self.flipped else 1
            launch_bomb(self.pos, V2(d, 0), self.fire_dist)
            yield from wait(0.1, C("anim", frame=1))
            yield from wait(0.1, C("anim", frame=2))
            yield from wait(0.1, C("anim", frame=3))

    def hit(self, d):
        pass


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

    def hit(self, d):
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
    rl.init_audio_device()

FONT = rl.load_font("TimesNewPixel.fnt", 16)

rl.gui_set_font(FONT)
rl.gui_set_style(rl.DEFAULT, rl.TEXT_SIZE, 32)

rl.gui_set_style(rl.DEFAULT, rl.BASE_COLOR_NORMAL, 0)
rl.gui_set_style(rl.DEFAULT, rl.BORDER_COLOR_NORMAL, -1)
rl.gui_set_style(rl.DEFAULT, rl.TEXT_COLOR_NORMAL, -1)

rl.gui_set_style(rl.DEFAULT, rl.BASE_COLOR_FOCUSED, -1)
rl.gui_set_style(rl.DEFAULT, rl.BORDER_COLOR_FOCUSED, 0)
rl.gui_set_style(rl.DEFAULT, rl.TEXT_COLOR_FOCUSED, 0)

rl.gui_set_style(rl.DEFAULT, rl.BASE_COLOR_PRESSED, -1)
rl.gui_set_style(rl.DEFAULT, rl.BORDER_COLOR_PRESSED, -1)
rl.gui_set_style(rl.DEFAULT, rl.TEXT_COLOR_PRESSED, 0)

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
        alive=True,
    ),
    knights_alive=15,
    knights=[],
    darts=[],
    bombs=[],
    path=[],
    boss=None,
    cur_level=0,
)

starting_knights_alive = state.knights_alive

INTRO = """\
ww2wwwwwwwwwwwwwwwww2www
w3wwwwwwwwwwww3wwwwwwwww
wwwww3wwww1.wwwwwwwwwwww
wwwwwwwwww..wwwwwwwwwwww
........................
........................
.................P......
........................
........................
........................
........................
........................
........................
........................
........................
........................
........................
........................
........................
........................
........................
........................
........................
........................\
""".split(
    "\n"
)


LEVEL1 = """\
wwww...-...w............
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
...P...-----------......
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
...>...-B...............
wwwwwww*ww....wwwwwwwwww
.......-...<............
.......-................
.......-................
.......-................
.......-................
.......-...<............
.......-................
..>..B.-................
.......-................
...P...-................
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

LEVEL3 = """\
...........-............
...........-......).....
...........-............
...........-......).....
...........-............
***---------......).....
*wwww...................
*.....BB.<..............
-w.....B.#..............
-w.......#..............
-w.......#..............
-w####wwwwwwwwwwww......
-..B....................
-.......................
-----------.............
-.........-.............
wwwww....>--...)........
...........-............
...........-....P.......
------------............
-...wwwwwwwwwwwwwwwww...
------------............
...........-............
...........-............\
""".split(
    "\n"
)

LEVEL4 = """\
...........-.....w......
...........-.<...w......
...........-..B<.w......
...........-.<...w......
...........-..B<.w......
....--------.<B..w......
....-......B.....w......
....-......<.....w......
....-.....B......w......
....-......<.....w......
....-.....B......w......
wwww*......<B....w......
wwww*wwwwwww...B.wwwwwww
....--------*****-----..
.....................-..
.....................-..
...............).....-..
.....................-..
................P....-..
.....................-..
.....................-..
...........-----------..
...........-............
...........-............\
""".split(
    "\n"
)

LEVEL5 = """\
...............w-w......
...............w-w......
...............w-w......
...............w-w......
...............w-w......
...............w-w......
...............w-w......
.........wwwwwww-w......
........-p--^^---w.....
........-wwwwwwwww......
....-----w..............
....-...................
....-...ss..............
....----^^p-....wwwwww..
...........-....w....w..
...........-....#....w..
...........-....wwwwww..
...........-............
...........-.....P......
...........-............
...........-............
...........-............
...........-............
...........-............\
""".split(
    "\n"
)

BOSS_FIGHT = """\
...........-............
...........-............
...........-............
...........-............
...........-............
...........-............
...........-............
...........-............
...........-............
...........-............
...........$............
...........-............
...........-............
...........-............
...........-............
...........-............
.....P.....-............
...........-............
...........-............
...........-............
...........-............
...........-............
...........-............
...........-............\
""".split(
    "\n"
)


Tile = Union[Wall, BreakWall, Pillar, DeadKnight, Planks]

foreground_layer: Dict[Tuple[int, int], Tile] = {}
bgmusic = rl.load_music_stream("bg.mp3")
rl.play_music_stream(bgmusic)
can_control = True


def load_level(level, do_find_path=True):
    global can_control
    foreground_layer.clear()
    state.darts.clear()
    state.bombs.clear()
    state.knights.clear()
    state.boss = None
    can_control = True

    for y, row in enumerate(level):
        for x, c in enumerate(row):
            if c == "*" or c == "#":
                foreground_layer[(x, y)] = BreakWall()
            if c == "w":
                foreground_layer[(x, y)] = Wall()
            if c == "$":
                state.boss = Lermin(V2(x, y) * GRID_SIZE)
            if c == "^":
                foreground_layer[(x, y)] = Spikes()
            if c == "s":
                foreground_layer[(x, y)] = Spikes()
            if c == "p":
                foreground_layer[(x, y)] = Planks()
            elif c == ">":
                foreground_layer[(x, y)] = Pillar(V2(x, y), flipped=False)
            elif c == "<":
                foreground_layer[(x, y)] = Pillar(V2(x, y), flipped=True)
            elif c == ")":
                foreground_layer[(x, y)] = Cannon(V2(x, y), flipped=True)
            elif c == "B":
                state.bombs.append(Bomb(V2(x, y) * GRID_SIZE, None))
            elif c == "P":
                state.player.pos = V2(x, y)
            elif c == "1":
                foreground_layer[(x, y)] = StaticSprite(sprites.portcullis)
            elif c == "2":
                foreground_layer[(x, y)] = StaticSprite(sprites.wall_variant_1)
            elif c == "3":
                foreground_layer[(x, y)] = StaticSprite(sprites.wall_variant_2)

    if do_find_path:
        state.path = find_path(level)


def choice_buttons(*options):
    sizes = [rl.measure_text_ex(FONT, opt, 32, 1) for opt in options]
    total_width = sum(v.x + 40 for v in sizes) - 20
    top_left_x = (SCREEN_WIDTH // 2) - (total_width // 2)
    top_left_y = SCREEN_HEIGHT - sizes[0].y - 20

    offsetx = top_left_x
    rl.draw_rectangle(
        int(top_left_x) - 8, int(top_left_y) - 8, int(total_width) + 16, 20, rl.WHITE
    )
    for text, size in zip(options, sizes):
        if rl.gui_button(
            rl.Rectangle(offsetx, top_left_y, size.x + 20, size.y + 4), text
        ):
            return text
        offsetx += size.x + 40


def intro():
    global text, portrait, portrait_emotion
    load_level(INTRO, do_find_path=False)

    yield from wait(0.1)

    portrait = sprites.hero_portrait
    text = "Yoooo"
    yield from wait_for_click()

    text = "I'm just chillin'"
    yield from wait_for_click()

    text = "living the necromanced corpse life."
    yield from wait_for_click()

    text = "hanging out,"
    yield from wait_for_click()

    text = "with a stick that's *great* for bonking things"
    yield from wait_for_click()

    text = "(which you can use by hitting SPACE, btw)"
    yield from wait_for_click()

    portrait_emotion = "sad"
    text = "You know. It isn't what it's cracked up to be."
    yield from wait_for_click()

    portrait_emotion = None
    text = "The necromanced life, I mean. Not the stick."
    yield from wait_for_click()

    text = "I was once an adventurer like you,"
    yield from wait_for_click()

    text = "then I took an arrow to the knee."
    yield from wait_for_click()

    text = "Kidding! A necromancer ripped out my heart."
    yield from wait_for_click()

    text = "Then he brought me back"
    yield from wait_for_click()

    text = "to guard his *dumb* castle."
    yield from wait_for_click()

    text = "And I've been stuck at this"
    yield from wait_for_click()

    portrait_emotion = "anger"
    text = "FOR 500 YEARS!"
    yield from wait_for_click()

    portrait_emotion = "sad"
    text = "*sigh*"
    yield from wait_for_click()

    text = "..."
    yield from wait_for_click()

    text = "My death lead to an unfortunate new beginning."
    yield from wait_for_click()

    portrait_emotion = None
    text = "If only I could ~conveniently~ trigger"
    yield from wait_for_click()

    text = "an event that leads to the downfall of LERMIN."
    yield from wait_for_click()

    text = "Oh. Yeah. That's the necromancer's name."
    yield from wait_for_click()

    text = '"Lermin"'
    yield from wait_for_click()

    text = "I know. Lame."
    yield from wait_for_click()

    portrait_emotion = "sad"
    text = "*sigh*"
    yield from wait_for_click()

    portrait_emotion = None
    text = "..."
    yield from wait_for_click()

    portrait_emotion = "anger"
    text = "I said: *SIGH*"
    yield from wait_for_click()

    portrait_emotion = None

    kpos = V2(SCREEN_WIDTH // 2, SCREEN_HEIGHT + 20)
    k = Knight(0, kpos)

    def knight_coro():
        yield from tween(kpos, (state.player.pos + V2(1, 0)) * GRID_SIZE, 4, "move")
        while True:
            yield C("slice", frame=0)

    k.coro = knight_coro()
    state.knights.append(k)

    def f():
        global portrait, text, portrait_emotion
        portrait = sprites.knight_portrait
        text = "Ho there!"
        yield from wait(4)

        text = "You look like a... strapping young skeleton"
        yield from wait_for_click()

        text = "Perhaps you'd like us to help us?"
        yield from wait_for_click()

        text = "We're here to defeat the evil sourcerer Lermin."
        yield from wait_for_click()

        portrait = None
        text = "Help overthrow Lermin?"
        while (choice := choice_buttons("yes", "no")) is None:
            yield

        if choice == "no":
            portrait = sprites.knight_portrait
            text = "Curses."
            yield from tween(kpos, V2(SCREEN_WIDTH // 2, SCREEN_HEIGHT + 20), 4)

            portrait = sprites.hero_portrait
            text = "Hmph. Might as well see what they'll do"
            yield from wait_for_click()

        if choice == "yes":
            yield
            portrait = sprites.knight_portrait
            text = "Wonderful!"
            yield from wait_for_click()

            text = "All you have to do is direct us,"
            yield from wait_for_click()

            text = "let us know where all the traps are,"
            yield from wait_for_click()

            text = "mark our path,"
            yield from wait_for_click()

            text = "provide us with resources,"
            yield from wait_for_click()

            text = "act as a human-"
            yield from wait_for_click()

            text = "-err... skeleton shield,"
            yield from wait_for_click()

            text = "(as you are unkillable),"
            yield from wait_for_click()

            text = "give orders,"
            yield from wait_for_click()

            text = "take orders,"
            yield from wait_for_click()

            text = "and withstand any and all attacks from Lermin."
            yield from wait_for_click()

            portrait = sprites.hero_portrait
            portrait_emotion = "anger"
            text = "..."
            yield from wait_for_click()

            portrait = sprites.knight_portrait
            portrait_emotion = None

            text = "Also! Nobody can know you're working with us."
            yield from wait_for_click()

            text = "Bad optics and all. I'm sure you understand."
            yield from wait_for_click()

            text = "So if you stand next to me,"
            yield from wait_for_click()

            text = "or anyone else, we *will* hit you."
            yield from wait_for_click()

            text = "Alright old chap. On with it!"
            yield from wait_for_click()

            portrait = sprites.hero_portrait
            portrait_emotion = "anger"
            text = "..."
            yield from wait_for_click()

            portrait_emotion = "sad"
            text = "Fine... let's get this over with."
            yield from wait_for_click()

    it = f()
    for _ in it:
        if not state.knights:
            portrait = sprites.hero_portrait
            portrait_emotion = None
            text = "Oops."
            yield from wait_for_click()

            text = "Welp, I guess that's another uprising quelled."
            yield from wait_for_click()

            portrait = None
            text = "Game completed (Ending 1): the fast any% route"
            yield from wait_for_click()

            sys.exit(0)
        else:
            yield

    text = ""
    portrait = None


def spawn_waves():
    global text
    wave_i = 0
    knights_alive = state.knights_alive
    nwaves, last_wave_count = divmod(knights_alive, 5)
    nwaves_display = nwaves + (1 if last_wave_count > 0 else 0)
    for wave_i in range(nwaves):
        text = f"wave {wave_i + 1}/{nwaves_display}"
        for i in range(5):
            state.knights.append(Knight(0, state.path[0] * GRID_SIZE + V2(0, i * 20)))

        yield from wait(10)

    if last_wave_count:
        wave_i += 1

    text = f"wave {wave_i+1}/{nwaves_display}"
    for i in range(last_wave_count):
        state.knights.append(Knight(0, state.path[0] * GRID_SIZE + V2(0, i * 20)))

    while state.knights:
        yield

    text = ""


def level_1():
    load_level(LEVEL1)
    yield from wait(4)

    yield from spawn_waves()


def level_2():
    load_level(LEVEL2)
    yield from wait(2)

    yield from spawn_waves()


def level_3():
    load_level(LEVEL3)
    yield from wait(2)

    yield from spawn_waves()


def level_4():
    load_level(LEVEL4)
    yield from wait(2)

    yield from spawn_waves()


def level_5():
    load_level(LEVEL5)
    yield from wait(2)

    yield from spawn_waves()


def boss_fight():
    global text, portrait, portrait_emotion, can_control
    load_level(BOSS_FIGHT)
    can_control = False

    yield from wait(1)

    portrait = sprites.lermin_portrait
    text = "I take it, this is your resignation?"
    yield from wait_for_click()

    portrait = sprites.hero_portrait
    text = "Yeah, I'm sick of you and this endless cycle."
    yield from wait_for_click()

    portrait_emotion = "anger"
    text = 'Also, you were a lousy "boss"'
    yield from wait_for_click()

    portrait = sprites.lermin_portrait
    portrait_emotion = None
    text = "Oh, you don't understand."
    yield from wait_for_click()

    text = "This will be your resignation from life."
    yield from wait_for_click()

    text = "You will not survive without my power."
    yield from wait_for_click()

    while (choice := choice_buttons("Back down", 'Lead the charge')) is None:
        yield

    yield

    if choice == 'Back down':
        portrait = sprites.hero_portrait
        text = "Err... ok, yeah you have a good point there."
        yield from wait_for_click()

        text = "I'm bailing."
        yield from wait_for_click()

        can_control = True

        portrait = None
        text = ""

        state.boss.vulnerable = True
        while state.player.pos.floor().y != GRID_COUNT - 1:
            if state.boss is None:
                text = "Very cheeky."
                yield from wait_for_click()

                text = "The old, hit 'em when they least expect it"
                yield from wait_for_click()

                explode_sprite(state.player.pos)
                state.player.alive = False

                portrait = sprites.hero_portrait
                text = "Worth it..."
                yield from wait_for_click()

                portrait = None
                text = "Game over (Ending 4): very cheeky"
                yield from wait_for_click()
                sys.exit(0)

            yield

        state.player.alive = False
        text = "With that, the story ends."
        yield from wait_for_click()

        text = "Our hero, returns to his post,"
        yield from wait_for_click()

        text = "and continues to live an uneventful life,"
        yield from wait_for_click()

        text = "indefinitely..."
        yield from wait_for_click()

        text = "Game over (Ending 5): stayin' alive"
        yield from wait_for_click()
        sys.exit(0)

    if choice == 'Lead the charge':
        portrait = sprites.hero_portrait
        portrait_emotion = "anger"
        text = "LET'S ROLL!"
        yield from wait_for_click()


    portrait_emotion = None


    can_control = True

    break_crystal = False
    if state.knights_alive < 5:
        portrait = sprites.lermin_portrait
        text = "Fool."
        yield from wait_for_click()

        text = "Everyone knows,"
        yield from wait_for_click()

        text = "it takes at least 5 knights to defeat a sourcerer."
        yield from wait_for_click()
    else:
        break_crystal = True

    portrait = None
    text = ""

    state.boss.coro = state.boss.drop_bombs()

    yield from spawn_waves()

    state.boss.coro = None

    if break_crystal:
        portrait = sprites.lermin_portrait
        text = "Uh... whoopsie"
        yield from wait_for_click()

        state.boss.vulnerable = True

        text = "it seems my magic crystal broke"
        yield from wait_for_click()

        text = "er..."
        yield from wait_for_click()

        text = "best 2 outta 3?"
        yield from wait_for_click()

        def f():
            global text
            text = "Wait! Don't kill me. All you have to do is leave."
            yield from wait_for_click()

            text = "I'll forget all about this."
            yield from wait_for_click()

            text = "Just walk out of here."
            yield from wait_for_click()

            while True:
                yield

        for x in f():
            if state.boss is None:
                wait(3)
                explode_sprite(state.player.pos * GRID_SIZE)
                state.player.alive = False

                portrait = None
                text = "That was the end of Lermin."
                yield from wait_for_click()

                text = "It was a new start for the region."
                yield from wait_for_click()

                text = "Nearby villages and farms were able to rebuild,"
                yield from wait_for_click()

                text = "without the threat of the necromancer."
                yield from wait_for_click()

                text = "(Imagine a 1-bit picture here of a happy town.)"
                yield from wait_for_click()

                text = "(I ran out of development time :P)"
                yield from wait_for_click()

                text = "The end."
                yield from wait_for_click()

                text = "Game complete (Ending 6):"
                yield from wait_for_click()
                text = "A new beginning for everyone else."
                break
            if int(state.player.pos.y) == GRID_COUNT - 1:
                state.player.alive = False
                state.boss.vulnerable = False
                portrait = sprites.hero_portrait

                text = "Fine."
                yield from wait_for_click()

                text = "At least you can't create anymore undeads."
                yield from wait_for_click()

                text = "I'm outta here."
                yield from wait_for_click()

                portrait = None

                text = "Game complete (Ending 3):"
                yield from wait_for_click()
                text = "Crystal smasher."
                yield from wait_for_click()
                sys.exit(0)

            yield
    else:
        portrait = sprites.lermin_portrait
        text = "All that work for nothing."
        yield from wait_for_click()

        text = "You'll just have to go back to the beginning."
        yield from wait_for_click()

        text = "And do the same thing again."
        yield from wait_for_click()

        text = "MwaHahhAHahHAHAHA"
        yield from wait_for_click()

        portrait = None
        text = "Game over (Ending 2): underpowered"
        yield from wait_for_click()

        sys.exit(0)
    while True:
        yield


def tile_free(state, tile):
    return (
        0 <= tile.x < GRID_COUNT
        and 0 <= tile.y < GRID_COUNT
        and (
            (tile.x, tile.y) not in foreground_layer
            or isinstance(
                foreground_layer[(tile.x, tile.y)], (Spikes, DeadKnight, Planks)
            )
        )
        and tile != state.player.pos
        and tile != state.player.next_pos
        and (state.boss is None or tile != (state.boss.pos / GRID_SIZE).round())
        and state.path
        and all(state.path[k.path_i] != tile for k in state.knights)
        and all((bomb.pos / GRID_SIZE).round() != tile for bomb in state.bombs)
    )


sprites = Sprites()
sprites.load("knight.png", 5, anims={"slice": [0, 2, 3], "dead": [4]})
sprites.load(
    "lermin_portrait.png", 2, anims={"idle": lambda: int((rl.get_time() % 7) > 6.9)}
)
sprites.load(
    "knight_portrait.png", 4, anims={"talk": [0, 1, 2, 3], "idle": [0, 1, 2, 3]}
)
sprites.load(
    "hero_portrait.png",
    5,
    anims={
        "talk": [0, 1],
        "idle": lambda: [0, 2][int((rl.get_time() % 5) > 4.9)],
        "anger": [3],
        "sad": [4],
    },
)
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
sprites.load("bomb.png", nframes=10)
sprites.load("cannon.png", nframes=4)
sprites.load("explosion.png", nframes=5)
sprites.load("spikes.png", nframes=1)
sprites.load("planks.png")
sprites.load("portcullis.png")
sprites.load("wall_variant_1.png")
sprites.load("wall_variant_2.png")

sprites.load(
    "lermin.png",
    nframes=5,
    origin=rl.Vector2(0, -GRID_SIZE),
    anims={"idle": [0, 1, 2, 3], "hit": [4]},
)


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
                and level[n.y][n.x] in "-*^$+p"
                and n not in visited
            ):
                path.append(n)
                visited.add(n)
                cur = n
                break
        steps += 1
        if level[cur.y][cur.x] == "+":
            path.append(V2Pause(cur.x, cur.y))
            break
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


def wait_for_click(y: str = "wait"):
    while not rl.is_mouse_button_released(rl.MOUSE_BUTTON_LEFT):
        yield y
    yield y


def final_screen():
    global state
    state.path.clear()
    foreground_layer.clear()
    state.knights.clear()
    while True:
        yield


input_tween = None

level_coros_gen_fs = [
    intro,
    level_1,
    level_2,
    level_3,
    level_4,
    level_5,
    boss_fight,
]

level_coro = level_coros_gen_fs[0]()
explosions = []
text = ""
portrait = None
portrait_emotion = None


class ContextStr(str):
    def __new__(cls, value, **kwargs):
        obj = str.__new__(cls, value)
        obj.ctx = kwargs
        return obj

    def __getitem__(self, k):
        return self.ctx[k]


C = ContextStr


def hit(pos: V2, d: V2, destroy=False):
    if state.player.pos.floor() == pos:
        state.player.shake_amount = 4

    for bomb in state.bombs:
        if (bomb.pos / GRID_SIZE).floor() == pos:
            bomb.hit(d)

    if state.boss:
        if (state.boss.pos / GRID_SIZE).floor() == pos:
            state.boss.hit(d)

    for i, knight in enumerate(state.knights):
        if (knight.pos / GRID_SIZE).round() == pos:
            state.knights[i].coro = knight_die(state.knights[i])

    p = (pos[0], pos[1])
    if p in foreground_layer:
        if hasattr(foreground_layer[p], "hit"):
            foreground_layer[p].hit(d)
            if destroy:
                explode_sprite(pos * GRID_SIZE, delay=0.1)
                del foreground_layer[p]
        elif isinstance(foreground_layer[p], Planks):
            newpos = (pos + d).round()
            del foreground_layer[p]
            foreground_layer[(newpos.x, newpos.y)] = Planks()


def player_bonk():
    yield from wait(0.1, C("bonk", frame=0))
    yield from wait(0.1, C("bonk", frame=1))
    bonked_tile_pos = V2(
        state.player.pos.x + (-1 if state.player.flip else 1),
        state.player.pos.y,
    )
    hit(bonked_tile_pos, (bonked_tile_pos - state.player.pos).norm())
    yield from wait(0.1, C("bonk", frame=2))


def _explode_coro(pos, shockwave):
    pos = pos.round()
    for i in range(sprites.explosion.frames):
        for _ in wait(0.1):
            if shockwave and i < 1:
                rl.draw_circle_lines(
                    pos.x + GRID_SIZE // 2,
                    pos.y + GRID_SIZE // 2,
                    (GRID_SIZE * 3) / 2,
                    rl.WHITE,
                )
            sprites.explosion.draw_frame(pos, i, False)
            yield


def explode_sprite(pos, delay=0, shockwave=False):
    explosions.append(itertools.chain(wait(delay), _explode_coro(pos, shockwave)))


def step_game(state):
    global input_tween, level_coro, starting_knights_alive, text, portrait, portrait_emotion

    rl.update_music_stream(bgmusic)

    if state.knights_alive > 0 or state.cur_level == 6:
        try:
            next(level_coro)
        except StopIteration:
            state.cur_level += 1
            starting_knights_alive = state.knights_alive
            level_coro = level_coros_gen_fs[state.cur_level]()
            text = ""
            portrait = None
            portrait_emotion = None
    else:
        text = "hit 'R' to restart (keep at least 1 knight alive)"

    if rl.is_key_released(rl.KEY_R):
        level_coro = level_coros_gen_fs[state.cur_level]()
        state.knights_alive = starting_knights_alive
        text = ""
        portrait = None
        portrait_emotion = None


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

    if input_tween is None and can_control:
        if rl.is_key_pressed(rl.KEY_SPACE):
            input_tween = player_bonk()
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

    tiles_to_del = []
    for pos, tile in foreground_layer.items():
        if isinstance(tile, Wall):
            rl.draw_rectangle(
                pos[0] * GRID_SIZE, pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE, rl.WHITE
            )
        elif isinstance(tile, Spikes):
            sprites.spikes.draw(
                V2(pos[0] * GRID_SIZE, pos[1] * GRID_SIZE),
            )
        elif isinstance(tile, Planks):
            sprites.planks.draw(
                V2(pos[0] * GRID_SIZE, pos[1] * GRID_SIZE),
            )
        elif isinstance(tile, BreakWall):
            sprites.breakwall.draw_frame(
                V2(pos[0] * GRID_SIZE, pos[1] * GRID_SIZE),
                int((1 - (tile.health / 10)) * 4),
                False,
            )
            if tile.health <= 0:
                tiles_to_del.append(pos)
        elif isinstance(tile, Cannon):
            tile_state = next(tile.coro)
            assert tile_state == "anim"
            sprites.cannon.draw_frame(
                V2(*pos) * GRID_SIZE,
                tile_state["frame"],
                tile.flipped,
            )
        elif isinstance(tile, StaticSprite):
            flip = hash(pos) % 2
            tile.sprite.draw_frame(V2(*pos) * GRID_SIZE, 0, flip)
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
                    sprites.pillar.anims["lit"][
                        int(((hash(pos) % 7 + rl.get_time()) % 0.3) / 0.1)
                    ],
                    tile.flipped,
                )
        elif isinstance(tile, DeadKnight):
            sprites.knight.draw_frame(
                V2(*pos) * GRID_SIZE,
                sprites.knight.anims["dead"][0],
                False,
            )
        else:
            assert False, type(tile)

    for p in tiles_to_del:
        del foreground_layer[p]

    for p in state.path:
        rl.draw_rectangle(
            p[0] * GRID_SIZE + GRID_SIZE // 2,
            p[1] * GRID_SIZE + GRID_SIZE // 2,
            1,
            1,
            rl.WHITE,
        )

    # draw after all updates
    if state.player.alive:
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
                    sprites.hero.anims["bonk"][input_state["frame"]],
                    state.player.flip,
                )
            else:
                assert False, input_state
        else:
            sprites.hero.draw_frame(
                state.player.pos * GRID_SIZE + shake_vec,
                sprites.hero.anims["idle"][int((rl.get_time() % 5) > 4.9)],
                state.player.flip,
            )

    for bomb_i, bomb in enumerate(state.bombs):
        if bomb.move_coro:
            try:
                next(bomb.move_coro)
            except StopIteration:
                bomb.move_coro = None

        if bomb.anim_coro:
            try:
                s = next(bomb.anim_coro)
                if s == "anim":
                    sprites.bomb.draw_frame(bomb.pos, s["frame"], False)
            except StopIteration:
                del state.bombs[bomb_i]
        else:
            sprites.bomb.draw(bomb.pos)

    for dart_i, dart in enumerate(state.darts):
        dart.pos += dart.vel
        dart_rect = rl.Rectangle(dart.pos.x, dart.pos.y, 5, 1)
        rl.draw_rectangle_rec(dart_rect, rl.WHITE)
        destroy_dart = False
        for knight_i, knight in enumerate(state.knights):
            if rl.check_collision_recs(dart_rect, knight.rect()):
                knight.coro = knight_die(knight)
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
            if e.value == "pass":
                del state.knights[i]
                # state.passed_knights += 1
            if e.value == "die":
                del state.knights[i]
                state.knights_alive -= 1
                foreground_layer[(knight.pos // GRID_SIZE).astuple()] = DeadKnight()
                explode_sprite(knight.pos.floor(), delay=0.1)

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
        elif kstate == "die":
            del state.knights[i]
            state.knights_alive -= 1
            foreground_layer[(knight.pos // GRID_SIZE).astuple()] = DeadKnight()
            explode_sprite(knight.pos.floor(), delay=0.1)
        else:
            assert False

    if state.boss:
        sprites.lermin.draw_frame(
            state.boss.pos, int((rl.get_time() % 0.4) / 0.1), False
        )
        if state.boss.coro:
            next(state.boss.coro)

    for i, explosion in enumerate(explosions):
        try:
            next(explosion)
        except StopIteration:
            del explosions[i]

    text_pos = rl.measure_text_ex(FONT, text, 32, 1)
    text_box_width = int(text_pos.x) + 10
    text_box_height = int(text_pos.y)
    text_top_left = V2(
        SCREEN_WIDTH // 2 - text_box_width // 2,
        SCREEN_HEIGHT - (2 * text_box_height) - 40,
    )

    if portrait:
        top_left = (text_top_left // 2) - V2(36, 36)
        rl.draw_rectangle(
            int(top_left.x) - 1, int(top_left.y) - 1, 32 + 2, 32 + 2, rl.WHITE
        )
        rl.draw_rectangle(int(top_left.x), int(top_left.y), 32, 32, rl.BLACK)
        anim = portrait_emotion or "idle"
        if callable(portrait.anims[anim]):
            frame_i = portrait.anims[anim]()
        else:
            frame_i = portrait.anims[anim][
                int((rl.get_time() % (len(portrait.anims[anim]) * 0.1)) / 0.1)
            ]
        portrait.draw_frame(
            top_left,
            frame_i,
            state.player.flip if portrait == sprites.hero_portrait else False,
        )
    rl.end_mode_2d()

    # draw UI

    if text:
        rl.draw_rectangle(
            *text_top_left - V2(2, 2), text_box_width + 4, text_box_height + 4, rl.WHITE
        )
        rl.draw_rectangle(*text_top_left, text_box_width, text_box_height, rl.BLACK)
        rl.draw_text_ex(
            FONT, text, rl.Vector2(*(text_top_left + V2(4, 2))), 32, 1, rl.WHITE
        )


if __name__ == "__main__":
    rl.set_target_fps(60)
    while not rl.window_should_close():
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)
        step_game(state)
        rl.end_drawing()
    rl.close_window()
