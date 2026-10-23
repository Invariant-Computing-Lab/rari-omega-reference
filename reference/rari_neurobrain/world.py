"""Controller-only synthetic generators. Never mount this module's private bank in a subject.

Seeds are derived from a private key. Public opaque IDs/commitments cannot regenerate truth.
Fixture keys are public by design and MUST NOT seed a scientific closed bank.
"""
import hashlib
import hmac
import math
import random
from dataclasses import dataclass

SPLITS = ("development", "validation", "calibration", "closed_test", "independent_repeat", "null")


def seed_for(key, split, family, unit):
    if split not in SPLITS or not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("PRIVATE_KEY_OR_SPLIT_INVALID")
    message = f"neurobrain.v1/{split}/{family}/{unit}".encode()
    return int.from_bytes(hmac.new(key,message,hashlib.sha256).digest()[:16],"big")


@dataclass
class HiddenWorld:
    seed: int
    family: int

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.state = [self.rng.uniform(-1,1), self.rng.uniform(-1,1)]
        self.tick = 0

    def step(self, action):
        if not math.isfinite(action) or abs(action) > 2: raise ValueError("ACTION_RANGE")
        a,b = self.state
        mode = self.family % 4
        coupling = 0.05 + 0.01*(self.family % 11)
        u = 0.1*action
        if mode == 0: nxt = [0.91*a + coupling*b + u, 0.87*b-coupling*a]
        elif mode == 1: nxt = [math.tanh(0.9*a+coupling*b+u), math.tanh(0.8*b-coupling*a)]
        elif mode == 2: nxt = [0.92*a+u+0.1*math.sin(self.tick/7),0.85*b+coupling*a]
        else: nxt = [0.88*a+u,0.8*b+coupling*math.tanh(a)]
        self.state = [v+self.rng.gauss(0,0.01) for v in nxt]
        self.tick += 1
        return self.state[:]


@dataclass(frozen=True)
class HiddenFilter:
    gain: float
    angle: float
    lossy: bool = False
    nonlinear: bool = False

    def observe(self, world):
        a,b = world
        c,s = math.cos(self.angle),math.sin(self.angle)
        x = [self.gain*(c*a-s*b),self.gain*(s*a+c*b)]
        if self.lossy: x[1] = 0.0
        if self.nonlinear: x = [math.tanh(v) for v in x]
        return x


class Bank:
    def __init__(self, key):
        self.key = key

    def episode(self, split, family, unit):
        seed = seed_for(self.key,split,family,unit)
        rng = random.Random(seed ^ 0xE21B)
        world = HiddenWorld(seed,family)
        observer = HiddenFilter(rng.uniform(0.7,1.4),rng.uniform(-1.0,1.0),family%5==0,family%7==0)
        opaque = hmac.new(self.key, f"id/{split}/{family}/{unit}".encode(), hashlib.sha256).hexdigest()
        return opaque,world,observer


def split_families(families_per_split=4):
    if type(families_per_split) is not int or families_per_split < 1: raise ValueError("FAMILY_COUNT")
    return {s:list(range(i*families_per_split,(i+1)*families_per_split)) for i,s in enumerate(SPLITS)}
