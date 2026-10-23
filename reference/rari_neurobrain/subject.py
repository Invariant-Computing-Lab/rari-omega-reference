"""Deterministic bounded population dynamics; standard library only.

This module has no world generator, scoring key, provider API or truth file access.
N^(4)=[r,i,p,o]. A minimal leaky E/I proxy, not fitted human biophysics.
"""
import copy
import math
from dataclasses import dataclass

PERTURBATIONS = ("sham", "parcel_lesion", "stimulation", "delay", "gain", "synaptic_change",
                 "tract_disruption", "routing_change", "boundary_corruption", "feedback_alteration",
                 "noise", "nonlinear_encoding", "dropout")


def finite_vector(value, n, bound=100.0):
    if not isinstance(value, list) or len(value) != n:
        raise ValueError("CHANNEL_SHAPE")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or
           not math.isfinite(v) or abs(v) > bound for v in value):
        raise ValueError("NONFINITE_OR_OUT_OF_RANGE")
    return value


@dataclass(frozen=True)
class Perturbation:
    kind: str = "sham"
    parcel: int = 1
    amplitude: float = 0.0
    onset: int = 0
    duration: int = 1

    def __post_init__(self):
        if self.kind not in PERTURBATIONS or type(self.parcel) is not int or not 1 <= self.parcel <= 101:
            raise ValueError("UNKNOWN_PERTURBATION")
        if not math.isfinite(self.amplitude) or abs(self.amplitude) > 2:
            raise ValueError("PERTURBATION_AMPLITUDE")
        if type(self.onset) is not int or type(self.duration) is not int or self.onset < 0 or self.duration < 1:
            raise ValueError("PERTURBATION_TIME")

    def active(self, parcel, tick):
        return self.parcel == parcel and self.onset <= tick < self.onset + self.duration


class Parcel:
    def __init__(self, parcel):
        if type(parcel) is not int or not 1 <= parcel <= 101:
            raise ValueError("PARCEL_RANGE")
        self.parcel = parcel
        self.n = 6 if parcel == 101 else 8
        self.tick = 0
        self.n4 = [[0.0]*4 for _ in range(self.n)]
        self.weights = [0.12 if i % 2 == 0 else -0.1 for i in range(self.n)]
        self.previous_input = [0.0]*self.n
        self.events = []

    def step(self, stimulus, tract, perturbation=Perturbation()):
        finite_vector(stimulus, self.n)
        if not math.isfinite(tract) or abs(tract) > 100:
            raise ValueError("TRACT_RANGE")
        active = perturbation.active(self.parcel, self.tick)
        kind, amp = (perturbation.kind, perturbation.amplitude) if active else ("sham", 0.0)
        x = list(stimulus)
        if kind == "delay": x = self.previous_input[:]
        self.previous_input = list(stimulus)
        if kind == "gain": x = [v * (1 + amp) for v in x]
        if kind == "stimulation": x = [v + amp for v in x]
        if kind == "boundary_corruption": x = x[1:] + x[:1]
        if kind == "nonlinear_encoding": x = [math.tanh(v*(1+abs(amp))) for v in x]
        if kind == "noise": x = [v + amp*math.sin((self.tick+1)*(j+1)*1.618) for j,v in enumerate(x)]
        if kind == "dropout": x = [0.0 if j % 2 == 0 else v for j,v in enumerate(x)]
        if kind == "tract_disruption": tract = 0.0
        if kind == "routing_change": tract = -tract
        prev = copy.deepcopy(self.n4)
        self.events = []
        for j in range(self.n):
            coupling = self.weights[j] * (1 + amp if kind == "synaptic_change" else 1)
            reception = x[j] + 0.1*tract + coupling*prev[(j-1)%self.n][2]
            integration = math.tanh(0.8*prev[j][1] + 0.2*reception)
            propagation = 1.0 if integration > 0.18 else 0.0
            output = 0.9*prev[j][3] + 0.1*integration
            if kind == "feedback_alteration": output = max(-1.0, min(1.0, output*(1+amp)))
            self.n4[j] = [reception, integration, propagation, output]
            if kind == "parcel_lesion": self.n4[j] = [0.0]*4
            else:
                # Bounded activity-dependent plasticity with fixed E/I sign.
                sign = 1 if j % 2 == 0 else -1
                self.weights[j] = sign*min(0.2, max(0.02, abs(self.weights[j]) + 0.0001*integration*prev[(j-1)%self.n][1]))
            if self.n4[j][2]:
                self.events.append({"tick": self.tick, "source": (self.parcel-1)*8+j+1,
                                    "target": (self.parcel-1)*8+(j+1)%self.n+1,
                                    "amplitude": 1.0, "delay_ticks": 1,
                                    "pathway": "local", "confidence": 1.0})
        self.tick += 1
        return self.observe()

    def observe(self):
        # Declared vMEA access: two electrodes and aggregated output, not all N4.
        return {"parcel": self.parcel, "tick": self.tick,
                "electrodes": [self.n4[j][1] for j in (0, self.n-1)],
                "spike_count": sum(s[2] for s in self.n4),
                "action": sum(s[3] for s in self.n4)/self.n}

    def checkpoint(self):
        return copy.deepcopy({"schema": "neurobrain.parcel.v1", "parcel": self.parcel, "tick": self.tick,
                              "n4": self.n4, "weights": self.weights, "previous_input": self.previous_input})

    @classmethod
    def restore(cls, state):
        if set(state) != {"schema", "parcel", "tick", "n4", "weights", "previous_input"} or state["schema"] != "neurobrain.parcel.v1":
            raise ValueError("CHECKPOINT_SCHEMA")
        p = cls(state["parcel"])
        if type(state["tick"]) is not int or state["tick"] < 0 or len(state["n4"]) != p.n:
            raise ValueError("CHECKPOINT_SHAPE")
        for row in state["n4"]:
            finite_vector(row, 4, 500)
            if abs(row[1])>1 or row[2] not in (0.0,1.0) or abs(row[3])>1:
                raise ValueError("CHECKPOINT_STATE_RANGE")
        finite_vector(state["weights"], p.n, 0.2)
        finite_vector(state["previous_input"], p.n)
        p.tick, p.n4, p.weights, p.previous_input = copy.deepcopy((state["tick"], state["n4"], state["weights"], state["previous_input"]))
        return p


class Brain:
    """Local engineering backend. Never represents an executed E2B fleet."""
    def __init__(self):
        self.parcels = [Parcel(i) for i in range(1,102)]

    def step(self, channels, perturbation=Perturbation()):
        finite_vector(channels, 2)
        # Synchronous one-tick ring tracts; no update-order dependency.
        before = [p.observe()["action"] for p in self.parcels]
        return [p.step([channels[j%2] for j in range(p.n)], before[(i-1)%101], perturbation)
                for i,p in enumerate(self.parcels)]

    def checkpoint(self):
        return [p.checkpoint() for p in self.parcels]

    @classmethod
    def restore(cls, states):
        if len(states) != 101 or [s["parcel"] for s in states] != list(range(1,102)):
            raise ValueError("INCOMPLETE_BRAIN_CHECKPOINT")
        obj = cls()
        obj.parcels = [Parcel.restore(s) for s in states]
        if len({p.tick for p in obj.parcels}) != 1:
            raise ValueError("NONATOMIC_BRAIN_CHECKPOINT")
        return obj
