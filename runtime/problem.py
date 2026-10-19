"""Data, feasibility, incumbent admission and greedy guard."""
from dataclasses import dataclass
import time
import numpy as np
TOL=1e-8
@dataclass
class Instance:
    values: np.ndarray
    weights: np.ndarray  # items x resources
    capacity: np.ndarray
    name: str = ""
    stratum: str = ""

    def validate(self, selected):
        selected = np.asarray(selected)
        if selected.shape != self.values.shape or not np.all((selected == 0) | (selected == 1)):
            raise ValueError("Solution must be a binary vector with one entry per item")
        load = self.weights.T @ selected.astype(np.float64)
        if np.any(load > self.capacity + TOL):
            raise ValueError(f"Capacity violation: {load} > {self.capacity}")
        return float(self.values @ selected)

class Incumbent:
    """Admit only feasible solutions validated before an absolute deadline."""

    def __init__(self, inst, budget):
        self.inst = inst
        self.started = time.perf_counter()
        self.deadline = self.started + budget
        self.selected = np.zeros(len(inst.values), dtype=bool)
        self.value = 0.0
        self.events = [(0.0, 0.0)]
        self.late = 0

    def offer(self, selected):
        value = self.inst.validate(selected)
        now = time.perf_counter()
        if now > self.deadline:
            self.late += 1
            return False
        if value > self.value + 1e-9:
            self.selected = np.array(selected, dtype=bool, copy=True)
            self.value = value
            self.events.append((now - self.started, value))
        return True

    def result(self):
        elapsed = time.perf_counter() - self.started
        return dict(value=self.value, elapsed_s=elapsed,
                    best_admitted_s=self.events[-1][0], late_offers=self.late,
                    overrun_s=max(0.0, elapsed - (self.deadline - self.started)))

def density(inst, prices=None):
    normalized = inst.weights / inst.capacity
    if prices is None:
        prices = np.ones(len(inst.capacity))
    return inst.values / np.maximum(normalized @ prices, 1e-12)

def greedy(inst, scores=None, initial=None, deadline=float("inf")):
    x = np.zeros(len(inst.values), dtype=bool) if initial is None else np.array(initial, dtype=bool, copy=True)
    inst.validate(x)
    load = inst.weights[x].sum(0)
    scores = density(inst) if scores is None else scores
    for j in np.argsort(-scores, kind="stable"):
        if time.perf_counter() >= deadline:
            break
        if not x[j] and np.all(load + inst.weights[j] <= inst.capacity + TOL):
            x[j] = True
            load += inst.weights[j]
    return x
