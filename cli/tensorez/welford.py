"""Welford's online algorithm for streaming per-pixel mean and variance.

Straight from
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
— lets us compute the mean *and* variance of an arbitrarily long frame stream
in one pass with two accumulators, never holding the sequence in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Welford:
    count: float = 0.0
    mean: torch.Tensor | None = field(default=None)
    m2: torch.Tensor | None = field(default=None)

    def update(self, value: torch.Tensor) -> None:
        if self.mean is None:
            self.mean = torch.zeros_like(value)
            self.m2 = torch.zeros_like(value)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> torch.Tensor:
        assert self.m2 is not None and self.count > 0
        return self.m2 / self.count

    @property
    def stdev(self) -> torch.Tensor:
        return torch.sqrt(self.variance)
