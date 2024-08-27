#!/usr/bin/env python3
"""Solve the PDEs for agnostic control with minimum regret"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    """Parameter for grid for solving PDE"""

    nq: int = 51  # number of grid points in q direction
    nx1: int = 27  # number of points in zeta1 direction
    nx2: int = 26  # number of points in zeta2 direction
    nt: int = 200  # number of points in time direction

    qmax: float = np.pi  # q ranges from -qmax to qmax
    x1max: float = 4.0  # x1 ranges from -x1max to x1max
    x2max: float = 8.0  # x2 is nonnegative and ranges from 0 to x2max
    tmax: float = 1.0  # t ranges from 0 to tmax

    @property
    def dt(self):
        return self.tmax / (self.nt - 1)

    @property
    def dq(self):
        return 2 * self.qmax / (self.nq - 1)

    @property
    def dx1(self):
        return 2 * self.x1max / (self.nx1 - 1)

    @property
    def dx2(self):
        return self.x2max / (self.nx2 - 1)

    @property
    def t(self):
        return np.linspace(0, self.tmax, self.nt)

    @property
    def q(self):
        return np.linspace(-self.qmax, self.qmax, self.nq)

    @property
    def x1(self):
        return np.linspace(-self.x1max, self.x1max, self.nx1)

    @property
    def x2(self):
        return np.linspace(0, self.x2max, self.nx2)


class Prior:
    """Define a prior probability distribution with support at n values of a"""

    def __init__(self, a, p):
        self.n = len(a)
        assert len(p) == self.n
        self.a = np.array(a)
        self.p = np.array(p)
        # ensure probabilities sum to one
        self.p /= np.sum(p)

    def __iter__(self):
        for a, p in zip(self.a, self.p):
            yield a, p


def a_bar(prior: Prior, grid: Grid, t: float):
    """Return values of abar on the grid in (q, x1, x2), for the given t"""
    shape = (grid.nq, grid.nx1, grid.nx2)
    abar = np.zeros(shape, dtype=float)
    post_sum = 0.0
    for a, p in prior:
        posterior = p * np.exp(
            a * ((0.5 * (grid.q**2 - t))[:, None, None] - grid.x1[None, :, None])
            - (a**2 / 2 * grid.x2)[None, None, :]
        )
        post_sum += posterior
        abar += posterior * a
    abar /= post_sum
    return abar


def main():
    grid = Grid()
    prior = Prior([-2, 1], [0.81, 0.19])
    for a, p in prior:
        print(f"{a=}, {p=}")
    print(grid)
    print(prior.a, prior.p)
    print(a_bar(prior, grid, t=0.1))


if __name__ == "__main__":
    main()
