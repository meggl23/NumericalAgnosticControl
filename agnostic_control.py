#!/usr/bin/env python
"""Solve the PDEs for agnostic control with minimum regret

Clancy Rowley
August, 2024
"""

import functools
import multiprocessing
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

# A Field is a variable defined on a grid of size (nq, nx1, nx2)
Field = np.ndarray

@dataclass
class Grid:
    """Parameters for grid for solving PDE"""

    nq: int = 51  # number of grid points in q direction
    nx1: int = 27  # number of points in zeta1 direction
    nx2: int = 26  # number of points in zeta2 direction

    qmax: float = np.pi  # q ranges from -qmax to qmax
    x1max: float = 4.0  # x1 ranges from -x1max to x1max
    x2max: float = 8.0  # x2 is nonnegative and ranges from 0 to x2max

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nq, self.nx1, self.nx2)
    
    @property
    def origin(self) -> tuple[int, int, int]:
        """Return indices corresponding to (q, x1, x2) = (0, 0, 0)"""
        return (self.nq//2, self.nx1//2, 0)

    @property
    def dq(self) -> float:
        return 2 * self.qmax / (self.nq - 1)

    @property
    def dx1(self) -> float:
        return 2 * self.x1max / (self.nx1 - 1)

    @property
    def dx2(self) -> float:
        return self.x2max / (self.nx2 - 1)

    @property
    def q(self) -> np.ndarray:
        return np.linspace(-self.qmax, self.qmax, self.nq)

    @property
    def x1(self) -> np.ndarray:
        return np.linspace(-self.x1max, self.x1max, self.nx1)

    @property
    def x2(self) -> np.ndarray:
        return np.linspace(0, self.x2max, self.nx2)


class Prior:
    """Define a prior probability distribution with support at n values of a"""

    def __init__(self, a: ArrayLike, p: ArrayLike):
        self.a = np.array(a)
        self.p = np.array(p)
        assert len(self.a) == len(self.p)
        # ensure probabilities are positive and sum to one
        self.p = np.abs(self.p)
        self.p /= np.sum(p)

    def __iter__(self):
        for a, p in zip(self.a, self.p):
            yield a, p

    def __len__(self):
        return len(self.a)


def calculate_abar(grid: Grid, prior: Prior, t: float) -> Field:
    """Return values of abar on the grid in (q, x1, x2), for the given t"""
    abar = np.zeros(grid.shape, dtype=float)
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

def optimal_cost(a: float, q: float | np.ndarray, t: float, tmax: float) -> float | np.ndarray:
    """Calculate the optimal cost for known a
    
    q may be a scalar or a numpy array

    The optimal cost J for known a is given by
        J(q, t) = p(t) q^2 + r(t)
    where
        p(T) = r(T) = 0
        -p' = 2 a p + 1 - p^2
        -r' = p
    """
    c = np.sqrt(1 + a**2)
    d = np.arctanh(a / c)
    p = a - c * np.tanh(c * (t - tmax) + d)
    r = a * (tmax - t) + np.log(np.cosh(c * (t - tmax) + d) / np.cosh(d))
    cost = p * q**2 + r
    return cost

def optimal_control(grid: Grid, J: np.ndarray) -> np.ndarray:
    """Calculate the optimal control, given the expected cost J
    
    J is an array of dimension [time, q, x1, x2], as returned by
    expected_cost()
    """
    J_q = np.zeros_like(J)
    J_q[:,1:-1,:,:] = (J[:,2:,:,:] - J[:,:-2,:,:]) / (2 * grid.dq)
    J_q[:,0,:,:] = (J[:,1,:,:] - J[:,0,:,:]) / grid.dq
    J_q[:,-1,:,:] = (J[:,-1,:,:] - J[:,-2,:,:]) / grid.dq

    J_x1 = np.zeros_like(J)
    J_x1[:,:,1:-1,:] = (J[:,:,2:,:] - J[:,:,:-2,:]) / (2 * grid.dx1)
    J_x1[:,:,0,:] = (J[:,:,1,:] - J[:,:,0,:]) / grid.dx1
    J_x1[:,:,-1,:] = (J[:,:,-1,:] - J[:,:,-2,:]) / grid.dx1

    u_opt = -0.5 * (J_q + grid.q[:,None,None] * J_x1)
    return u_opt

def rk4_step(q: Field, t: float, dt: float, rhs: Callable[[Field, float], Field]):
    k1 = rhs(q, t)
    k2 = rhs(q + dt/2 * k1, t + dt/2)
    k3 = rhs(q + dt/2 * k2, t + dt/2)
    k4 = rhs(q + dt * k3, t + dt)
    return q + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

def calculate_derivs(grid: Grid, S: Field) -> tuple[Field, Field, Field, Field]:
    dq = grid.dq
    dx1 = grid.dx1
    dx2 = grid.dx2
    # Finite difference approximations of partial derivatives
    S_q = np.zeros_like(S)
    S_q[1:-1,:,:] = (S[2:,:,:] - S[:-2,:,:]) / (2 * dq)
    S_q[0,:,:] = (S[1,:,:] - S[0,:,:]) / dq
    S_q[-1,:,:] = (S[-1,:,:] - S[-2,:,:]) / dq

    S_qq = np.zeros_like(S)
    S_qq[1:-1,:,:] = (S[2:,:,:] - 2 * S[1:-1,:,:] + S[:-2,:,:]) / (dq**2)
    S_qq[0,:,:] = S_qq[1,:,:]
    S_qq[-1,:,:] = S_qq[-2,:,:]

    S_x1 = np.zeros_like(S)
    S_x1[:,1:-1,:] = (S[:,2:,:] - S[:,:-2,:]) / (2 * dx1)
    S_x1[:,0,:] = (S[:,1,:] - S[:,0,:]) / dx1
    S_x1[:,-1,:] = (S[:,-1,:] - S[:,-2,:]) / dx1

    S_x2 = np.zeros_like(S)
    S_x2[:,:,1:-1] = (S[:,:,2:] - S[:,:,:-2]) / (2 * dx2)
    S_x2[:,:,0] = (S[:,:,1] - S[:,:,0]) / dx2
    S_x2[:,:,-1] = (S[:,:,-1] - S[:,:,-2]) / dx2
    return S_q, S_qq, S_x1, S_x2

def rhs_expected_cost(grid: Grid, prior: Prior, S: Field, t: float) -> Field:
    """Right-hand side of the PDE for minimum expected cost, for a given prior
    """
    S_q, S_qq, S_x1, S_x2 = calculate_derivs(grid, S)
    abar = calculate_abar(grid, prior, t)
    q = grid.q
    q_sq = q**2

    S_t = (-abar * q[:,None,None] * S_q
            - q_sq[:,None,None] * S_x2
            - 0.5 * S_qq
            - q_sq[:,None,None]
            + 0.25 * (S_q + q[:,None,None] * S_x1)**2)
    return S_t

def apply_bc(grid: Grid, a: float, t: float, tmax: float, S: Field) -> None:
    """Set values at the boundaries to optimal cost for known a

    We prescribe Dirichlet boundary conditions at the following locations:
        q_min
        q_max
        x1_min
        x2_max
    """
    q = grid.q
    J0 = optimal_cost(a, q, t, tmax)
    S[0,:,:] = J0[0]            # q_min
    S[-1,:,:] = J0[-1]          # q_max
    S[:,0,:] = J0[:,None]       # x1_min
    S[:,:,-1] = J0[:,None]      # x2_max
    return None

def expected_cost(grid: Grid, prior: Prior, nsteps: int, tmax: float) -> np.ndarray:
    """Integrate the PDE for expected cost backward in time
    
    Starting at time tmax, integrate PDE for optimal cost J, for the given prior
    distribution on the parameter a.
    The PDE is integrated backward in time, starting at t = tmax, saving values
    of the cost J at each step.
    Return the array J[time, q, x1, x2]
    """
    dt = tmax / nsteps
    rhs = functools.partial(rhs_expected_cost, grid, prior)
    shape = (nsteps + 1,) + grid.shape
    J = np.zeros(shape)
    a_max = np.max(prior.a)
    for i in reversed(range(nsteps)):
        t = i * dt
        J[i] = rk4_step(J[i+1], t, -dt, rhs)
        # at boundary values, it is most likely that the value of a is the
        # maximum of the candidates in the prior, so we let J take the value
        # of the cost for that value of a
        apply_bc(grid, a_max, t, tmax, J[i])
        # print(f"{i=}, {np.max(np.abs(J[i]))=}")
    return J

def rhs_cost_known_a(grid: Grid, dt: float, u_opt: np.ndarray, a: float, S: Field, t: float) -> Field:
    """Right-hand side of the PDE for the cost for given a, for the input u_opt"""
    tstep = round(t / dt)
    u = u_opt[tstep]
    q = grid.q
    q_sq = q**2
    S_q, S_qq, S_x1, S_x2 = calculate_derivs(grid, S)
    S_t = (-q_sq[:,None,None] - u**2
            - ((a * q)[:,None,None] + u) * S_q
            - u * q[:,None,None] * S_x1
            - q_sq[:,None,None] * S_x2
            - 0.5 * S_qq)
    return S_t

def cost_known_a(grid: Grid, nsteps: int, tmax: float, u_opt: np.ndarray, a: float) -> Field:
    """Integrate PDE for cost for known a backward in time, returning value at t=0"""
    dt = tmax / nsteps
    rhs = functools.partial(rhs_cost_known_a, grid, dt, u_opt, a)
    J = np.zeros(grid.shape)
    for i in reversed(range(nsteps)):
        t = i * dt
        J = rk4_step(J, t, -dt, rhs)
        # set boundary values to the optimal cost for known a
        apply_bc(grid, a, t, tmax, J)
    return J

def regret_from_cost(grid: Grid, cost: Field, a: float, tmax: float, epsilon: float = 0.02) -> float:
    """Evaluate the given cost at (q, x1, x2) = (0, 0, 0) and return the regret
    
    The (hybrid) regret is given by
        regret = (cost + epsilon) / (opt_cost + epsilon)
    where `cost` is the cost of our strategy, and `opt_cost` is the cost of the
    optimal strategy for known a
    """
    J0 = cost[grid.origin]
    opt_cost = optimal_cost(a, 0, 0, tmax)
    regret = (J0 + epsilon) / (opt_cost + epsilon)
    return regret

def calculate_regret(grid: Grid, tmax: float, u: np.ndarray, a:float) -> float:
    """Calculate the regret given the array of control inputs u, for the given a
    
    First solve the PDE for the cost for known a, and then evaluate the regret
    """
    nsteps = len(u) - 1
    J = cost_known_a(grid, nsteps, tmax, u, a)
    return regret_from_cost(grid, J, a, tmax)

def calculate_regret_vals(grid: Grid,
                          tmax: float,
                          u: np.ndarray,
                          a_vals: ArrayLike,
                          num_procs: int|None = None
                          ) -> np.ndarray:
    """Calculate the regret for the array of values of a in a_vals"""
    regret_fn = functools.partial(calculate_regret, grid, tmax, u)
    regret_vals = []
    with multiprocessing.Pool(processes=num_procs) as p:
        with tqdm(total=len(a_vals)) as pbar:
            for regret in p.imap(regret_fn, a_vals):
                regret_vals.append(regret)
                pbar.update()
    return np.array(regret_vals)

def main():
    grid = Grid()
    prior = Prior([-2, 1], [0.19, 0.81])
    print(f"{len(prior)=}")
    for a, p in prior:
        print(f"  a = {a:5.2f}, p = {p:.2f}")
    print(grid)

    nsteps = 200
    tmax = 1.0
    J_expected = expected_cost(grid, prior, nsteps, tmax)
    u_opt = optimal_control(grid, J_expected)

    # calculate regret for a range of values of `a`
    num_vals = 32
    a_vals = np.linspace(-3, 1, num_vals)
    regret_vals = calculate_regret_vals(grid, tmax, u_opt, a_vals)
    ind = np.argmax(regret_vals)
    print(f"Maximum regret: {regret_vals[ind]:.3f} at a = {a_vals[ind]:.3f}")

if __name__ == "__main__":
    main()
