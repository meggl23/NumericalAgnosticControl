import numpy as np
from numpy.testing import assert_allclose

import agnostic_control as ac


def test_derivs():
    rtol = 1.e-9
    atol = 1.e-11
    grid = ac.Grid(nq=7, nx1=5, nx2=9)
    q_sq = grid.q**2
    x1_sq = grid.x1**2
    x2_sq = grid.x2**2
    S = q_sq[:,None,None] * x1_sq[None,:,None] * x2_sq[None,None,:]
    E_q = 2 * grid.q[:,None,None] * x1_sq[None,:,None] * x2_sq[None,None,:]
    E_qq = 2 * np.ones_like(grid.q)[:,None,None] * x1_sq[None,:,None] * x2_sq[None,None,:]
    E_x1 = 2 * q_sq[:,None,None] * grid.x1[None,:,None] * x2_sq[None,None,:]
    E_x2 = 2 * q_sq[:,None,None] * x1_sq[None,:,None] * grid.x2[None,None,:]
    print(E_q[:,1,1])
    S_q, S_qq, S_x1, S_x2 = ac.calculate_derivs(grid, S)
    print(S_q[:,1,1])
    interior = np.s_[1:-1, 1:-1, 1:-1]
    assert_allclose(S_q[interior], E_q[interior], rtol=rtol, atol=atol)
    assert_allclose(S_qq[interior], E_qq[interior], rtol=rtol, atol=atol)
    assert_allclose(S_x1[interior], E_x1[interior], rtol=rtol, atol=atol)
    assert_allclose(S_x2[interior], E_x2[interior], rtol=rtol, atol=atol)
