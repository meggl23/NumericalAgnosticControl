#!/usr/bin/env python3

'''
Calculate the expected cost incurred by online linear regression during the learning phase and control phase

Amlan Sinha
August 2020
'''

from mpi4py import MPI

import numpy as np
import sys
import h5py
import time
import itertools
import os

comm = MPI.COMM_WORLD

__all__ = ["SingleSolve","CalculateRegret"]

pi = np.pi
np.set_printoptions(precision=17)

################################################################################################################################

class Parameters:
    """
    Class representing a collection of constant system parameters.

    Attributes:
        tmin (float): The minimum time.
        tmax (float): The maximum time.
        avec (float): Vector of the possible candidate a's
        arho (float): Vector of the associated probabilities
        q0 (float):   Initial state of the system
        lam (float):  Weighting of control versus the state in the cost function
        a (float, optional): The true value of the unknown a.
        
    Methods:
        __init__(tmin, tmax, avec, arho, q0, lam=1, a=None):
            Initializes the Parameters instance with the given values.
    """
    def __init__(self,tmin,tmax,avec,arho,q0,lam=1,a=None):

        self.a = a if a is not None else 0.
        
        self.q0   = q0
        self.lam  = lam
        
        self.avec = avec
        self.arho = arho
        
        self.tmin = tmin
        self.tmax = tmax
        
################################################################################################################################
      
class Grid:    
    """
    Contains information about the spatial and temporal grid.

    Attributes:
        num_q (int): Number of points in the q-dimension.
        qmin (float): Minimum value in the q-dimension.
        qmax (float): Maximum value in the q-dimension.
        dq (float): Step size in the q-dimension.
        qvec (np.ndarray): Linearly spaced vector in the q-dimension.
        qarr (np.ndarray): 3D array with q values.
        
        num_x1 (int): Number of points in the x1-dimension.
        x1min (float): Minimum value in the x1-dimension.
        x1max (float): Maximum value in the x1-dimension.
        dx1 (float): Step size in the x1-dimension.
        x1vec (np.ndarray): Linearly spaced vector in the x1-dimension.
        x1arr (np.ndarray): 3D array with x1 values.
        
        num_x2 (int): Number of points in the x2-dimension.
        x2min (float): Minimum value in the x2-dimension.
        x2max (float): Maximum value in the x2-dimension.
        dx2 (float): Step size in the x2-dimension.
        x2vec (np.ndarray): Linearly spaced vector in the x2-dimension.
        x2arr (np.ndarray): 3D array with x2 values.
        
        num_t (int): Number of points in the time dimension.
        tmin (float): Minimum time value.
        tmax (float): Maximum time value.
        dt (float): Step size in the time dimension.
        tvec (np.ndarray): Linearly spaced vector in the time dimension.

    Methods:
        __init__(qpts, qmin, qmax, x1pts, x1min, x1max, x2pts, x2min, x2max, tpts, tmin, tmax):
            Initializes the Grid instance with the given values and computes the necessary grid arrays.
    """

    def __init__(self,qpts,qmin,qmax,x1pts,x1min,x1max,x2pts,x2min,x2max,tpts,tmin,tmax):

        # Grid Parameters
        # q        
        self.num_q = qpts
        self.qmin  = qmin
        self.qmax  = qmax
        self.dq    = (qmax-qmin)/(qpts-1)
        
        self.qvec  = np.linspace(qmin,qmax,qpts)
        
        self.qarr  = np.zeros((qpts,x1pts,x2pts))
        for i in range(qpts):
            self.qarr[i,:,:] = self.qvec[i]

        # x1                        
        self.num_x1 = x1pts
        self.x1min  = x1min
        self.x1max  = x1max
        self.dx1    = (x1max-x1min)/(x1pts-1)
        
        self.x1vec  = np.linspace(x1min,x1max,x1pts)

        self.x1arr  = np.zeros((qpts,x1pts,x2pts))
        for j in range(x1pts):
            self.x1arr[:,j,:] = self.x1vec[j]
            
        # x2
        self.num_x2 = x2pts
        self.x2min  = x2min
        self.x2max  = x2max
        self.dx2    = (x2max-x2min)/(x2pts-1)
        
        self.x2vec  = np.linspace(x2min,x2max,x2pts)
        
        self.x2arr  = np.zeros((qpts,x1pts,x2pts))
        for k in range(x2pts):
            self.x2arr[:,:,k] = self.x2vec[k]
            
        # t
        self.num_t = tpts
        self.tmin  = tmin
        self.tmax  = tmax
        self.dt    = (tmin-tmax)/(tpts-1)
        
        self.tvec  = np.linspace(tmax,tmin,tpts)

################################################################################################################################

class Field:
    """
    Create a scalar field over the 2-dimensional spatial grid.

    Attributes:
        grid (Grid): The spatial and temporal grid.
        params (Parameters): The system parameters.

    Methods:
        __init__(grid, params):
            Initializes the Field instance with the given grid and parameters.
        abar(t):
            Computes the expected value of a given the values of q,xi_1, xi_2 and t.
    """
    def __init__(self,grid,params):
        
        self.grid   = grid
        self.params = params
        
    def abar(self,t):
        """
        Computes the expected value of a given the values of q,xi_1, xi_2 and t.

        Args:
            t (float): The time at which to compute the scalar field.

        Returns:
            np.ndarray: The computed scalar field at time t.
        """

        grid   = self.grid
        params = self.params
        
        avec   = params.avec
        arho   = params.arho
        q0     = params.q0
        
        q      = grid.qarr
        x1     = grid.x1arr
        x2     = grid.x2arr
        
        f      = arho * avec
        g      = 0.5 * (q * q - q0 * q0 - t) - x1
        h      = 0.5 * x2
        
        num    = np.zeros(q.shape)
        den    = np.zeros_like(num)
        
        for i in range(f.shape[0]):
            
            num += f[i]*np.exp(g*avec[i]-h*(avec[i]**2))
            den += arho[i]*np.exp(g*avec[i]-h*(avec[i]**2))

        return num/den #Normalisation step
    
################################################################################################################################
        
class Timestepper:
    """
    Parent class for an explicit and implicit timestepper.

    Attributes:
        dt (float): The time step size.

    Methods:
        __init__(dt):
            Initializes the Timestepper instance with the given time step size.
    """
    def __init__(self, dt):
        
        self.dt = dt

################################################################################################################################

class RK4(Timestepper):
    """
    Four-stage Runge-Kutta scheme for numerical integration of toy-problem 2.
    S is the calculation of the control given the expected value of a (abar)
    Sh is the evaluation of the control calculated from S applied with the true value of the unknown a.
    S0 is the optimal control with perfect knowledge. 
    S0h is the optimal control but taking the average over the candidate values of a. (boundary condition)

    Attributes:
        dt (float): The time step size inherited from Timestepper.

    Methods:
        step(t, S, Sh, RHS, abar, grid, params):
            Performs a single time step update for the fields S and Sh.
        S_Sh_step(t, S, Sh, rhs_Sh_S,  abar, grid, params):
            Computes the RK4 updates for the fields S and Sh.
        S0_step(t, grid, params):
            Computes the RK4 update for the optimal cost S0 with known a
        S0h_step(t, grid, params):
            Computes the RK4 updates for the optimal cost S0h as an weighted average over all the candidate a's (this is for boundary conditions)
        BC_step(t, S, Sh, grid, params):
            Applies boundary conditions to the fields S and Sh.
    """

    def step(self,t,S,Sh,RHS,abar,grid,params):
        
        #S,Sh = self.S_Sh_step(t,S,Sh,RHS.rhs_S,RHS.rhs_Sh,abar,grid,params)
        S,Sh = self.S_Sh_step2(t,S,Sh,RHS.rhs_Sh_S,abar,grid,params)
        S,Sh = self.BC_step(t,S,Sh,grid,params)
        
        return S,Sh

    def S_Sh_step(self,t,S,Sh,rhs_S,rhs_Sh,abar,grid,params):
                
        dt     = self.dt

        field  = Field(grid,params)

        k1 = dt * rhs_Sh(Sh,abar,grid,params)
        l1 = dt * rhs_S(S,Sh,grid,params)
        
        abarTemp = field.abar(t+0.5*dt)

        k2 = dt * rhs_Sh(Sh+0.5*k1,abarTemp,grid,params)
        l2 = dt * rhs_S(S+0.5*l1,Sh+0.5*k1,grid,params)
        
        k3 = dt * rhs_Sh(Sh+0.5*k2,abarTemp,grid,params)
        l3 = dt * rhs_S(S+0.5*l2,Sh+0.5*k2,grid,params)

        abarTemp = field.abar(t+dt)
        
        k4 = dt * rhs_Sh(Sh+k3,abarTemp,grid,params)
        l4 = dt * rhs_S(S+l3,Sh+k3,grid,params)
        
        return S + (l1+2*l2+2*l3+l4)/6. , Sh + (k1+2*k2+2*k3+k4)/6.

    def S_Sh_step2(self,t,S,Sh,rhs_Sh_S,abar,grid,params):

        """
        Computes the RK4 updates for the fields S and Sh.

        Args:
            t (float): Current time.
            S (np.ndarray): Field S.
            Sh (np.ndarray): Field Sh.
            rhs_Sh_S (callable): Function to compute the right-hand side of S,Sh.
            abar (callable): Function to compute abar.
            grid (Grid): Spatial and temporal grid.
            params (Parameters): System parameters.

        Returns:
            tuple: Updated fields S and Sh.
        """

        dt     = self.dt

        field  = Field(grid,params)

        k1,l1 = rhs_Sh_S(S,Sh,abar,grid,params)
        k1 *= dt
        l1 *= dt
        

        abarTemp = field.abar(t+0.5*dt)

        k2,l2 = rhs_Sh_S(S+0.5*l1,Sh+0.5*k1,abarTemp,grid,params)
        k2 *= dt
        l2 *= dt


        k3,l3 = rhs_Sh_S(S+0.5*l2,Sh+0.5*k2,abarTemp,grid,params)
        k3 *= dt
        l3 *= dt

        abarTemp = field.abar(t+dt)

        k4,l4 = rhs_Sh_S(S+l3,Sh+k3,abarTemp,grid,params)
        k4 *= dt
        l4 *= dt
        
        return S + (l1+2*l2+2*l3+l4)/6. , Sh + (k1+2*k2+2*k3+k4)/6.
    
    def S0_step(self,t,grid,params):
        """
        Computes the optimal cost with known a.

        Args:
            t (float): Current time.
            grid (Grid): Spatial and temporal grid.
            params (Parameters): System parameters.

        Returns:
            np.ndarray: Optimal cost field S0.
        """
        a   = params.a
        T   = params.tmax
        lam = params.lam

        # Grid
        q   = grid.qarr
        
        c   = np.sqrt((1.+(a**2)*lam)/lam)
        p   = lam*(a-c*np.tanh(c*(t-T)+np.arctanh(a/c)))
        r   = lam*(-a*(t-T)+np.log((np.cosh(c*(t-T)+np.arctanh(a/c)))/(np.cosh(np.arctanh(a/c)))))
        
        return (q**2)*p+r
    
    def S0h_step(self,t,grid,params):
        """
        Computes the optimal cost with known a over all possible values of a.

        Args:
            t (float): Current time.
            grid (Grid): Spatial and temporal grid.
            params (Parameters): System parameters.

        Returns:
            np.ndarray: Optimal cost field S0h.
        """

        # Extracting parameters
        avec = params.avec
        arho = params.arho
        T    = params.tmax
        lam  = params.lam
        q0   = params.q0
        
        # Grid
        q    = grid.qarr
        x1   = grid.x1arr
        x2   = grid.x2arr
        
        # Temporary arrays
        num  = np.zeros_like(q)
        den  = np.zeros_like(q)
        
        g    = 0.5 * (q * q - q0 * q0 - t) - x1
        h    = 0.5 * x2

        for i in range(avec.shape[0]):
                        
            a = avec[i]
            c = np.sqrt((1.+(a**2)*lam)/lam)
            p = lam*(a-c*np.tanh(c*(t-T)+np.arctanh(a/c)))
            r = lam*(-a*(t-T)+np.log((np.cosh(c*(t-T)+np.arctanh(a/c)))/(np.cosh(np.arctanh(a/c)))))

            num += arho[i]*((q**2)*p+r)*np.exp(g*avec[i]-h*(avec[i]**2))
            den += arho[i]*np.exp(g*avec[i]-h*(avec[i]**2))
                
        return num/den

    def BC_step(self,t,S,Sh,grid,params):
        """
        Applies boundary conditions to the fields S and Sh.

        Args:
            t (float): Current time.
            S (np.ndarray): Field S.
            Sh (np.ndarray): Field Sh.
            grid (Grid): Spatial and temporal grid.
            params (Parameters): System parameters.

        Returns:
            tuple: Fields S and Sh with applied boundary conditions.
        """

        # Optimal cost with known a
        S0 = self.S0_step(t,grid,params)

        # Optimal cost with known a over all possible values of a
        S0h = self.S0h_step(t,grid,params)
        
        # qmin, qmax, x1min, x2max
        S[0,:,:]  = S0[0,:,:]
        S[-1,:,:] = S0[-1,:,:]
        S[:,0,:]  = S0[:,0,:]
        S[:,:,-1] = S0[:,:,-1]
       
        # qmin, qmax, x1min, x2max
        Sh[0,:,:]  = S0h[0,:,:]
        Sh[-1,:,:] = S0h[-1,:,:]
        Sh[:,0,:]  = S0h[:,0,:]
        Sh[:,:,-1] = S0h[:,:,-1]

        return S,Sh

################################################################################################################################

class RHS:
    """
    Computes the right-hand-side for all cost functions.

    Methods:
        rhs_Sh_S(S, abar, grid, params):
            Computes the right-hand side for Sh and S using second order finite difference approximations.
        rhs_Sh(S, abar, grid, params) (redundant):
            Computes the right-hand side for Sh using second order finite difference approximations.
        rhs_S(S, Sh, grid, params) (redundant):
            Computes the right-hand side for S using second order finite difference approximations.
        The latter equations are redundancy here - but it was for the purpose of making sure that everything was 
        clear and proper. They were used for all the results previously seen.
    """

    def rhs_Sh_S(self,S,Sh,abar,grid,params):
        
        # Extracting parameters
        a = params.a
        lam = params.lam
        
        # Grid
        q     = grid.qarr
        dq    = grid.dq
        dx1   = grid.dx1
        dx2   = grid.dx2
        
        # Temporary arrays for approximating derivatives using finite differences
        dSdq   = np.zeros_like(q)
        ddSddq = np.zeros_like(q)
        dSdx1  = np.zeros_like(q)
        dSdx2  = np.zeros_like(q)

        dShdq   = np.zeros_like(q)
        ddShddq = np.zeros_like(q)
        dShdx1  = np.zeros_like(q)
        dShdx2  = np.zeros_like(q)
        
        ustar  = np.zeros_like(q)
        
        dS     = np.zeros_like(q)
        dSh     = np.zeros_like(q)
        
        # Interior points
        dSdq[1:-1,1:-1,1:-1]   = (S[2:,1:-1,1:-1]-S[0:-2,1:-1,1:-1])/(2*dq)
        ddSddq[1:-1,1:-1,1:-1] = (S[2:,1:-1,1:-1]-2*S[1:-1,1:-1,1:-1]+S[0:-2,1:-1,1:-1])/(dq**2)
        dSdx1[1:-1,1:-1,1:-1]  = (S[1:-1,2:,1:-1]-S[1:-1,0:-2,1:-1])/(2*dx1)
        dSdx2[1:-1,1:-1,1:-1]  = (S[1:-1,1:-1,2:]-S[1:-1,1:-1,0:-2])/(2*dx2)

        dShdq[1:-1,1:-1,1:-1]   = (Sh[2:,1:-1,1:-1]-Sh[0:-2,1:-1,1:-1])/(2*dq)
        ddShddq[1:-1,1:-1,1:-1] = (Sh[2:,1:-1,1:-1]-2*Sh[1:-1,1:-1,1:-1]+Sh[0:-2,1:-1,1:-1])/(dq**2)
        dShdx1[1:-1,1:-1,1:-1]  = (Sh[1:-1,2:,1:-1]-Sh[1:-1,0:-2,1:-1])/(2*dx1)
        dShdx2[1:-1,1:-1,1:-1]  = (Sh[1:-1,1:-1,2:]-Sh[1:-1,1:-1,0:-2])/(2*dx2)
        
        # x1max
        dSdq[1:-1,-1,1:-1]     = (S[2:,-1,1:-1]-S[0:-2,-1,1:-1])/(2*dq)
        ddSddq[1:-1,-1,1:-1]   = (S[2:,-1,1:-1]-2*S[1:-1,-1,1:-1]+S[0:-2,-1,1:-1])/(dq**2)
        dSdx1[1:-1,-1,1:-1]    = (S[1:-1,-1,1:-1]-S[1:-1,-2,1:-1])/(dx1)
        dSdx2[1:-1,-1,1:-1]    = (S[1:-1,-1,2:]-S[1:-1,-1,0:-2])/(2*dx2)

        dShdq[1:-1,-1,1:-1]     = (Sh[2:,-1,1:-1]-Sh[0:-2,-1,1:-1])/(2*dq)
        ddShddq[1:-1,-1,1:-1]   = (Sh[2:,-1,1:-1]-2*Sh[1:-1,-1,1:-1]+Sh[0:-2,-1,1:-1])/(dq**2)
        dShdx1[1:-1,-1,1:-1]    = (Sh[1:-1,-1,1:-1]-Sh[1:-1,-2,1:-1])/(dx1)
        dShdx2[1:-1,-1,1:-1]    = (Sh[1:-1,-1,2:]-Sh[1:-1,-1,0:-2])/(2*dx2)
        
        # x2min
        dSdq[1:-1,1:-1,0]      = (S[2:,1:-1,0]-S[0:-2,1:-1,0])/(2*dq)
        ddSddq[1:-1,1:-1,0]    = (S[2:,1:-1,0]-2*S[1:-1,1:-1,0]+S[0:-2,1:-1,0])/(dq**2)
        dSdx1[1:-1,1:-1,0]     = (S[1:-1,2:,0]-S[1:-1,0:-2,0])/(2*dx1)
        dSdx2[1:-1,1:-1,0]     = (S[1:-1,1:-1,1]-S[1:-1,1:-1,0])/(dx2)

        dShdq[1:-1,1:-1,0]      = (Sh[2:,1:-1,0]-Sh[0:-2,1:-1,0])/(2*dq)
        ddShddq[1:-1,1:-1,0]    = (Sh[2:,1:-1,0]-2*Sh[1:-1,1:-1,0]+Sh[0:-2,1:-1,0])/(dq**2)
        dShdx1[1:-1,1:-1,0]     = (Sh[1:-1,2:,0]-Sh[1:-1,0:-2,0])/(2*dx1)
        dShdx2[1:-1,1:-1,0]     = (Sh[1:-1,1:-1,1]-Sh[1:-1,1:-1,0])/(dx2)
        
        # x1max,x2min
        dSdq[1:-1,-1,0]        = (S[2:,-1,0]-S[0:-2,-1,0])/(2*dq)
        ddSddq[1:-1,-1,0]      = (S[2:,-1,0]-2*S[1:-1,-1,0]+S[0:-2,-1,0])/(dq**2)
        dSdx1[1:-1,-1,0]       = (S[1:-1,-1,0]-S[1:-1,-2,0])/(dx1)
        dSdx2[1:-1,-1,0]       = (S[1:-1,-1,1]-S[1:-1,-1,0])/(dx2)

        dShdq[1:-1,-1,0]        = (Sh[2:,-1,0]-Sh[0:-2,-1,0])/(2*dq)
        ddShddq[1:-1,-1,0]      = (Sh[2:,-1,0]-2*Sh[1:-1,-1,0]+Sh[0:-2,-1,0])/(dq**2)
        dShdx1[1:-1,-1,0]       = (Sh[1:-1,-1,0]-Sh[1:-1,-2,0])/(dx1)
        dShdx2[1:-1,-1,0]       = (Sh[1:-1,-1,1]-Sh[1:-1,-1,0])/(dx2)
        
        # Optimal control
        ustar[1:-1,1:,0:-1]    = -(dShdq[1:-1,1:,0:-1]+q[1:-1,1:,0:-1]*dShdx1[1:-1,1:,0:-1])/(2*lam)
        
        # rhs
        dS[1:-1,1:,0:-1]       = -((a*q[1:-1,1:,0:-1]+ustar[1:-1,1:,0:-1])*dSdq[1:-1,1:,0:-1]+
                                    ustar[1:-1,1:,0:-1]*q[1:-1,1:,0:-1]*dSdx1[1:-1,1:,0:-1]+
                                    (q[1:-1,1:,0:-1]**2)*dSdx2[1:-1,1:,0:-1]+0.5*ddSddq[1:-1,1:,0:-1]+
                                    (q[1:-1,1:,0:-1]**2)+lam*(ustar[1:-1,1:,0:-1]**2))

        dSh[1:-1,1:,0:-1]       = -(abar[1:-1,1:,0:-1]*q[1:-1,1:,0:-1]*dShdq[1:-1,1:,0:-1]+
                                    (q[1:-1,1:,0:-1]**2)*dShdx2[1:-1,1:,0:-1]+0.5*ddShddq[1:-1,1:,0:-1]+
                                    (q[1:-1,1:,0:-1]**2)-lam*(ustar[1:-1,1:,0:-1]**2))

        return dSh,dS

    def rhs_Sh(self,Sh,abar,grid,params):
        """
        Computes the right-hand side for Sh using finite difference approximations.

        Args:
            Sh (np.ndarray): Field Sh.
            abar (np.ndarray): Scalar field abar.
            grid (Grid): Spatial and temporal grid.
            params (Parameters): System parameters.

        Returns:
            np.ndarray: The right-hand side for Sh.
        """

        # Extracting parameters
        lam = params.lam
        
        # Grid
        q   = grid.qarr
        dq  = grid.dq
        dx1 = grid.dx1
        dx2 = grid.dx2
        
        # Temporary arrays for approximating derivatives using finite differences
        dShdq   = np.zeros_like(q)
        ddShddq = np.zeros_like(q)
        dShdx1  = np.zeros_like(q)
        dShdx2  = np.zeros_like(q)
        
        ustar  = np.zeros_like(q)
        
        dSh     = np.zeros_like(q)
        
        # Interior points
        dShdq[1:-1,1:-1,1:-1]   = (Sh[2:,1:-1,1:-1]-Sh[0:-2,1:-1,1:-1])/(2*dq)
        ddShddq[1:-1,1:-1,1:-1] = (Sh[2:,1:-1,1:-1]-2*Sh[1:-1,1:-1,1:-1]+Sh[0:-2,1:-1,1:-1])/(dq**2)
        dShdx1[1:-1,1:-1,1:-1]  = (Sh[1:-1,2:,1:-1]-Sh[1:-1,0:-2,1:-1])/(2*dx1)
        dShdx2[1:-1,1:-1,1:-1]  = (Sh[1:-1,1:-1,2:]-Sh[1:-1,1:-1,0:-2])/(2*dx2)
        
        # x1max
        dShdq[1:-1,-1,1:-1]     = (Sh[2:,-1,1:-1]-Sh[0:-2,-1,1:-1])/(2*dq)
        ddShddq[1:-1,-1,1:-1]   = (Sh[2:,-1,1:-1]-2*Sh[1:-1,-1,1:-1]+Sh[0:-2,-1,1:-1])/(dq**2)
        dShdx1[1:-1,-1,1:-1]    = (Sh[1:-1,-1,1:-1]-Sh[1:-1,-2,1:-1])/(dx1)
        dShdx2[1:-1,-1,1:-1]    = (Sh[1:-1,-1,2:]-Sh[1:-1,-1,0:-2])/(2*dx2)

        # x2min
        dShdq[1:-1,1:-1,0]      = (Sh[2:,1:-1,0]-Sh[0:-2,1:-1,0])/(2*dq)
        ddShddq[1:-1,1:-1,0]    = (Sh[2:,1:-1,0]-2*Sh[1:-1,1:-1,0]+Sh[0:-2,1:-1,0])/(dq**2)
        dShdx1[1:-1,1:-1,0]     = (Sh[1:-1,2:,0]-Sh[1:-1,0:-2,0])/(2*dx1)
        dShdx2[1:-1,1:-1,0]     = (Sh[1:-1,1:-1,1]-Sh[1:-1,1:-1,0])/(dx2)

        # x1max,x2min
        dShdq[1:-1,-1,0]        = (Sh[2:,-1,0]-Sh[0:-2,-1,0])/(2*dq)
        ddShddq[1:-1,-1,0]      = (Sh[2:,-1,0]-2*Sh[1:-1,-1,0]+Sh[0:-2,-1,0])/(dq**2)
        dShdx1[1:-1,-1,0]       = (Sh[1:-1,-1,0]-Sh[1:-1,-2,0])/(dx1)
        dShdx2[1:-1,-1,0]       = (Sh[1:-1,-1,1]-Sh[1:-1,-1,0])/(dx2)

        # Optimal control given abar
        ustar[1:-1,1:,0:-1]    = -(dShdq[1:-1,1:,0:-1]+q[1:-1,1:,0:-1]*dShdx1[1:-1,1:,0:-1])/(2*lam)

        # rhs
        dSh[1:-1,1:,0:-1]       = -(abar[1:-1,1:,0:-1]*q[1:-1,1:,0:-1]*dShdq[1:-1,1:,0:-1]+
                                    (q[1:-1,1:,0:-1]**2)*dShdx2[1:-1,1:,0:-1]+0.5*ddShddq[1:-1,1:,0:-1]+
                                    (q[1:-1,1:,0:-1]**2)-lam*(ustar[1:-1,1:,0:-1]**2))

        return dSh
    
    def rhs_S(self,S,Sh,grid,params):
        
        # Extracting parameters
        a = params.a
        lam = params.lam
        
        # Grid
        q     = grid.qarr
        dq    = grid.dq
        dx1   = grid.dx1
        dx2   = grid.dx2
        
        # Temporary arrays for approximating derivatives using finite differences
        dSdq   = np.zeros_like(q)
        ddSddq = np.zeros_like(q)
        dSdx1  = np.zeros_like(q)
        dSdx2  = np.zeros_like(q)
        
        dShdq  = np.zeros_like(q)
        dShdx1 = np.zeros_like(q)
        
        ustar  = np.zeros_like(q)
        
        dS     = np.zeros_like(q)
        
        # Interior points
        dSdq[1:-1,1:-1,1:-1]   = (S[2:,1:-1,1:-1]-S[0:-2,1:-1,1:-1])/(2*dq)
        ddSddq[1:-1,1:-1,1:-1] = (S[2:,1:-1,1:-1]-2*S[1:-1,1:-1,1:-1]+S[0:-2,1:-1,1:-1])/(dq**2)
        dSdx1[1:-1,1:-1,1:-1]  = (S[1:-1,2:,1:-1]-S[1:-1,0:-2,1:-1])/(2*dx1)
        dSdx2[1:-1,1:-1,1:-1]  = (S[1:-1,1:-1,2:]-S[1:-1,1:-1,0:-2])/(2*dx2)

        dShdq[1:-1,1:-1,1:-1]  = (Sh[2:,1:-1,1:-1]-Sh[0:-2,1:-1,1:-1])/(2*dq)
        dShdx1[1:-1,1:-1,1:-1] = (Sh[1:-1,2:,1:-1]-Sh[1:-1,0:-2,1:-1])/(2*dx1)
        
        # x1max
        dSdq[1:-1,-1,1:-1]     = (S[2:,-1,1:-1]-S[0:-2,-1,1:-1])/(2*dq)
        ddSddq[1:-1,-1,1:-1]   = (S[2:,-1,1:-1]-2*S[1:-1,-1,1:-1]+S[0:-2,-1,1:-1])/(dq**2)
        dSdx1[1:-1,-1,1:-1]    = (S[1:-1,-1,1:-1]-S[1:-1,-2,1:-1])/(dx1)
        dSdx2[1:-1,-1,1:-1]    = (S[1:-1,-1,2:]-S[1:-1,-1,0:-2])/(2*dx2)

        dShdq[1:-1,-1,1:-1]    = (Sh[2:,-1,1:-1]-Sh[0:-2,-1,1:-1])/(2*dq)
        dShdx1[1:-1,-1,1:-1]   = (Sh[1:-1,-1,1:-1]-Sh[1:-1,-2,1:-1])/(dx1)
        
        # x2min
        dSdq[1:-1,1:-1,0]      = (S[2:,1:-1,0]-S[0:-2,1:-1,0])/(2*dq)
        ddSddq[1:-1,1:-1,0]    = (S[2:,1:-1,0]-2*S[1:-1,1:-1,0]+S[0:-2,1:-1,0])/(dq**2)
        dSdx1[1:-1,1:-1,0]     = (S[1:-1,2:,0]-S[1:-1,0:-2,0])/(2*dx1)
        dSdx2[1:-1,1:-1,0]     = (S[1:-1,1:-1,1]-S[1:-1,1:-1,0])/(dx2)

        dShdq[1:-1,1:-1,0]     = (Sh[2:,1:-1,0]-Sh[0:-2,1:-1,0])/(2*dq)
        dShdx1[1:-1,1:-1,0]    = (Sh[1:-1,2:,0]-Sh[1:-1,0:-2,0])/(2*dx1)
        
        # x1max,x2min
        dSdq[1:-1,-1,0]        = (S[2:,-1,0]-S[0:-2,-1,0])/(2*dq)
        ddSddq[1:-1,-1,0]      = (S[2:,-1,0]-2*S[1:-1,-1,0]+S[0:-2,-1,0])/(dq**2)
        dSdx1[1:-1,-1,0]       = (S[1:-1,-1,0]-S[1:-1,-2,0])/(dx1)
        dSdx2[1:-1,-1,0]       = (S[1:-1,-1,1]-S[1:-1,-1,0])/(dx2)

        dShdq[1:-1,-1,0]       = (Sh[2:,-1,0]-Sh[0:-2,-1,0])/(2*dq)
        dShdx1[1:-1,-1,0]      = (Sh[1:-1,-1,0]-Sh[1:-1,-2,0])/(dx1)
        
        # Optimal control
        ustar[1:-1,1:,0:-1]    = -(dShdq[1:-1,1:,0:-1]+q[1:-1,1:,0:-1]*dShdx1[1:-1,1:,0:-1])/(2*lam)
        
        # rhs
        dS[1:-1,1:,0:-1]       = -((a*q[1:-1,1:,0:-1]+ustar[1:-1,1:,0:-1])*dSdq[1:-1,1:,0:-1]+ustar[1:-1,1:,0:-1]*q[1:-1,1:,0:-1]*dSdx1[1:-1,1:,0:-1]+(q[1:-1,1:,0:-1]**2)*dSdx2[1:-1,1:,0:-1]+0.5*ddSddq[1:-1,1:,0:-1]+(q[1:-1,1:,0:-1]**2)+lam*(ustar[1:-1,1:,0:-1]**2))

        return dS

################################################################################################################################

def SingleSolve(avec,arho,grid,params,writeflag=0):
    """
    Calculates the cost incurred for a single value of 'a'.

    Args:
        avec (np.ndarray): Array of 'a' values.
        arho (np.ndarray): Array of 'arho' values.
        grid (Grid): The spatial and temporal grid.
        params (Parameters): System parameters.
        writeflag (int, optional): Flag to control writing results to files. Defaults to 0.

    Returns:
        tuple: Final value of fields Sh, S, and S0.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Extracting the components of the time array
    tpts    = grid.num_t
    tvec    = grid.tvec
    dt      = grid.dt

    params.avec = avec
    params.arho = arho
    
    # abar
    field   = Field(grid,params)    
    
    # Initializing the time-stepper
    stepper = RK4(dt)
    
    # Initializing the right hand side
    rhs     = RHS()
    
    # Initializing the cost fields
    S0      = np.zeros_like(grid.qarr)
    Sh      = np.zeros_like(grid.qarr)
    S       = np.zeros_like(grid.qarr)

    S0arr   = np.zeros((grid.num_t,grid.num_q,grid.num_x1,grid.num_x2))    
    Sharr   = np.zeros((grid.num_t,grid.num_q,grid.num_x1,grid.num_x2))
    Sarr    = np.zeros((grid.num_t,grid.num_q,grid.num_x1,grid.num_x2))
    if (writeflag==1):
        
        os.makedirs('./results/Sh', exist_ok=True)
        os.makedirs('./results/S', exist_ok=True)
        os.makedirs('./results/S0', exist_ok=True)

        f = h5py.File('./results/Sh/Sh_00000.h5', 'w')
        f.create_dataset('Sh', data=Sh)
        f.close()
        f = h5py.File('./results/S/S_00000.h5', 'w')
        f.create_dataset('S', data=S)
        f.close()
        f = h5py.File('./results/S0/S0_00000.h5', 'w')
        f.create_dataset('S', data=S)
        f.close()

    for i in range(1,tpts):
        
        ti   = tvec[i]
        
        if(rank==0 and i%200==0): print('i = %f\tt = %f'%(i,ti))
        
        if ((writeflag==1) and (i%200==0)):
            f = h5py.File('./results/Sh/Sh_'+str(i).zfill(5)+'.h5', 'w')
            f.create_dataset('Sh', data=Sh)
            f.close()
            f = h5py.File('./results/S/S_'+str(i).zfill(5)+'.h5', 'w')
            f.create_dataset('S', data=S)
            f.close()
            f = h5py.File('./results/S0/S0_'+str(i).zfill(5)+'.h5', 'w')
            f.create_dataset('S0', data=S0)
            f.close()
        
        sys.stdout.flush()
                
        abar = field.abar(ti)
        
        S0   = stepper.S0_step(ti,grid,params)
        S,Sh = stepper.step(ti,S,Sh,rhs,abar,grid,params)
        
    return Sh,S,S0


################################################################################################################################
class NewtonSolver:


    """
    Implements a Newton-Raphson solver to find the distibution of the initial guess for a.

    Attributes:
        tol (float): The tolerance for convergence.
        eps (float): The perturbation value for numerical derivatives.
        OptMode (str): The mode of optimization ('full', 'probs' (only probabilities), 
        'probsBS' (only probabilities + a bisection for a new value of a1), 
        or a combination of parameters).

    Methods:
        Solver(grid, a_val, a_prob, params):
            Main solver method that dispatches to specific solvers based on the number of points in a_val.

        Solver_4pts(grid, a_val, a_prob, params):
            Optimizes for 4 points to minimize regret.

        Solver_3pts(grid, a_val, a_prob, params):
            Optimizes for 3 points to minimize regret.

        Solver_npts(grid, a_val, a_prob, params):
            Optimizes for an arbitrary number of points to minimize regret.

        GenJacVecs(i_x1, a_vec, a_prob, na, Dict):
            Generates vectors and dictionaries for Jacobian evaluations.

        Jacobian_Evaluations(a_val, a_prob):
            Performs Jacobian evaluations based on the number of points in a_val.

        Jacobian_DoubleDerivs_4pts(true_a, other_a1, other_a2, other_a3, true_prob, other_p1, other_p2, probsign0=1, probsign1=0):
            Computes double derivatives for 4 points.

        Jacobian_DoubleDerivs_3pts(true_a, other_a1, other_a2, true_prob, other_prob, probsign0=1, probsign1=0):
            Computes double derivatives for 3 points.
    """

    def __init__(self,tol,eps,OptMode='full'):
        self.tol = tol
        self.eps = eps
        self.OptMode = OptMode

    def Solver(self,grid,a_val,a_prob,params):
        """
        Main solver method that dispatches to specific solvers based on the number of points in a_val.

        Args:
            grid (Grid): The spatial and temporal grid.
            a_val (np.ndarray): Array of 'a' values.
            a_prob (np.ndarray): Array of probability values.
            params (Parameters): System parameters.

        Returns:
            tuple: final a_val, a_prob, and regret.
        """
        if(len(a_val)==2):
            a_val,a_prob,regret = self.Solver_2pts(grid, a_val, a_prob,params)
        elif(len(a_val)==3):
            a_val,a_prob,regret = self.Solver_3pts(grid, a_val, a_prob,params)
        elif(len(a_val)==4):
            a_val,a_prob,regret = self.Solver_4pts(grid, a_val, a_prob,params)
        else:
            a_val,a_prob,regret = self.Solver_npts(grid, a_val, a_prob,params)
        return a_val,a_prob,regret

    def Solver_4pts(self,grid,a_val,a_prob,params):

        """
        Optimizes for 4 a's and corresponding p that minimizes regret
        for all a
        
        Args:
            grid (Grid): The spatial and temporal grid.
            a_val (np.ndarray): Array of 4 'a' values.
            a_prob (np.ndarray): Array of 4 probability values.
            params (Parameters): System parameters.

        Returns:
            tuple: Updated a_val, a_prob, and regret.
        """


        rank = comm.Get_rank()
        size = comm.Get_size()
        
        tol = self.tol
        eps = self.eps

        Direction = np.ones(7)
        regretvec = np.zeros(64)

        # The OptMode means that we only evaluate the parts of the jacobian that really need to be evaluated. This means that we are more efficient. 
        # The way this works is that we generate a set of indices that correspond to the entries in the jacobian. If these are present in the test_indx vector then we calculate 
        # that particular entry (with the right perturbation so that we get the derivative).

        if(self.OptMode=='full'):

            if rank==0: print('We are optimising all vars')
            test_indx = range(0,64)

        elif(self.OptMode=='probs'):

            if rank==0: print('We are optimising the probabilities')
            test_indx = [0,8,10,12,20,22,24,32,34]

        elif(self.OptMode=='probsBS'):
            # In this case, we optimise the probabilities and then shift the point. This is done, because we note that most of the points are very stable and only really one point shifts very much.
            a_left = -5
            a_right = a_val[0]
            if rank==0: print('We are optimising the probabilities + a Bisection for a1')
            test_indx = [0,8,10,12,20,22,24,32,34,13]

        else:        
            # Here we can pick which of the things we want to optimise - as each of the points refers to a dictionary which maps the jacobian entries to the optimisation of those two parameters.
            if rank==0: print('We are optimising:',self.OptMode)
            comb = list(itertools.product(self.OptMode, self.OptMode))            
            Dict_indx = {'a0' : {'a0' : [2,1,3,0], 'a1' : [21,17,20,16], 'a2' : [37,33,36,32], 'a3' : [53,49,52,48], 'p0' : [3,0,20,16], 'p1' : [3,0,36,32], 'p2' : [3,0,52,48]},
                         'a1' : {'a0' : [5,1,4,0], 'a1' : [18,17,19,16], 'a2' : [39,33,38,32], 'a3' : [55,49,54,48], 'p0' : [4,0,19,16], 'p1' : [4,0,38,32], 'p2' : [4,0,54,48]},
                         'a2' : {'a0' : [7,1,6,0], 'a1' : [23,17,22,16], 'a2' : [34,33,35,32], 'a3' : [57,49,56,48], 'p0' : [6,0,22,16], 'p1' : [6,0,35,32], 'p2' : [6,0,56,48]},
                         'a3' : {'a0' : [9,1,8,0], 'a1' : [25,17,24,16], 'a2' : [41,33,40,32], 'a3' : [50,49,51,48], 'p0' : [8,0,24,16], 'p1' : [8,0,40,32], 'p2' : [8,0,51,48]},
                         'p0' : {'a0' :[11,1,10,0],'a1' : [27,17,26,16], 'a2' : [43,33,42,32], 'a3' : [59,49,58,48], 'p0' : [10,0,26,16],'p1' : [10,0,42,32],'p2' : [10,0,58,48]},
                         'p1' : {'a0' :[13,1,12,0],'a1' : [29,17,28,16], 'a2' : [45,33,44,32], 'a3' : [61,49,60,48], 'p0' : [12,0,28,16],'p1' : [12,0,44,32],'p2' : [12,0,60,48]},
                         'p2' : {'a0' :[15,1,14,0],'a1' : [31,17,30,16], 'a2' : [47,33,46,32], 'a3' : [63,48,62,48], 'p0' : [14,0,30,16],'p1' : [14,0,46,32],'p2' : [14,0,62,48]},
                         }
            test_indx = []
            Jac_Dict = {'a0' : 0, 'a1' : 1, 'a2' : 2,'a3' : 3, 'p0' : 4, 'p1' : 5,'p2': 6}
            for i in range(len(comb)):
                test_indx.extend(Dict_indx[comb[i][0]][comb[i][1]])
            test_indx = list(set(test_indx)) 

        if rank==0: print('These are the indices we will evaluate:',test_indx)

        while((sum(abs(Direction)))>tol):

            a0 = a_val[0]
            a1 = a_val[1]
            a2 = a_val[2]
            a3 = a_val[3]

            p0 = a_prob[0]
            p1 = a_prob[1]
            p2 = a_prob[2]

            a_test = self.Jacobian_Evaluations(a_val,a_prob)

            i=0
            while((i+1)*size<=len(test_indx)):
                regretvec_temp = np.zeros(size)
                indx = test_indx[rank+i*size]
                params.a = a_test[indx,0]

                a_temp = a_test[indx,1:-3]
                params.avec = a_test[indx,1:-3]
                
                ap_temp = np.array([a_test[indx,-3],a_test[indx,-2],a_test[indx,-1],1-a_test[indx,-1]-a_test[indx,-2]-a_test[indx,-3]])
                params.arho = np.array([a_test[indx,-3],a_test[indx,-2],a_test[indx,-1],1-a_test[indx,-1]-a_test[indx,-2]-a_test[indx,-3]])
                
                Sh,S,S0 = SingleSolve(a_temp,ap_temp,grid,params)

                regret    = CalculateRegret(S,S0,grid)
                
                comm.Allgather(regret,regretvec_temp)

                for j in range(size):
                    regretvec[test_indx[j+i*size]] = regretvec_temp[j] 
                i=i+1

            if rank==0: print(regretvec)

            F1 = (regretvec[1]-regretvec[0])/(eps)

            F2 = (regretvec[17]-regretvec[16])/(eps)

            F3 = (regretvec[33]-regretvec[32])/(eps)

            F4 = (regretvec[49]-regretvec[48])/(eps)

            F5 = regretvec[0]-regretvec[16]

            F6 = regretvec[0]-regretvec[32]

            F7 = regretvec[0]-regretvec[48]


            Func = np.array([F1,F2,F3,F4,F5,F6,F7])

            Jac = np.zeros([7,7])

            #All f1 derivatives

            #d2R0dada0
            Jac[0,0] = ((regretvec[2]-regretvec[1])/(eps) - (regretvec[3]-regretvec[0])/(eps))/eps #

            #d2R0dada1
            Jac[0,1] = ((regretvec[5]-regretvec[1])/(eps) - (regretvec[4]-regretvec[0])/(eps))/eps #

            #d2R0dada2
            Jac[0,2] = ((regretvec[7]-regretvec[1])/(eps) - (regretvec[6]-regretvec[0])/(eps))/eps #

            #d2R0dada3
            Jac[0,3] = ((regretvec[9]-regretvec[1])/(eps) - (regretvec[8]-regretvec[0])/(eps))/eps #

            #d2R0dadp0
            Jac[0,4] = ((regretvec[11]-regretvec[1])/(eps) - (regretvec[10]-regretvec[0])/(eps))/eps #

            #d2R0dadp1
            Jac[0,5] = ((regretvec[13]-regretvec[1])/(eps) - (regretvec[12]-regretvec[0])/(eps))/eps #

            #d2R0dadp2
            Jac[0,6] = ((regretvec[15]-regretvec[1])/(eps) - (regretvec[14]-regretvec[0])/(eps))/eps #


            #All f2 derivatives
            #d2R1dada0
            Jac[1,0] = ((regretvec[21]-regretvec[17])/(eps) - (regretvec[20]-regretvec[16])/(eps))/eps #

            #d2R1dada1
            Jac[1,1] = ((regretvec[18]-regretvec[17])/(eps) - (regretvec[19]-regretvec[16])/(eps))/eps #

            #d2R1dada2
            Jac[1,2] = ((regretvec[23]-regretvec[17])/(eps) - (regretvec[22]-regretvec[16])/(eps))/eps #

            #d2R1dada2
            Jac[1,3] = ((regretvec[25]-regretvec[17])/(eps) - (regretvec[24]-regretvec[16])/(eps))/eps #

            #d2R1dadp0
            Jac[1,4] = ((regretvec[29]-regretvec[17])/(eps) - (regretvec[28]-regretvec[16])/(eps))/eps #

            #d2R1dadp1
            Jac[1,5] = ((regretvec[27]-regretvec[17])/(eps) - (regretvec[26]-regretvec[16])/(eps))/eps #

            #d2R1dadp2
            Jac[1,6] = ((regretvec[31]-regretvec[17])/(eps) - (regretvec[30]-regretvec[16])/(eps))/eps #

            #All f3 derivatives
            #d2R2dada0
            Jac[2,0] = ((regretvec[37]-regretvec[33])/(eps) - (regretvec[36]-regretvec[32])/(eps))/eps #

            #d2R2dada1
            Jac[2,1] = ((regretvec[39]-regretvec[33])/(eps) - (regretvec[38]-regretvec[32])/(eps))/eps #

            #d2R2dada2
            Jac[2,2] = ((regretvec[34]-regretvec[33])/(eps) - (regretvec[35]-regretvec[32])/(eps))/eps #

            #d2R2dada3
            Jac[2,3] = ((regretvec[41]-regretvec[33])/(eps) - (regretvec[40]-regretvec[32])/(eps))/eps #

            #d2R2dadp0
            Jac[2,4] = ((regretvec[45]-regretvec[33])/(eps) - (regretvec[44]-regretvec[32])/(eps))/eps #

            #d2R20dadp1
            Jac[2,5] = ((regretvec[47]-regretvec[33])/(eps) - (regretvec[46]-regretvec[32])/(eps))/eps #

            #d2R20dadp2
            Jac[2,6] = ((regretvec[43]-regretvec[33])/(eps) - (regretvec[42]-regretvec[32])/(eps))/eps #


            #All f4 derivatives
            Jac[3,0] = ((regretvec[53]-regretvec[49])/(eps) - (regretvec[52]-regretvec[48])/(eps))/eps #

            #d2R2dada1
            Jac[3,1] = ((regretvec[55]-regretvec[49])/(eps) - (regretvec[54]-regretvec[48])/(eps))/eps #

            #d2R2dada2
            Jac[3,2] = ((regretvec[57]-regretvec[49])/(eps) - (regretvec[56]-regretvec[48])/(eps))/eps #

            #d2R2dada3
            Jac[3,3] = ((regretvec[50]-regretvec[49])/(eps) - (regretvec[51]-regretvec[48])/(eps))/eps #

            #d2R2dadp0
            Jac[3,4] = ((regretvec[61]-regretvec[49])/(eps) - (regretvec[60]-regretvec[48])/(eps))/eps #

            #d2R20dadp1
            Jac[3,5] = ((regretvec[63]-regretvec[49])/(eps) - (regretvec[62]-regretvec[48])/(eps))/eps #

            #d2R20dadp2
            Jac[3,6] = ((regretvec[59]-regretvec[49])/(eps) - (regretvec[58]-regretvec[48])/(eps))/eps #


            #All f4 derivatives
            #d(R0-R1)da0
            Jac[4,0] = ((regretvec[3]-regretvec[0])-(regretvec[20]-regretvec[16]))/eps #

            #d(R0-R1)da1
            Jac[4,1] = ((regretvec[4]-regretvec[0])-(regretvec[19]-regretvec[16]))/eps #

            #d(R0-R1)da2
            Jac[4,2] = ((regretvec[6]-regretvec[0])-(regretvec[22]-regretvec[16]))/eps #

            #d(R0-R1)da3
            Jac[4,3] = ((regretvec[8]-regretvec[0])-(regretvec[24]-regretvec[16]))/eps #

            #d(R0-R1)dp0
            Jac[4,4] = ((regretvec[10]-regretvec[0])-(regretvec[28]-regretvec[16]))/eps # 

            #d(R0-R1)dp1
            Jac[4,5] = ((regretvec[12]-regretvec[0])-(regretvec[26]-regretvec[16]))/eps #

            #d(R0-R1)dp1
            Jac[4,6] = ((regretvec[14]-regretvec[0])-(regretvec[30]-regretvec[16]))/eps #


            #All f5 derivatives
            #d(R0-R2)da0
            Jac[5,0] = ((regretvec[3]-regretvec[0])-(regretvec[36]-regretvec[32]))/eps #

            #d(R0-R1)da1
            Jac[5,1] = ((regretvec[4]-regretvec[0])-(regretvec[38]-regretvec[32]))/eps #

            #d(R0-R1)da2
            Jac[5,2] = ((regretvec[6]-regretvec[0])-(regretvec[35]-regretvec[32]))/eps #

            #d(R0-R1)da2
            Jac[5,3] = ((regretvec[8]-regretvec[0])-(regretvec[40]-regretvec[32]))/eps #

            #d(R0-R1)dp0
            Jac[5,4] = ((regretvec[10]-regretvec[0])-(regretvec[44]-regretvec[32]))/eps #

            #d(R0-R1)dp1
            Jac[5,5] = ((regretvec[12]-regretvec[0])-(regretvec[46]-regretvec[32]))/eps #

            #d(R0-R1)dp1
            Jac[5,6] = ((regretvec[14]-regretvec[0])-(regretvec[42]-regretvec[32]))/eps #


            #All f6 derivatives
            #d(R0-R2)da0
            Jac[6,0] = ((regretvec[3]-regretvec[0])-(regretvec[52]-regretvec[48]))/eps #

            #d(R0-R1)da1
            Jac[6,1] = ((regretvec[4]-regretvec[0])-(regretvec[54]-regretvec[48]))/eps #

            #d(R0-R1)da2
            Jac[6,2] = ((regretvec[6]-regretvec[0])-(regretvec[56]-regretvec[48]))/eps #

            #d(R0-R1)da2
            Jac[6,3] = ((regretvec[8]-regretvec[0])-(regretvec[51]-regretvec[48]))/eps #

            #d(R0-R1)dp0
            Jac[6,4] = ((regretvec[10]-regretvec[0])-(regretvec[60]-regretvec[48]))/eps #

            #d(R0-R1)dp1
            Jac[6,5] = ((regretvec[12]-regretvec[0])-(regretvec[62]-regretvec[48]))/eps #

            #d(R0-R1)dp1
            Jac[6,6] = ((regretvec[14]-regretvec[0])-(regretvec[58]-regretvec[48]))/eps #

            # Working out direction given by : f/df where f = dR/da = 0 at a_1

            # Given that some of the entries of the jacobian may be 0, we delete those that we didnt evaluate 
            # (if we didnt evaluate everything)
            if(self.OptMode=='full'):
                Direction =  np.linalg.inv(Jac) @ Func
            elif(self.OptMode=='probs'):
                Direction[-2:]  = np.linalg.inv(Jac[-2:,-2:]) @ Func[-2:]
            elif(self.OptMode=='probsBS'):
                Direction[-2:]  = np.linalg.inv(Jac[-2:,-2:]) @ Func[-2:]
            else:
                Small_Jac = np.copy(Jac)
                Small_Func = np.copy(Func)
                if 'p2' not in self.OptMode:
                    Small_Jac = np.delete(Small_Jac,6,0)
                    Small_Jac = np.delete(Small_Jac,6,1)
                    Small_Func = np.delete(Small_Func,6,0)
                if 'p1' not in self.OptMode:
                    Small_Jac = np.delete(Small_Jac,5,0)
                    Small_Jac = np.delete(Small_Jac,5,1)
                    Small_Func = np.delete(Small_Func,5,0)
                if 'p0' not in self.OptMode:
                    Small_Jac = np.delete(Small_Jac,4,0)
                    Small_Jac = np.delete(Small_Jac,4,1)
                    Small_Func = np.delete(Small_Func,4,0)
                if 'a3' not in self.OptMode:
                    Small_Jac = np.delete(Small_Jac,3,0)
                    Small_Jac = np.delete(Small_Jac,3,1)
                    Small_Func = np.delete(Small_Func,3,0)
                if 'a2' not in self.OptMode:
                    Small_Jac = np.delete(Small_Jac,2,0)
                    Small_Jac = np.delete(Small_Jac,2,1)
                    Small_Func = np.delete(Small_Func,2,0)
                if 'a1' not in self.OptMode:
                    Small_Jac = np.delete(Small_Jac,1,0)
                    Small_Jac = np.delete(Small_Jac,1,1)
                    Small_Func = np.delete(Small_Func,1,0)
                if 'a0' not in self.OptMode:
                    Small_Jac = np.delete(Small_Jac,0,0)
                    Small_Jac = np.delete(Small_Jac,0,1)
                    Small_Func = np.delete(Small_Func,0,0)

                Small_Direction = np.linalg.inv(Small_Jac) @ Small_Func
                for i in range(len(self.OptMode)):
                    Direction[Jac_Dict[self.OptMode[i]]] = Small_Direction[i]

            if rank==0:
                print(Direction)

            if rank==0: print("before update: a0 =",a0,"a1 =",a1,"a2 =",a2, ", p0 = ",p0,"p1 = ",p1,'p2=',p2)

            a_val[0]   = a0 - Direction[0]
            a_val[1]   = a1 - Direction[1]
            a_val[2]   = a2 - Direction[2]
            a_val[3]   = a3 - Direction[3]
            a_prob[0]  = p0 - Direction[4]
            a_prob[1]  = p1 - Direction[5]
            a_prob[2]  = p2 - Direction[6]
            a_prob[3]  = 1-a_prob[0]-a_prob[1]-a_prob[2]

            if((sum(abs(Direction)))<tol and self.OptMode=='probsBS'):
                print('We found a temporary prior for',a_val,'and', a_prob)
                if(regretvec[17]>regretvec[16]):
                    a_left = a1
                else:
                    a_right = a1
                a_val[0] = (a_left+a_right)/2
                Direction = np.ones(7)
            if rank==0: print("after update: a0 =",a_val[0],"a1 =",a_val[1],"a2 =",a_val[2] ,", p0 = ",a_prob[0],"p1 = ",a_prob[1],'p2=',a_prob[2])

        if rank==0: print("this is value of regret at ap:",regretvec[0])
        if rank==0: print("this is value of regret at am:",regretvec[16])

        return a_val,a_prob,regretvec[0]
    
    
    def Solver_3pts(self,grid,a_val,a_prob,params):
        """
        Optimizes for 3 points to minimize regret.

        Args:
            grid (Grid): The spatial and temporal grid.
            a_val (np.ndarray): Array of 3 'a' values.
            a_prob (np.ndarray): Array of 3 probability values.
            params (Parameters): System parameters.

        Returns:
            tuple: Updated a_val, a_prob, and regret.
        """
        
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        tol = self.tol
        eps = self.eps
        OptMode = self.OptMode
        Direction = np.ones(5)

        regretvec = np.zeros(36)

        if(self.OptMode=='full'):
            if rank==0: print('We are optimising all vars')
            test_indx = range(0,36)
        elif(self.OptMode=='probs'):
            if rank==0: print('We are optimising the probabilities')
            test_indx = [0,8,10,12,20,22,24,32,34]
        elif(self.OptMode=='probsBS'):
            a_left = -5
            a_right = a_val[0]
            if rank==0: print('We are optimising the probabilities + a Bisection for a1')
            test_indx = [0,8,10,12,20,22,24,32,34,13]
        else:        
            if rank==0: print('We are optimising:',self.OptMode)
            comb = list(itertools.product(self.OptMode, self.OptMode))            
            Dict_indx = {'a0' : {'a0' : [2,1,3,0], 'a1' : [17,13,16,12], 'a2' : [29,25,28,24], 'p0' : [3,0,16,12], 'p1' : [3,0,28,24]},
                         'a1' : {'a0' : [5,1,4,0], 'a1' : [14,13,15,12], 'a2' : [31,25,30,24], 'p0' : [4,0,15,12], 'p1' : [4,0,30,24]},
                         'a2' : {'a0' : [7,1,6,0], 'a1' : [19,13,18,12], 'a2' : [26,25,27,24], 'p0' : [6,0,18,12], 'p1' : [6,0,27,24]},
                         'p0' : {'a0' : [9,1,8,0], 'a1' : [23,13,22,12], 'a2' : [35,25,34,24], 'p0' : [8,0,22,12], 'p1' : [8,0,34,24]},
                         'p1' : {'a0' :[11,1,10,0],'a1' : [21,13,20,12], 'a2' : [33,25,32,24], 'p0' : [10,0,20,12],'p1' : [10,0,32,24]}}
            test_indx = []
            Jac_Dict = {'a0' : 0, 'a1' : 1, 'a2' : 2, 'p0' : 3, 'p1' : 4}
            for i in range(len(comb)):
                test_indx.extend(Dict_indx[comb[i][0]][comb[i][1]])
            test_indx = list(set(test_indx)) 

        if rank==0: print('These are the indices we will evaluate:',test_indx)

        while((sum(abs(Direction)))>tol):

            a0 = a_val[0]
            a1 = a_val[1]
            a2 = a_val[2]

            p0 = a_prob[0]
            p1 = a_prob[1]

            a_test = self.Jacobian_Evaluations(a_val,a_prob)

            i=0
                    
            while((i+1)*size<=len(test_indx)):
                regretvec_temp = np.zeros(size)
                indx = test_indx[rank+i*size]
                params.a = a_test[indx,0]

                a_temp = a_test[indx,1:-2]
                params.avec = a_test[indx,1:-2]
                
                ap_temp = np.array([a_test[indx,-2],a_test[indx,-1],1-a_test[indx,-1]-a_test[indx,-2]])
                params.arho = np.array([a_test[indx,-2],a_test[indx,-1],1-a_test[indx,-1]-a_test[indx,-2]])
                
                Sh,S,S0 = SingleSolve(a_temp,ap_temp,grid,params)

                regret    = CalculateRegret(S,S0,grid)
                
                comm.Allgather(regret,regretvec_temp)
                for j in range(size):
                    regretvec[test_indx[j+i*size]] = regretvec_temp[j] 
                i=i+1



            if rank==0: print(regretvec)

            F1 = (regretvec[1]-regretvec[0])/(eps)

            F2 = (regretvec[13]-regretvec[12])/(eps)

            F3 = (regretvec[25]-regretvec[24])/(eps)

            F4 = regretvec[0]-regretvec[12]

            F5 = regretvec[0]-regretvec[24]

            Func = np.array([F1,F2,F3,F4,F5])

            Jac = np.zeros([5,5])

            #All f1 derivatives

            #d2R0dada0
            Jac[0,0] = ((regretvec[2]-regretvec[1])/(eps) - (regretvec[3]-regretvec[0])/(eps))/eps #

            #d2R0dada1
            Jac[0,1] = ((regretvec[5]-regretvec[1])/(eps) - (regretvec[4]-regretvec[0])/(eps))/eps #

            #d2R0dada2
            Jac[0,2] = ((regretvec[7]-regretvec[1])/(eps) - (regretvec[6]-regretvec[0])/(eps))/eps #

            #d2R0dadp0
            Jac[0,3] = ((regretvec[9]-regretvec[1])/(eps) - (regretvec[8]-regretvec[0])/(eps))/eps #

            #d2R0dadp1
            Jac[0,4] = ((regretvec[11]-regretvec[1])/(eps) - (regretvec[10]-regretvec[0])/(eps))/eps #


            #All f2 derivatives
            #Needs checking
            #d2R1dada0
            Jac[1,0] = ((regretvec[17]-regretvec[13])/(eps) - (regretvec[16]-regretvec[12])/(eps))/eps #

            #d2R1dada1
            Jac[1,1] = ((regretvec[14]-regretvec[13])/(eps) - (regretvec[15]-regretvec[12])/(eps))/eps #

            #d2R1dada2
            Jac[1,2] = ((regretvec[19]-regretvec[13])/(eps) - (regretvec[18]-regretvec[12])/(eps))/eps #

            #d2R1dadp0
            Jac[1,3] = ((regretvec[23]-regretvec[13])/(eps) - (regretvec[22]-regretvec[12])/(eps))/eps #

            #d2R1dadp1
            Jac[1,4] = ((regretvec[21]-regretvec[13])/(eps) - (regretvec[20]-regretvec[12])/(eps))/eps #

            #All f3 derivatives
            #Needs checking
            #d2R2dada0
            Jac[2,0] = ((regretvec[29]-regretvec[25])/(eps) - (regretvec[28]-regretvec[24])/(eps))/eps #

            #d2R2dada1
            Jac[2,1] = ((regretvec[31]-regretvec[25])/(eps) - (regretvec[30]-regretvec[24])/(eps))/eps #

            #d2R2dada2
            Jac[2,2] = ((regretvec[26]-regretvec[25])/(eps) - (regretvec[27]-regretvec[24])/(eps))/eps #

            #d2R2dadp0
            Jac[2,3] = ((regretvec[35]-regretvec[25])/(eps) - (regretvec[34]-regretvec[24])/(eps))/eps #

            #d2R20dadp1
            Jac[2,4] = ((regretvec[33]-regretvec[25])/(eps) - (regretvec[32]-regretvec[24])/(eps))/eps #


            #All f4 derivatives

            #d(R0-R1)da0
            Jac[3,0] = ((regretvec[3]-regretvec[0])-(regretvec[16]-regretvec[12]))/eps #

            #d(R0-R1)da1
            Jac[3,1] = ((regretvec[4]-regretvec[0])-(regretvec[15]-regretvec[12]))/eps #

            #d(R0-R1)da2
            Jac[3,2] = ((regretvec[6]-regretvec[0])-(regretvec[18]-regretvec[12]))/eps #

            #d(R0-R1)dp0
            Jac[3,3] = ((regretvec[8]-regretvec[0])-(regretvec[22]-regretvec[12]))/eps # 

            #d(R0-R1)dp1
            Jac[3,4] = ((regretvec[10]-regretvec[0])-(regretvec[20]-regretvec[12]))/eps #


            #All f5 derivatives

            #d(R0-R2)da0
            Jac[4,0] = ((regretvec[3]-regretvec[0])-(regretvec[28]-regretvec[24]))/eps #

            #d(R0-R1)da1
            Jac[4,1] = ((regretvec[4]-regretvec[0])-(regretvec[30]-regretvec[24]))/eps #

            #d(R0-R1)da2
            Jac[4,2] = ((regretvec[6]-regretvec[0])-(regretvec[27]-regretvec[24]))/eps #

            #d(R0-R1)dp0
            Jac[4,3] = ((regretvec[8]-regretvec[0])-(regretvec[34]-regretvec[24]))/eps #

            #d(R0-R1)dp1
            Jac[4,4] = ((regretvec[10]-regretvec[0])-(regretvec[32]-regretvec[24]))/eps #

            # Working out direction given by : f/df where f = dR/da = 0 at a_1
            #Direction = dRda/d2Rdadap
            #Direction =  np.linalg.inv(Jac) @ Func
            Direction = np.zeros(5)
            if(OptMode=='full'):
                Direction =  np.linalg.inv(Jac) @ Func
            elif(OptMode=='probs'):
                Direction[-2:]  = np.linalg.inv(Jac[-2:,-2:]) @ Func[-2:]
            elif(OptMode=='probsBS'):
                Direction[-2:]  = np.linalg.inv(Jac[-2:,-2:]) @ Func[-2:]
            else:
                Small_Jac = np.copy(Jac)
                Small_Func = np.copy(Func)
                if 'p1' not in OptMode:
                    Small_Jac = np.delete(Small_Jac,4,0)
                    Small_Jac = np.delete(Small_Jac,4,1)
                    Small_Func = np.delete(Small_Func,4,0)
                if 'p0' not in OptMode:
                    Small_Jac = np.delete(Small_Jac,3,0)
                    Small_Jac = np.delete(Small_Jac,3,1)
                    Small_Func = np.delete(Small_Func,3,0)
                if 'a2' not in OptMode:
                    Small_Jac = np.delete(Small_Jac,2,0)
                    Small_Jac = np.delete(Small_Jac,2,1)
                    Small_Func = np.delete(Small_Func,2,0)
                if 'a1' not in OptMode:
                    Small_Jac = np.delete(Small_Jac,1,0)
                    Small_Jac = np.delete(Small_Jac,1,1)
                    Small_Func = np.delete(Small_Func,1,0)
                if 'a0' not in OptMode:
                    Small_Jac = np.delete(Small_Jac,0,0)
                    Small_Jac = np.delete(Small_Jac,0,1)
                    Small_Func = np.delete(Small_Func,0,0)

                Small_Direction = np.linalg.inv(Small_Jac) @ Small_Func
                for i in range(len(OptMode)):
                    Direction[Jac_Dict[OptMode[i]]] = Small_Direction[i]

            if rank==0:
                print('Update vector:',Direction)
            if rank==0: print("before update: a0 =",a0,"a1 =",a1,"a2 =",a2, ", p0 = ",p0,"p1 = ",p1)
            a_val[0] = a0 - Direction[0]
            a_val[1] = a1 - Direction[1]
            a_val[2] = a2 - Direction[2]
            a_prob[0]  = np.clip(p0 - Direction[3],0,1)
            a_prob[1]  = np.clip(p1 - Direction[4],0,1)
            a_prob[2]  = 1-a_prob[0]-a_prob[1]
            if((sum(abs(Direction)))<tol and OptMode=='probsBS'):
                print('We found a temporary prior for',a_val,'and', a_prob)
                if(regretvec[13]>regretvec[12]):
                    a_left = a1
                else:
                    a_right = a1
                a_val[0] = (a_left+a_right)/2
                Direction = np.ones(5)
            if rank==0: print("after update: a0 =",a_val[0],"a1 =",a_val[1],"a2 =",a_val[2] ,", p0 = ",a_prob[0],"p1 = ",a_prob[1])

        if rank==0: print("this is value of regret at ap:",regretvec[0])
        if rank==0: print("this is value of regret at am:",regretvec[12])
        return a_val,a_prob,regretvec[0]
    
    def Solver_2pts(self,grid,a_val,a_prob,params):
        
        """
        Optimizes to find the optimal ap and am and the corresponding p that minimizes regret
        for all a
        
        Args:
            grid (Grid): The spatial and temporal grid.
            a_val (np.ndarray): Array of 4 'a' values.
            a_prob (np.ndarray): Array of 4 probability values.
            params (Parameters): System parameters.

        Returns:
            tuple: Updated a_val, a_prob, and regret.
        """

        
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        tol = self.tol
        eps = self.eps
        Direction = np.ones(3)
        regretvec = np.zeros(16)
        while((sum(abs(Direction)))>tol):
            true_a = np.copy(a_val)
            p_crit = a_prob[1]

            regret = []
            selection = [0,4,7,8,9,10,11,14,15] #this is the selection for all points except for a = 1 (don't need to optimise this)
            a_test = self.Jacobian_Evaluations(a_val,a_prob)


            i=0     
            while((i+1)*size<=len(selection)):
                regretvec_temp = np.zeros(size)
                indx = selection[rank+i*size]
                params.a = a_test[indx,0]

                a_temp = a_test[indx,1:-1]
                params.avec = a_test[indx,1:-1]
                
                ap_temp = np.array([a_test[indx,-1],1-a_test[indx,-1]])
                params.arho = np.array([a_test[indx,-1],1-a_test[indx,-1]])
                
                Sh,S,S0 = SingleSolve(a_temp,ap_temp,grid,params)

                regret    = CalculateRegret(S,S0,grid)
                
                comm.Allgather(regret,regretvec_temp)
                for j in range(size):
                    regretvec[selection[j+i*size]] = regretvec_temp[j] 
                i=i+1

            if rank==0: print(regretvec)

            F2 = (regretvec[9]-regretvec[8])/(eps)

            F3 = regretvec[0]-regretvec[8]

            Func = np.array([F2,F3])

            Jac = np.zeros([2,2])

            Jac[0,0] = ((regretvec[10]-regretvec[9])/(eps) - (regretvec[11]-regretvec[8])/(eps))/eps #

            #d2Rmdadp
            Jac[0,1] = ((regretvec[14]-regretvec[9])/(eps) - (regretvec[15]-regretvec[8])/(eps))/eps #



            Jac[1,0] = ((regretvec[4]-regretvec[0])-(regretvec[11]-regretvec[8]))/eps #

            #d(Rp-Rm)dp
            Jac[1,1] = ((regretvec[7]-regretvec[0])-(regretvec[15]-regretvec[8]))/eps #


            # Working out direction given by : f/df where f = dR/da = 0 at a_1
            #Direction = dRda/d2Rdadap
            Direction =  np.linalg.inv(Jac) @ Func
            if rank==0:
                print(Direction)
            if rank==0: print("before update: ap =",true_a[1],"am =",true_a[0], ", p = ",p_crit)
            a_val[1] = true_a[1]
            a_val[0] = true_a[0] - Direction[0]
            a_prob[1]  = np.clip(p_crit  - Direction[1],0,0.99) #so that some probability gets assigned to the positive value
            a_prob[0]  = 1-a_prob[1] 
            if rank==0: print("after update: ap =",a_val[1],"am =",a_val[0], ", p = ",a_prob[1])

        return a_val,a_prob,regretvec[0]

    def Solver_npts(self,grid,a_val,a_prob,params):
        
        """
        Optimizes for an arbitrary number of points to minimize regret. We build the jacobian based on the number
        of provided a_values

        Args:
            grid (Grid): The spatial and temporal grid.
            a_val (np.ndarray): Array of 'a' values.
            a_prob (np.ndarray): Array of probability values.
            params (Parameters): System parameters.

        Returns:
            tuple: Updated a_val, a_prob, and regret.
        """

        rank = comm.Get_rank()
        size = comm.Get_size()
        
        na = len(a_val)

        Direction = np.ones(2*na-1)

        tol = self.tol
        eps = self.eps

        while((sum(abs(Direction)))>tol):

            a_arr = np.copy(a_val)
            ap_arr = a_prob[:-1]
            ListDict = []
            VecList = []

            # Here we generated the jacobian dictionary by adding a epsilon in the right direction.
            for i in range(na):
                Dict = {}
                NewtVec, Dict = self.GenJacVecs(i,a_arr,ap_arr,na,Dict)
                ListDict.append(Dict)
                VecList.append(NewtVec)

            VecList = np.array(VecList)
            VecList = VecList.reshape(VecList.shape[0]*VecList.shape[1],VecList.shape[2])
            regret = []

            regretvec = []

            NumLoops = len(VecList)//size
            for i in range(NumLoops+1):
                if(rank+i*size>= len(VecList)):
                    pass
                else:
                    regretvec_temp = np.zeros(size)
                    vals = VecList[rank+i*size,:]

                    params.a = vals[0]
                    a_temp = vals[1:na+1]
                    ap_temp = np.hstack([vals[(na+1):],1-np.sum(vals[na+1:])])

                    params.avec = vals[1:na+1]
                    params.arho = np.hstack([vals[(na+1):],1-np.sum(vals[na+1:])])

                    Sh,S,S0 = SingleSolve(a_temp,ap_temp,grid,params)
                    regret    = CalculateRegret(S,S0,grid)
                    if np.isnan(regret).any(): sys.exit() #Failsafe

                comm.Allgather(regret,regretvec_temp)
                regretvec.append(regretvec_temp)


            regret = np.ravel(regretvec)

            
            Jac = np.zeros([2*na-1,2*na-1])
            Func = np.zeros(2*na-1)
            
            # Calculating the entries in the jacobian
            for i in range(na):
                Func[i] = (regret[ListDict[i]["eps0_0"]]-regret[ListDict[i]["0_0"]])/self.eps
                for j in range(2*na-1):
                    if(i==j):
                        Jac[i][i] = (regret[ListDict[i]["eps0_2_eps"+str(i+1)]]-regret[ListDict[i]["eps0_eps"+str(i+1)]]-regret[ListDict[i]["eps0_0"]]+regret[ListDict[i]["0_0"]])/(self.eps*self.eps)
                    elif(i<na):
                        Jac[i][j] = (regret[ListDict[i]["eps0_eps"+str(j+1)]]-regret[ListDict[i]["0_eps"+str(j+1)]]-regret[ListDict[i]["eps0_0"]]+regret[ListDict[i]["0_0"]])/(self.eps*self.eps)

            for i in range(na+1,2*na):
                Func[i-1] = regret[ListDict[0]["0_0"]]-regret[ListDict[i-na]["0_0"]]
                for j in range(2*na-1):
                    if j==0:
                        Jac[i-1][j] = (regret[ListDict[0]["eps0_eps"+str(1)]]-regret[ListDict[0]["0_0"]]-regret[ListDict[i-na]["0_eps"+str(j+1)]]+regret[ListDict[i-na]["0_0"]] )/(self.eps)
                    elif((i-na)==j):
                        Jac[i-1][j] = -(regret[ListDict[i-na]["eps0_eps"+str(j+1)]]-regret[ListDict[i-na]["0_0"]] - regret[ListDict[0]["0_eps"+str(j+1)]]+regret[ListDict[0]["0_0"]])/(self.eps)
                    else:
                        Jac[i-1][j] = -(regret[ListDict[i-na]["0_eps"+str(j+1)]]-regret[ListDict[i-na]["0_0"]] - regret[ListDict[0]["0_eps"+str(j+1)]]+regret[ListDict[0]["0_0"]])/(self.eps)

            Small_Jac = np.delete(Jac,4,0)
            Small_Jac = np.delete(Small_Jac,4,1)
            Small_Func = np.delete(Func,4,0)
            Direction =  np.linalg.inv(Small_Jac) @ Small_Func
            if rank==0: print("before update: a:",a_arr," p = ",ap_arr)
            a_val[:-1] = a_arr[:-1]- Direction[:na-1]
            a_val[-1]  = 1.0
            a_prob[:-1] = ap_arr - Direction[na-1:]
            a_prob[-1]  = 1-np.sum(a_prob[:-1])
            if rank==0: print("before update: a:",a_val," p = ",a_prob[:-1])  
            if(np.isnan(a_val).any() or np.isnan(a_prob).any()):
                sys.exit()
        return a_val,a_prob,regret[ListDict[0]["0_0"]]

    def GenJacVecs(self,i_x1,a_vec,a_prob,na,Dict):

        """
        Generates vectors and dictionaries for Jacobian evaluations in the case that we have an arbitrary number of candidate a's. 

        Args:
            i_x1 (int): Index for the current 'a' value.
            a_vec (np.ndarray): Array of 'a' values.
            a_prob (np.ndarray): Array of probability values.
            na (int): Number of 'a' values.
            Dict (dict): Dictionary to store indices.

        Returns:
            tuple: NewtVec (first entries are a values, second entries are probs) and Dict for Jacobian evaluations.
        """

        NewtVec = np.array([np.hstack([a_vec[i_x1],np.concatenate([a_vec,a_prob])]),]*(2*(2*na)+1))
        NewtVec[1,0] += self.eps
        Dict["0_0"] = 0+i_x1*(2*(2*na)+1)
        Dict["eps0_0"] = 1+i_x1*(2*(2*na)+1)
        for i in range(1,na):
            NewtVec[2*i,i]   += self.eps
            NewtVec[2*i+1,i] += self.eps
            NewtVec[2*i+1,0] += self.eps
            Dict["0_eps"+str(i)] = 2*i+i_x1*(2*(2*na)+1)
            Dict["eps0_eps"+str(i)] = 2*i+1+i_x1*(2*(2*na)+1)
        for i in range(na,2*na):
            NewtVec[2*i,i]   += self.eps
            NewtVec[2*i+1,i] += self.eps
            NewtVec[2*i+1,0] += self.eps
            Dict["0_eps"+str(i)] = 2*i+i_x1*(2*(2*na)+1)
            Dict["eps0_eps"+str(i)] = 2*i+1+i_x1*(2*(2*na)+1)

        NewtVec[-1,0]   += 2*self.eps
        NewtVec[-1,i_x1+1]+= self.eps
        Dict["eps0_2_eps"+str(i_x1+1)] = 2*(2*na)+i_x1*(2*(2*na)+1)

        return NewtVec,Dict

        
    
    def Jacobian_Evaluations(self,a_val,a_prob):

        """
        Performs Jacobian evaluations based on the number of points in a_val.

        Args:
            a_val (np.ndarray): Array of 'a' values.
            a_prob (np.ndarray): Array of probability values.

        Returns:
            np.ndarray: Evaluated Jacobian values.
        """
        if(len(a_val)==2):
            ap = a_val[1]
            am = a_val[0]
            dF1 = self.Jacobian_DoubleDerivs_2pts(ap,am,a_prob[1])
            dF2 = self.Jacobian_DoubleDerivs_2pts(am,ap,a_prob[0],probsign=-1)

            a_test = np.concatenate([dF1,dF2])

        elif(len(a_val)==3):
            a0 = a_val[0]
            a1 = a_val[1]
            a2 = a_val[2]

            p0 = a_prob[0]
            p1 = a_prob[1]

            dFa0 = self.Jacobian_DoubleDerivs_3pts(a0,a1,a2,p0,p1)
            dFa1 = self.Jacobian_DoubleDerivs_3pts(a1,a0,a2,p1,p0)
            dFa2 = self.Jacobian_DoubleDerivs_3pts(a2,a0,a1,1-p0-p1,p0,-1,-1)

            a_test = np.concatenate([dFa0,dFa1,dFa2])

        elif(len(a_val)==4):
            a0 = a_val[0]
            a1 = a_val[1]
            a2 = a_val[2]
            a3 = a_val[3]

            p0 = a_prob[0]
            p1 = a_prob[1]
            p2 = a_prob[2]

            dFa0 = self.Jacobian_DoubleDerivs_4pts(a0,a1,a2,a3,p0,p1,p2)
            dFa1 = self.Jacobian_DoubleDerivs_4pts(a1,a0,a2,a3,p1,p0,p2)
            dFa2 = self.Jacobian_DoubleDerivs_4pts(a2,a0,a1,a3,p2,p0,p1)
            dFa3 = self.Jacobian_DoubleDerivs_4pts(a3,a0,a1,a2,1-p0-p1-p2,p0,p1,-1,-1)

            a_test = np.concatenate([dFa0,dFa1,dFa2,dFa3])
            
        return a_test

    def Jacobian_DoubleDerivs_4pts(self,true_a,other_a1,other_a2,other_a3,true_prob,other_p1,other_p2,probsign0=1,probsign1=0):
        
        """
        Provides the perturbed points for the the all the derivatives (4 points)

        Args:
            true_a (float): The first a point around which we perturb
            other_a1 (float): The first other 'a' value.
            other_a2 (float): The second other 'a' value.
            other_a3 (float): The third other 'a' value.
            true_prob (float): the probability of the first a
            other_p1 (float): The first other probability value.
            other_p2 (float): The second other probability value.
            probsign0 (int, optional): Sign for the first probability perturbation. Defaults to 1.
            probsign1 (int, optional): Sign for the second probability perturbation. Defaults to 0.

        Returns:
            np.ndarray: Array of perturbed points
        """

        eps = self.eps          # 0                    1                      2                    3
        permuts = np.array([[0,0,0,0,0,0,0,0],  [eps,0,0,0,0,0,0,0],[2*eps,eps,0,0,0,0,0,0],[eps,eps,0,0,0,0,0,0],
                                # 4                    5                        6                   7
                            [0,0,eps,0,0,0,0,0],[eps,0,eps,0,0,0,0,0],[0,0,0,eps,0,0,0,0],[eps,0,0,eps,0,0,0,0],
                               #  8                     9                       10                          11                         
                            [0,0,0,0,eps,0,0,0],[eps,0,0,0,eps,0,0,0],[0,0,0,0,0,probsign0*eps,0,0],[eps,0,0,0,0,probsign0*eps,0,0],
                               # 12                     13                      14              15
                            [0,0,0,0,0,probsign1*eps,eps,0],[eps,0,0,0,0,probsign1*eps,eps,0],[0,0,0,0,0,probsign1*eps,0,eps],[eps,0,0,0,0,probsign1*eps,0,eps]])

        dF = np.zeros([16,8])
        i=0
        a = np.array([true_a,true_a,other_a1,other_a2,other_a3,true_prob,other_p1,other_p2])
        for x in permuts:
            dF[i,:] = a+x
            i+=1

        return dF
    
    def Jacobian_DoubleDerivs_3pts(self,true_a,other_a1,other_a2,true_prob,other_prob,probsign0=1,probsign1=0):
        
        """
        Provides the perturbed points for the the all the derivatives (3 points)

        Args:
            true_a (float): The first a point around which we perturb
            other_a1 (float): The first other 'a' value.
            other_a2 (float): The second other 'a' value.
            true_prob (float): the probability of the first a
            other_prob (float): The other probability value.
            probsign0 (int, optional): Sign for the first probability perturbation. Defaults to 1.
            probsign1 (int, optional): Sign for the second probability perturbation. Defaults to 0.

        Returns:
            np.ndarray: Array of perturbed points
        """

        eps = self.eps
        #                       0,12,24           1,13             2,14               3,15,27            4,16 ,28            
        permuts = np.array([[0,0,0,0,0,0],[eps,0,0,0,0,0],[2*eps,eps,0,0,0,0],[eps,eps,0,0,0,0],[0,0,eps,0,0,0],
                            #      5    ,29               6                7              8,20                            9                        10,22,34                    11,35
                            [eps,0,eps,0,0,0],[0,0,0,eps,0,0],[eps,0,0,eps,0,0],[0,0,0,0,probsign0*eps,0],[eps,0,0,0,probsign0*eps,0],[0,0,0,0,probsign1*eps,eps],[eps,0,0,0,probsign1*eps,eps]])
        dF = np.zeros([12,6])
        i=0
        a = np.array([true_a,true_a,other_a1,other_a2,true_prob,other_prob])
        for x in permuts:
            dF[i,:] = a+x
            i+=1
        
        return dF
      
    def Jacobian_DoubleDerivs_2pts(self,true_a,other_a,true_prob,probsign=1):
        """
        Computes the double derivatives for two points using finite differences.

        This method generates perturbations of the input parameters (`true_a`, `other_a`, and `true_prob`)
        and calculates the corresponding finite difference approximations for the Jacobian matrix. The 
        method returns an array of perturbed values, which can be used to compute the double derivatives.

        Args:
            true_a (float): The first a point around which we perturb
            other_a (float): The other value of the parameter 'a'
            true_prob (float): the probability of the first a
            probsign (int, optional): A multiplier for the perturbation of the probability (default is 1).

        Returns:
            np.ndarray: Array of perturbed points.
                        Each row corresponds to a different perturbation, and each column
                        represents a different parameter (true_a, other_a, true_prob).
        """

        eps = self.eps
        #probsign=probsign*0.1
        #                       0, 8       1,9        2,10            3,11            4,12        5,13        6,14            7,15
        permuts = np.array([[0,0,0,0],[eps,0,0,0],[2*eps,eps,0,0],[eps,eps,0,0],[0,0,eps,0],[eps,0,eps,0],[eps,0,0,probsign*eps],[0,0,0,probsign*eps]])
        dF = np.zeros([8,4])
        i=0
        a = np.array([true_a,true_a,other_a,true_prob])
        for x in permuts:
            dF[i,:] = a+x
            i+=1

        return dF
      


################################################################################################################################

def CalculateRegret(S,S0,grid):
    '''
    
    For given cost functions, calculate the multiplicative regret at the middle of the grid.
    
    '''
    
    iq  = grid.num_q//2
    ix1 = grid.num_x1//2
    ix2 = 0
    
    return S[iq,ix1,ix2]/(S0[iq,ix1,ix2]+0.02)

################################################################################################################################

def EgRun(params,grid,case):
    

    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if (rank==0):
        tic  = time.perf_counter()


    
    if (case==0):
        
        print(params.avec)
        Sh,S,S0 = SingleSolve(params.avec,params.arho,grid,params)
        
        para_regvec = CalculateRegret(S,S0,grid)

        print(para_regvec)
   
    elif (case==1):
        
        natrue      = 32
        atruevec    = np.linspace(-10.,1.,natrue)
        
        nloops      = int(np.ceil(natrue/size))
        
        Shvec       = np.zeros((natrue,))
        Svec        = np.zeros((natrue,))
        regvec      = np.zeros((natrue,))

        para_Shvec  = np.zeros((int(size*nloops),))
        para_Svec   = np.zeros((int(size*nloops),))
        para_regvec = np.zeros((int(size*nloops),))
        avec = np.zeros((int(size*nloops),))
        
        for ii in range(1,(nloops+1)):

            if ((rank+(ii-1)*size)<natrue):
                print('Working on a = ',atruevec[rank+(ii-1)*size])
                params.a = atruevec[rank+(ii-1)*size]
                Sh,S,S0  = SingleSolve(params.avec,params.arho,grid,params)
                regret   = CalculateRegret(S,S0,grid)
            else:
                Sh       = 0.
                S        = 0.
                regret   = 0.  
            iq  = grid.num_q//2
            ix1 = grid.num_x1//2
            ix2 = 0   
            print(f'a = {params.a},regret={regret}')
    else:

        if rank==0: print("Doing a Newton run for T=",tmax)
        OptMode = 'probs'
        newton = NewtonSolver(1e-4,1e-3)#,OptMode)
        avec,arho,regret = newton.Solver(grid,params.avec,params.arho,params)
        if rank==0: print("For am",avec,",p(ap)=",arho,"an optimum was found with regret:",regret)
                
    return 0 
    
if __name__ == '__main__':
    
    # =============================================================================
    #   Cases
    #   Case 0: Just a direct solve given a true value of a
    #   Case 1: Loop over a set of a to get the regret over a range of true a
    #   Case 2: Find the optimum bayesian prior given some set of T max
    #           
    # =============================================================================
    FlagFine = False
    case = 2

    # =============================================================================
    #   Constants     
    # =============================================================================

    q0   = 0.0
    lam  = 1.0

    tmin = 0.
    tmax = 1
    tptsFine = int(tmax*2000)+1
    tptsRough = int(tmax*200)+1

    # example 2 point - try and optimise!
    avec   = np.array([-2                 , 1.                ])
    arho   = np.array([0.81  ,0.19])

    # Example 3 point (will be unstable for too small T as probability could be negative/>1)
    #avec   = np.array([-1.0167245867833061 ,-4.2                 , 1.                ])
    #arho   = np.array([0.8122629503976486  ,0.00177673767050202 ,0.18596031193184936])

    # Example 5 point (will be unstable for too small T as probability could be negative/>1)
    #avec   = np.array([-4.07027029 ,-2.25053166 ,-0.54982791 ,-1.25279826  ,1.        ])
    #arho   = np.array( [7.81472513e-05 ,5.51085578e-02 ,3.82644438e-01 ,3.68597361e-01, 0.19357149594])

        
    # =============================================================================
    #   Parameters     
    # =============================================================================
    params = Parameters(tmin,tmax,avec,arho,q0,lam,a=None)

    if(FlagFine):
        grid = Grid(101,-pi,pi,51,-4.,4.,51,0.,8.,tptsFine,tmin,tmax)
    else:
        grid = Grid(51,-pi,pi,26,-4.,4.,26,0.,8.,tptsRough,tmin,tmax)




    EgRun(params,grid,case)
