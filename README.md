# Numerical Agnostic Control
Numerical treatment of the system described in https://arxiv.org/pdf/2309.10138 and https://arxiv.org/pdf/2309.10142

This GitHub repository provides the code-base to solve the PDE system introduced the [current pre-print](https://arxiv.org/pdf/2309.10138). As part of this code-base we have provided a python code (PDESolver) that both solves the PDE and optimises the bayesian prior using the Newton-raphsom method as well as a demonstrative jupyter-notebook that describes the underlying methodology. 

## Dependencies
The following packages need to be installed to run the code.
```
numpy
h5py
mpi4py
```
While mpi4py is not strictly necessary for the running of the code (and the code can be run in serial), for efficiencies sake we suggest taking advantage of mpi functionality.
