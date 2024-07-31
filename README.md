# Numerical Agnostic Control
Numerical treatment of the system described in https://arxiv.org/pdf/2309.10138 and https://arxiv.org/pdf/2309.10142

This GitHub repository provides the code-base to solve the PDE system introduced the [current pre-print](https://arxiv.org/pdf/2309.10138). As part of this code-base we have provided a python code (PDESolver) that both solves the PDE and optimises the bayesian prior using the Newton-raphsom method as well as a demonstrative jupyter-notebook that describes the underlying methodology. 

# Dependencies
The following packages need to be installed to run the code.
```
numpy
h5py
mpi4py
```
While mpi4py is not strictly necessary for the running of the code (and the code can be run in serial), for efficiencies sake we suggest taking advantage of mpi functionality.

# Example execution
If you run the python file, you can set the parameters of the starting point q0, tmin, tmax and true value a, of the simulation, as well as set an initial bayesian prior (via avec and arho). Then there are three cases to run the code:
 - Case 1: Just a direct solve given a true value of a given a bayesian prior
 - Case 2: Loop over a set of a (currently hard-coded to be from -10 to 1) to evaluate the regret given a bayesian prior
 - Case 3: Use the newton solver to optimise a given bayesian prior.

Then you can run the code as:
```
python PDESolver.py
mpirun -n x python PDESolver.py
```
where x refers to the number of cores you want to utilise.
