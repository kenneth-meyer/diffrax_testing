# diffrax_testing

Exploring the diffrax repo! [Diffrax](https://docs.kidger.site/diffrax/) "is a JAX-based library providing numerical differential equation solvers" developed by patrick-kidger, who is a legend. Make sure to check out his work and cite him/the diffrax library when appropriate!

## Main features
- solvers for ODEs, SDEs, and CDEs
- 

## Things to explore
A list of things to explore!

### Using JAX in multi-host and multi-process environments
A guide to this is found [here](https://jax.readthedocs.io/en/latest/multi_process.html). Improving our code to run on multiple GPUs is a great goal/idea. Starting small is a good way to start as well. A library that does MPI for JAX is [mpi4jax](https://github.com/mpi4jax/mpi4jax?tab=readme-ov-file); I don't know much about it but it could be useful!

### diffrax class structure and datatypes used. Goal: integrate with other frameworks
`diffrax.AbstractSolver`, which is detailed [here](https://docs.kidger.site/diffrax/api/solvers/abstract_solvers/). The goal of this is to take the scope of diffrax - which can be used to solve "any numerical method which iterates over timesteps" - and combine it with a finite element method, or some other tool/method that uses a complex discretization of space.

## A brief introduction:

Q: should I be computing this at the points I evaluate at? And do I need to average anything? (look at theory/expected convergence!)

With naive for-loop:
32-bit precision convergence example takes: 9.39s
64-bit precision convergence example takes: 9.64s
