"""

This is a quick check of convergence rates vs. step size for different ODE solvers

"""

from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
import diffrax as diffrax

import numpy as np
import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt
import time


def l2_norm(my_vec):
    """ Computes $L^2(\Omega) norm of a quantity

    Computes:
    $$|| u ||_{L^2(\Omega)} = \sqrt{\sum_{i=1}^n u_i^2}$$

    Parameters
    ----------
    my_vec: (CHECK THIS TYPE)

    """

    return jnp.linalg.norm(my_vec)

if __name__ == "__main__":

    jax.config.update("jax_enable_x64", True)

    vector_field = lambda t, y, args: -y**3
    term = ODETerm(vector_field)
    
    solver_list = ["Dopri5", "Dopri8", "Euler", "Heun"]
    dt_list = [1.0, 0.5, 0.1, 0.05, 0.01]
    plot_at = dt_list[2]                                # plot the solution for specific dt

    ts = np.linspace(0,3,16)
    
    # for the specified problem
    y_true = 1 / np.sqrt(2*ts + 1) # analytical solution

    saveat = SaveAt(ts=ts)
    
    # can't use an adaptive step size here - not passing the arg gives a constant step size
    #stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

    start = time.time()

    sol_list = []
    err_list = np.zeros((len(solver_list), len(dt_list)))
    # this looks like something that can be vmapped....look into this and do some performace
    # benchmarking
    solver_i = 0
    dt_j = 0

    # structure for saving things will need to change once this is vmapped
    for solver_name in solver_list:
        for dt in dt_list:
            # better to save the solutions somewhere instead of calling this each time
            # forward eval is cheap right now (have analytical sol) but might not always
            #ts = np.arange(0,3,dt)
            #y_true = 1 / np.sqrt(2*ts + 1) # analytical solution

            # chose which solver to use from the specified list
            solver = getattr(diffrax, solver_name)()

            # solve the ODE
            sol = diffeqsolve(term, solver, t0=0, t1=3, dt0=dt, y0=1, saveat=saveat)

            # append l2 norm of error to an array
            err_list[solver_i][dt_j] = l2_norm(sol.ys - y_true)

            # save solution at desired time step             
            if np.isclose(dt, plot_at):
                sol_list.append(sol)
            dt_j += 1
        dt_j = 0
        solver_i += 1
    end = time.time()

    # plot the results
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(len(solver_list)):
        # ts should be the same because of SaveAt and because adaptive time stepping isnt used
        ax.plot(sol_list[i].ts, sol_list[i].ys,"o-")
    
    ax.plot(ts, y_true)
    ax.legend(solver_list + ["$y_{true}$"])
    
    plt.show()

    # plot the error of the results, wrt time step!
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(len(solver_list)):
        # ts should be the same because of SaveAt and because adaptive time stepping isnt used
        ax.loglog(dt_list, err_list[i],"o-")
    
    ax.legend(solver_list)
    
    plt.show()

    # save important results to a file, which will contain name of github version!!
    # (print for now for testing)
    print("Total time this took: " + str(end - start))

