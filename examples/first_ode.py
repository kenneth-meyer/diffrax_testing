"""

This is a copy-paste of the first example of using diffrax.    

"""
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController

import numpy as np

if __name__ == "__main__":

    vector_field = lambda t, y, args: -y**3
    term = ODETerm(vector_field)
    solver = Dopri5()

    ts = np.array([0., 1., 2., 3.])
    y_true = 1 / np.sqrt(2*ts + 1) # analytical solution

    saveat = SaveAt(ts=ts)
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

    sol = diffeqsolve(term, solver, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat,
                    stepsize_controller=stepsize_controller)

    print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
    print(sol.ys)  # DeviceArray([1.   , 0.368, 0.135, 0.0498])
    print(y_true)