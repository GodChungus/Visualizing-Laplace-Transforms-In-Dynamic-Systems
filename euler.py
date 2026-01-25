import numpy as np

def euler(f, t_span, y0, n_steps):
    """
    Solves a first-order ODE numerically by Euler method.

    Parameters:
          f: function defining the ODE dy/dt = f(t, y)
          t_span: time span of the ODE
          y0: initial condition
          n_steps: number of time steps
    Returns:
          ts: array of time values
          ys: array of solution values y(t)
    """
    # Start and end times
    t0, tf = t_span
    
    ts = np.linspace(t0, tf, n_steps) # Array of evenly spaced values
    ys = np.zeros(n_steps) # Array to store the solution values

    ys[0] = y0
    dt = ts[1] - ts[0] # Δt, the step size

    # ======================================================
    # Iteration for the Euler Method
    # ======================================================
    for i in range(1, n_steps):
        ys[i] = ys[i-1] + dt * f(ts[i-1], ys[i-1])
        
    return ts, ys
