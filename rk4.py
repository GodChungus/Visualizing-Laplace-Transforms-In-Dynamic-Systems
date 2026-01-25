import numpy as np

def rk4(f, t_span, y0, n_steps):
    """
    Solves a first-order ODE numerically using Runge-Kutta 4.

    Parameters:
        f: function defining the ODE dy/dt = f(t, y)
        t_span: tuple specifying the time interval
        y0: initial condition
        n_steps: number of time steps
    Returns:
        ts: array of time values
        ys: array of solution values
    """
    # Start and End times
    t0, tf = t_span

    ts = np.linspace(t0, tf, n_steps) # Array of evenly spaced values
    ys = np.zeros(n_steps) # Array to store the numerical solution

    ys[0] = y0
    dt = ts[1] - ts[0] #Δt, the step size

    # ======================================================
    # Iteration for the RK4
    # ======================================================
    for i in range(1, n_steps):
        # Current time and solution value
        t = ts[i-1]
        y = ys[i-1]

        # Slope at the beginning of the interval
        k1 = f(t, y)

        # Slope at the midpoint
        k2 = f(t + dt/2, y + dt/2 * k1)

        # Better midpoint slope(using k2 instead of k1)
        k3 = f(t + dt/2, y + dt/2 * k2)

        # Slope at the end of the interval
        k4 = f(t + dt, y + dt * k3)

        # Weighted average of all the slopes
        ys[i] = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
    return ts, ys
