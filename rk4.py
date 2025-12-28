import numpy as np

def rk4(f, t_span, y0, n_steps):
    t0, tf = t_span
    ts = np.linspace(t0, tf, n_steps)
    ys = np.zeros(n_steps)
    ys[0] = y0
    dt = ts[1] - ts[0]

    for i in range(1, n_steps):
        t = ts[i-1]
        y = ys[i-1]

        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt, y + dt * k3)

        ys[i] = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return ts, ys
