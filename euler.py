import numpy as np

def euler(f, t_span, y0, n_steps):
    t0, tf = t_span
    ts = np.linspace(t0, tf, n_steps)
    ys = np.zeros(n_steps)
    ys[0] = y0
    dt = ts[1] - ts[0]

    for i in range(1, n_steps):
        ys[i] = ys[i-1] + dt * f(ts[i-1], ys[i-1])
    return ts, ys
