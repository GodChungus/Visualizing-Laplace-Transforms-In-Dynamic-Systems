import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from euler import euler
from rk4 import rk4
from laplace import compute_laplace, compute_inverse_laplace
from odesolutions import solve_ode

# defining symbols
t, s = sp.symbols('t s')

# input first order ODE here
y = sp.Function('y')(t)
ode_rhs = sp.sin(3*t) - 2*y  # General ODE
y0 = 0  # initial condition

# solve ODE
f_rhs, y_exact_expr, y_exact_func, t_sym = solve_ode(ode_rhs, y0=y0, t_symbol=t)

# time grid for numerical solutions
t_start, t_end = 0, 5
n_steps = 1000
t_vals = np.linspace(t_start, t_end, n_steps)

# euler method
t_e, y_euler = euler(f_rhs, (t_start, t_end), y0, n_steps)

# RK4 method
t_r, y_rk4 = rk4(f_rhs, (t_start, t_end), y0, n_steps)

# exact solution values
y_exact_vals = y_exact_func(t_vals)

# laplace and inverse laplace
Y_s_expr, Y_s_num = compute_laplace(y_exact_expr, t, s)
y_ilaplace_expr, y_ilaplace_num = compute_inverse_laplace(Y_s_expr, s, t)
y_ilaplace_vals = y_ilaplace_num(t_vals)

s_vals = np.linspace(0, 10, 1000)
Y_s_vals = Y_s_num(s_vals)

plt.figure(figsize=(16, 10))

# euler
plt.subplot(3, 2, 1)
plt.plot(t_e, y_euler, label='Euler', color='orange')
plt.plot(t_vals, y_exact_vals, '--', label='Exact', alpha=0.7)
plt.title('Euler Method')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()

# RK4
plt.subplot(3, 2, 2)
plt.plot(t_r, y_rk4, label='RK4', color='green')
plt.plot(t_vals, y_exact_vals, '--', label='Exact', alpha=0.7)
plt.title('Runge-Kutta 4')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()

# exact solution
plt.subplot(3, 2, 3)
plt.plot(t_vals, y_exact_vals, color='black')
plt.title('Exact Solution')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.grid()

# comparison
plt.subplot(3, 2, 4)
plt.plot(t_e, y_euler, label='Euler', alpha=0.7)
plt.plot(t_r, y_rk4, label='RK4')
plt.plot(t_vals, y_exact_vals, '--', label='Exact')
plt.title('Comparison')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()

# laplace transform
plt.subplot(3, 2, 5)
plt.plot(s_vals, Y_s_vals, color='blue')
plt.title('Laplace Transform $Y(s)$')
plt.xlabel('s')
plt.ylabel('Y(s)')
plt.grid()

# inverse Laplace
plt.subplot(3, 2, 6)
plt.plot(t_vals, y_ilaplace_vals, label='Inverse Laplace', color='purple')
plt.plot(t_vals, y_exact_vals, '--', label='Exact', color='black')
plt.title('Inverse Laplace vs Exact')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
