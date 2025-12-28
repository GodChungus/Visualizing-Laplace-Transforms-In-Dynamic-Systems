import sympy as sp

def solve_ode(ode_expr, y0=0, t_symbol=None):
    if t_symbol is None:
        t = sp.symbols('t')
    else:
        t = t_symbol

    y = sp.Function('y')(t)
    ode = sp.Eq(y.diff(t), ode_expr)
    y_exact_expr = sp.dsolve(ode, y, ics={y.subs(t, 0): y0}).rhs
    y_exact_func = sp.lambdify(t, y_exact_expr, 'numpy')
    f_rhs = sp.lambdify((t, y), ode_expr, 'numpy')

    return f_rhs, y_exact_expr, y_exact_func, t
