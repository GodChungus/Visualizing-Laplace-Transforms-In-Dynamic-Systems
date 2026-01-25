import sympy as sp

def solve_ode(ode_expr, y0=0, t_symbol=None):
    """
    Solves a first-order ODE.

    Parameters:
        ode_expr: sympy expression defining the RHS expression
        y0: initial condition
        t_symbol: sympy expression defining the time symbol

    Returns:
        f_rhs: numerical function of ode_expr
        y_exact_expr: exact solution of ode_expr
        y_exact_func: numerical version of exact solution
        t: sympy time symbol
    """
    if t_symbol is None:
        # Make a new time symbol if it doesn't exist already
        t = sp.symbols('t')
    else:
        t = t_symbol

    # Defining y(t)
    y = sp.Function('y')(t)

    # Constructing the ODE
    ode = sp.Eq(y.diff(t), ode_expr)

    # Solving the ODE exactly
    y_exact_expr = sp.dsolve(ode, y, ics={y.subs(t, 0): y0}).rhs

    y_exact_func = sp.lambdify(t, y_exact_expr, 'numpy') # Converting to a function
    f_rhs = sp.lambdify((t, y), ode_expr, 'numpy') # Converting to a numerical function. Used for Euler, RK4.

    return f_rhs, y_exact_expr, y_exact_func, t
