import sympy as sp

def compute_laplace(expr, t_symbol, s_symbol):
    """
    Computes the Laplace transform of a time-domain expression.

    Parameters:
        expr: sympy expression
        t_symbol: sympy symbol representing time
        s_symbol: sympy symbol representing the Laplace variable
    Returns:
        Y_s: Laplace transform of expression(expr)
        Y_s_num: numerical version of Y_s
    """
    # Compute the Laplace Transform
    Y_s = sp.laplace_transform(expr, t_symbol, s_symbol, noconds=True) # noconds removes convergence conditions

    # Turn the symbolic Laplace Transform into a numerical function
    Y_s_num = sp.lambdify(s_symbol, Y_s, modules='numpy')
    return Y_s, Y_s_num

def compute_inverse_laplace(Y_s, s_symbol, t_symbol):
    """
    Computes the inverse laplace transform of a time-domain expression.

    Parameters:
        Y_s: Laplace transform of expression(expr)
        s_symbol: sympy symbol representing Laplace variable
        t_symbol: sympy symbol representing time
    Returns:
        y_t: time-domain function
        y_t_num: numerical version of y_t
    """

    # Compute the inverse Laplace Transform
    y_t = sp.inverse_laplace_transform(Y_s, s_symbol, t_symbol)

    # Turn the symbolic inverse transform into a numerical function
    y_t_num = sp.lambdify(t_symbol, y_t, modules='numpy')
    return y_t, y_t_num
