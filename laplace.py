import sympy as sp

def compute_laplace(expr, t_symbol, s_symbol):
    """
    Computes the Laplace transform of a time-domain expression, converting
    a function from the time domain (t) to the complex frequency domain (s).

    This function performs two distinct operations:
    1. Symbolic transformation: Uses SymPy to find the exact algebraic expression.
    2. Numerical conversion: Creates a fast, callable function for numerical plotting/evaluation.

    Parameters:
        expr (sympy.Expr): The symbolic mathematical expression in the time domain.
        t_symbol (sympy.Symbol): The independent variable representing time (e.g., t).
        s_symbol (sympy.Symbol): The complex variable representing frequency (e.g., s).

    Returns:
        Y_s (sympy.Expr): The resulting symbolic Laplace transform of the input expression.
        Y_s_num (function): A callable Python function that accepts numerical numpy arrays
                            for 's' and evaluates Y_s mathematically.
    """
    # Compute the symbolic Laplace Transform.
    # The 'noconds=True' argument forces SymPy to only return the transformed expression.
    # Without it, SymPy would also return the convergence conditions (the region of the 
    # complex plane where the integral actually converges), which often complicates unpacking.
    Y_s = sp.laplace_transform(expr, t_symbol, s_symbol, noconds=True)

    # Turn the symbolic Laplace Transform into a callable numerical function.
    # sp.lambdify translates SymPy expressions into equivalent NumPy-compatible functions.
    # This is required because you cannot easily pass numerical arrays directly into a 
    # pure SymPy symbol without evaluating it point-by-point (which is very slow).
    Y_s_num = sp.lambdify(s_symbol, Y_s, modules='numpy')
    
    return Y_s, Y_s_num

def compute_inverse_laplace(Y_s, s_symbol, t_symbol):
    """
    Computes the inverse Laplace transform, converting a mathematical expression
    from the complex frequency domain (s) back into the original time domain (t).

    Like compute_laplace, this returns both an exact symbolic formula and a 
    callable numerical function optimized for data arrays.

    Parameters:
        Y_s (sympy.Expr): The symbolic expression in the Laplace (frequency) domain.
        s_symbol (sympy.Symbol): The complex variable representing frequency (e.g., s).
        t_symbol (sympy.Symbol): The independent variable representing time (e.g., t).

    Returns:
        y_t (sympy.Expr): The resulting exact symbolic time-domain function.
        y_t_num (function): A callable Python function that accepts numerical numpy arrays
                            for 't' (like a time vector) and evaluates y_t numerically.
    """
    # Compute the exact symbolic inverse Laplace Transform.
    # This evaluates the complex Bromwich contour integral to revert 's' back to 't'.
    # It will often introduce Heaviside step functions if the signal 'turns on' at t=0.
    y_t = sp.inverse_laplace_transform(Y_s, s_symbol, t_symbol)

    # Turn the symbolic inverse transform into a callable numerical function.
    # By specifying modules='numpy', the resulting function 'y_t_num' will use NumPy's 
    # vectorized math operations (like np.exp instead of math.exp), allowing it to 
    # efficiently compute the time-domain values across thousands of time steps at once.
    y_t_num = sp.lambdify(t_symbol, y_t, modules='numpy')
    
    return y_t, y_t_num
