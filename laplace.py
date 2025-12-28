import sympy as sp

def compute_laplace(expr, t_symbol, s_symbol):
    Y_s = sp.laplace_transform(expr, t_symbol, s_symbol, noconds=True)
    Y_s_num = sp.lambdify(s_symbol, Y_s, modules='numpy')
    return Y_s, Y_s_num

def compute_inverse_laplace(Y_s, s_symbol, t_symbol):
    y_t = sp.inverse_laplace_transform(Y_s, s_symbol, t_symbol)
    y_t_num = sp.lambdify(t_symbol, y_t, modules='numpy')
    return y_t, y_t_num
