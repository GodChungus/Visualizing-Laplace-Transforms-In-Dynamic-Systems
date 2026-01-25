import sympy as sp
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from laplace import compute_laplace

# ======================================================
# Defining the symbols
# ======================================================
t, s = sp.symbols('t s', real=True)
sigma, omega = sp.symbols('sigma omega', real=True)

# ======================================================
# RHS of the ODE
# ======================================================
rhs = sp.sin(3*t)

R_s, _ = compute_laplace(rhs, t, s) # Laplace Transform of the RHS of the ODE

Y = sp.Symbol('Y') # Y(s)

Y_s = sp.solve(s*Y - (R_s - 2*Y), Y)[0] # Solution for Y(s) [note this depends on the actual ODE]
Y_s = sp.simplify(Y_s)

print("Y(s) =")
sp.pprint(Y_s)

# ======================================================
# Converting Y(s) into a numerical function
# Note s = σ + iω
# ======================================================
Y_func = sp.lambdify((sigma, omega), Y_s.subs(s, sigma + sp.I*omega), 'numpy')

# ======================================================
# Grid in the complex plane
# ======================================================
sigma_vals = np.linspace(-4, 2, 250) # Range for real axis
omega_vals = np.linspace(-8, 8, 300) # Range for complex axis

SIGMA, OMEGA = np.meshgrid(sigma_vals, omega_vals)

Z = np.abs(Y_func(SIGMA, OMEGA)) # |Y(s)|, the magnitude

sigma_cut = -2 # The x-coordinate of the plane that "slices" the 3D plot

# ======================================================
# 3D Surface plot of the magitude of Y(s)
# ======================================================
surface = go.Surface(
    x=SIGMA,
    y=OMEGA,
    z=Z,
    colorscale='Jet',
    opacity=0.9,
    name='|Y(s)|'
)

# ======================================================
# Vertical plane of the "slice"
# ======================================================
omega_plane = np.linspace(-8, 8, 400)
z_plane = np.linspace(0, Z.max(), 200)

OMEGA_PLANE, Z_PLANE = np.meshgrid(omega_plane, z_plane)
SIGMA_PLANE = np.full_like(OMEGA_PLANE, sigma_cut)

plane = go.Surface(
    x=SIGMA_PLANE,
    y=OMEGA_PLANE,
    z=Z_PLANE,
    colorscale=[[0, "red"], [1, "red"]],
    opacity=0.5,
    showscale=False,
    name=f'Plane σ={sigma_cut}'
)

# ======================================================
# Intersection Curve
# ======================================================

slice_vals = np.abs(Y_func(sigma_cut, omega_plane))

curve = go.Scatter3d(
    x=np.full_like(omega_plane, sigma_cut),
    y=omega_plane,
    z=slice_vals,
    mode="lines",
    line=dict(color="black", width=6),
    name='Slice'
)

# ======================================================
# Plotting the 3D Figure
# ======================================================
fig = go.Figure(data=[surface, plane, curve])
fig.update_layout(
    title="Laplace Transform of ODE",
    scene=dict(
        xaxis_title="σ",
        yaxis_title="ω",
        zaxis_title="Magnitude"
    )
)
fig.show()

# ======================================================
# 2D Plot of the slice
# ======================================================
plt.figure(figsize=(8,4))
plt.plot(omega_plane, slice_vals, color='red')
plt.title(f'Slice of ODE Solution at {sigma_cut}')
plt.xlabel('ω')
plt.ylabel('|Y(σ + iω)|')
plt.grid(True)
plt.show()
