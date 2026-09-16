<div align="center">

# Laplace Transform & ODE Visualization

### A Python-based exploration of Laplace Transforms, Ordinary Differential Equations, and numerical methods

**🥉 3rd Place — School Exhibition, 2026**

</div>

---

## Overview

This project is a computational exploration of the **Laplace Transform** and its application to solving and analyzing **ordinary differential equations (ODEs)**.

The project was originally developed for a school exhibition held in **February 2026**. The goal was to combine mathematical theory with computational visualization, making concepts that are often difficult to visualize on paper more intuitive through interactive graphs and numerical comparisons.

The project uses **Python** as its primary language, taking advantage of libraries such as **SymPy**, **NumPy**, **Matplotlib**, and **Plotly** to perform symbolic mathematics, numerical computation, and visualization.

Rather than focusing exclusively on the Laplace Transform itself, the project also compares several approaches to solving an ODE:

- Exact symbolic solutions
- Euler's numerical method
- Fourth-order Runge-Kutta (RK4)
- Laplace Transforms
- Inverse Laplace Transforms
- Visualization in the complex frequency domain

The project is structured as several independent Python modules, each responsible for a particular part of the computation.

---

## The Mathematics

The project works with first-order ordinary differential equations of the form

$$
\frac{dy}{dt} = f(t,y)
$$

and compares numerical and analytical techniques for obtaining their solutions.

The primary example used by the program is:

$$
\frac{dy}{dt} = \sin(3t) - 2y
$$

with the initial condition

$$
y(0)=0.
$$

This equation can be solved in several ways, allowing the project to demonstrate the differences between numerical approximation, exact symbolic solutions, and Laplace-domain methods.

### Laplace Transform

For a function $f(t)$, its Laplace Transform is defined as $\mathcal{L}\{f(t)\} = \int_0^\infty e^{-st}f(t) \mathrm{d}t$.

The transform converts a function from the **time domain** into the **complex frequency domain**. One of the major advantages of this transformation is that differentiation in the time domain can be converted into algebraic operations in the $s$-domain.

For example, $\mathcal{L}\{y'(t)\} = sY(s)-y(0)$.

This makes Laplace Transforms particularly useful for solving differential equations with initial conditions.

The project uses **SymPy** to perform these transformations symbolically and then converts the resulting expressions into numerical functions for visualization.

---

## Numerical Methods

### Euler's Method

Euler's method provides a simple numerical approximation to the solution of an ODE.

Given

$$
\frac{dy}{dt}=f(t,y),
$$

the method approximates the next value using

$$
y_{n+1}=y_n+h f(t_n,y_n),
$$

where $h$ is the step size.

The implementation can be found in [`euler.py`](euler.py).

---

### Fourth-Order Runge-Kutta

The project also implements the classical **fourth-order Runge-Kutta method (RK4)**.

For each time step, RK4 evaluates the differential equation at several points and combines these slopes using a weighted average:

$$y_{n+1} = y_n + \frac h6 (k_1 + 2k_2 + 3k_3 + k_4)$$.

Compared with Euler's method, RK4 generally provides a much more accurate approximation for the same step size.

The implementation can be found in [`rk4.py`](rk4.py).

---

## Symbolic ODE Solutions

The project uses **SymPy** to obtain an exact symbolic solution to the differential equation.

The module [`odesolutions.py`](odesolutions.py) is responsible for:

1. Constructing the differential equation.
2. Applying the initial condition.
3. Solving the equation symbolically using SymPy.
4. Converting the symbolic solution into a numerical function.
5. Providing a numerical version of the ODE for Euler and RK4.

This makes it possible to directly compare numerical approximations with an analytical solution.

---

## Laplace Transform Implementation

The core Laplace functionality is contained in [`laplace.py`](laplace.py).

It provides two primary operations:

### `compute_laplace()`

Computes the symbolic Laplace Transform of a time-domain expression and simultaneously creates a numerical function that can be evaluated using NumPy.

Conceptually:

```text
Time Domain
    │
    │  Laplace Transform
    ▼
Frequency Domain
     Y(s)
```

### `compute_inverse_laplace()`

Performs the reverse operation, converting a symbolic expression in the \(s\)-domain back into the time domain.

```text
Frequency Domain
     Y(s)
       │
       │  Inverse Laplace Transform
       ▼
Time Domain
     y(t)
```

Both functions return the symbolic expression as well as a numerical callable function, allowing the results to be used both mathematically and computationally.

---

## Visualization

One of the main purposes of the project is to make the mathematics visually understandable.

The program generates several plots comparing the different solution methods.

### Numerical vs. Exact Solutions

The main program plots:

- Euler's method
- RK4
- The exact symbolic solution
- A combined comparison of all three

This makes it possible to see how closely each numerical method follows the exact solution.

### Laplace vs. Inverse Laplace

The program also compares the inverse Laplace result against the original exact solution, providing a visual check that the transformation and inverse transformation are consistent.

### Complex-Plane Visualization

The project additionally includes a 3D visualization of the magnitude of the transformed solution.

The complex variable is represented as

$$
s=\sigma+i\omega,
$$

where:

- $\sigma$ is the real component
- $\omega$ is the imaginary component

The visualization plots

$$
|Y(s)|
$$

across a region of the complex plane.

A vertical plane is then introduced at a chosen value of $\sigma$, and its intersection with the surface is plotted separately. This provides a way of examining a **slice of the Laplace-domain function** while retaining the surrounding 3D context.

The visualization is implemented in [`laplacevisualize.py`](laplacevisualize.py) using **Plotly**.

---

## Project Structure

```text
.
├── mainprogram.py
├── laplace.py
├── laplacevisualize.py
├── odesolutions.py
├── euler.py
├── rk4.py
└── README.md
```

### File Descriptions

| File | Purpose |
|---|---|
| `mainprogram.py` | Main entry point that connects the different components of the project |
| `laplace.py` | Computes Laplace and inverse Laplace Transforms using SymPy |
| `laplacevisualize.py` | Creates the 3D complex-domain visualization |
| `odesolutions.py` | Solves the ODE symbolically and prepares numerical functions |
| `euler.py` | Implements Euler's numerical method |
| `rk4.py` | Implements the fourth-order Runge-Kutta method |
| `README.md` | Project documentation |

---

## Technologies Used

### Python

Python was chosen primarily for its readability and extensive ecosystem for scientific computing.

### SymPy

Used for symbolic mathematics, including:

- Solving differential equations
- Computing Laplace Transforms
- Computing inverse Laplace Transforms
- Simplifying mathematical expressions
- Converting symbolic expressions into numerical functions

### NumPy

Used for numerical computation and generation of the arrays required for the numerical methods and visualizations.

### Matplotlib

Used for the project's 2D plots and comparisons between numerical and analytical solutions.

### Plotly

Used to create the interactive 3D visualization of the Laplace-domain function.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

Install the required Python packages:

```bash
pip install numpy sympy matplotlib plotly
```

You can also install them individually if you prefer:

```bash
pip install numpy
pip install sympy
pip install matplotlib
pip install plotly
```

---

## Running the Project

The main program is:

```bash
python mainprogram.py
```

This will:

1. Define the differential equation.
2. Solve it symbolically.
3. Approximate the solution using Euler's method.
4. Approximate the solution using RK4.
5. Compute its Laplace Transform.
6. Compute the inverse Laplace Transform.
7. Generate comparison plots.
8. Launch the 3D Laplace-domain visualization.

The program currently operates using the example equation defined directly in `mainprogram.py` rather than accepting arbitrary equations through a command-line interface.

---

## Example Workflow

At a high level, the project follows this pipeline:

```text
                 Differential Equation
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Exact Solution          Numerical Methods
             │                 ┌─────┴─────┐
             │                 │           │
             │              Euler         RK4
             │                 │           │
             └──────────┬──────┴───────────┘
                        │
                        ▼
                 Compare Solutions
                        │
                        ▼
                 Laplace Transform
                        │
                        ▼
                       Y(s)
                        │
                        ▼
              Inverse Laplace Transform
                        │
                        ▼
                       y(t)
                        │
                        ▼
                Compare with Exact
```

Alongside this workflow, `laplacevisualize.py` evaluates $Y(s)$ over a region of the complex plane and produces a 3D representation of its magnitude.

---

## Background

This project was also a continuation of my previous work with mathematical visualization in Python. Before developing this project, I had created a **Fourier Transform plotter**, which introduced me to the computational visualization of transforms and provided some of the foundation for this project.

This project expanded that idea into differential equations, symbolic mathematics, numerical methods, and the complex frequency domain.

---

## Limitations & Possible Improvements

The current implementation was built primarily as an exhibition project, so there are several areas that could be expanded in the future.

### User Input

The differential equation is currently defined directly in the source code. A future version could provide a proper interface for entering arbitrary ODEs.

### Broader ODE Support

The current implementation focuses on first-order ODEs. It could potentially be extended to support more classes of differential equations.

### Numerical Error Analysis

The project could calculate and visualize the error between Euler, RK4, and the exact solution rather than relying solely on visual comparison.

For example:

$$
E(t)=|y_{\text{numerical}}(t)-y_{\text{exact}}(t)|.
$$

### Improved Interactivity

The visualization could be expanded with interactive controls for changing parameters such as:

- The differential equation
- Initial conditions
- Time interval
- Number of numerical steps
- The $\sigma$-slice used in the complex-plane visualization

### Convergence Analysis

Another possible extension would be to demonstrate how numerical accuracy changes as the step size is reduced, providing a computational illustration of the convergence properties of Euler's method and RK4.

---

</div>
