#!/usr/bin/env python3
"""
ASTE 404 Mini-Project
Time integration of a damped mass-spring system:
Compare Forward Euler, Semi-Implicit Euler, and RK4
"""

import numpy as np
import matplotlib.pyplot as plt

# parameters
m = 1.0          # mass
k = 1.0          # spring constant
zeta = 0.1       # damping ratio (underdamped: 0 < zeta < 1)
c = 2.0 * zeta * np.sqrt(k * m)  # damping coefficient

x0 = 1.0         # initial displacement
v0 = 0.0         # initial velocity

t0 = 0.0
t_final = 20.0   # total simulation time


def deriv(t, y):
    """
    RHS of first-order system:
    y = [x, v]
    dy/dt = [v, - (c/m) v - (k/m) x]
    """
    x, v = y
    dxdt = v
    dvdt = -(c / m) * v - (k / m) * x
    return np.array([dxdt, dvdt])


# Time-stepping schemes
def step_euler(t, y, dt):
    """Forward Euler step."""
    return y + dt * deriv(t, y)

def step_semi_implicit_euler(t, y, dt):
    """Semi-implicit (symplectic) Euler step."""
    x, v = y
    # first update v using current x, v
    dvdt = -(c / m) * v - (k / m) * x
    v_new = v + dt * dvdt
    # then update x using new v
    x_new = x + dt * v_new
    return np.array([x_new, v_new])


def step_rk4(t, y, dt):
    """4th-order Runge–Kutta step."""
    k1 = deriv(t, y)
    k2 = deriv(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = deriv(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = deriv(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate(step_func, dt):
    """
    Generic integrator wrapper.
    Returns t, x, v arrays.
    """
    n_steps = int(np.ceil((t_final - t0) / dt))
    t = np.linspace(t0, t_final, n_steps + 1)

    x = np.zeros_like(t)
    v = np.zeros_like(t)

    y = np.array([x0, v0])
    x[0], v[0] = y

    for n in range(n_steps):
        y = step_func(t[n], y, dt)
        x[n+1], v[n+1] = y

    return t, x, v


# Analytic solution
def analytic_solution(t):
    """
    Analytic underdamped solution x(t).
    """
    omega0 = np.sqrt(k / m)
    omega_d = omega0 * np.sqrt(1.0 - zeta**2)

    # C1 and C2 from ICs
    C1 = x0
    C2 = (v0 + zeta * omega0 * x0) / omega_d

    x = np.exp(-zeta * omega0 * t) * (C1 * np.cos(omega_d * t) + C2 * np.sin(omega_d * t))
    return x


# Energy
def energy(x, v):
    return 0.5 * m * v**2 + 0.5 * k * x**2


# Main experiment
def main():
    # choose a set of dt values for convergence study
    dt_values = [0.5, 0.25, 0.125, 0.0625]

    methods = {
        "Euler": step_euler,
        "SemiImplicitEuler": step_semi_implicit_euler,
        "RK4": step_rk4,
    }

    # store final-time errors for each method and dt
    errors = {name: [] for name in methods.keys()}

    #  Example trajectories at one dt (for plotting)
    dt_plot = 0.05
    t_e, x_e, v_e = integrate(step_euler, dt_plot)
    t_si, x_si, v_si = integrate(step_semi_implicit_euler, dt_plot)
    t_rk, x_rk, v_rk = integrate(step_rk4, dt_plot)

    t_ana = t_rk  # use RK4 time grid for analytic comparison
    x_ana = analytic_solution(t_ana)

    #  Trajectory comparison plot 
    plt.figure()
    plt.plot(t_ana, x_ana, "k-", label="Analytic")
    plt.plot(t_e, x_e, "--", label=f"Euler dt={dt_plot}")
    plt.plot(t_si, x_si, "-.", label=f"Semi-Implicit dt={dt_plot}")
    plt.plot(t_rk, x_rk, ":", label=f"RK4 dt={dt_plot}")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("Damped mass-spring: trajectories")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/trajectory_compare.png", dpi=200)

    # --- Energy vs time (for RK4, best method) ---
    E_rk = energy(x_rk, v_rk)
    plt.figure()
    plt.plot(t_rk, E_rk)
    plt.xlabel("t")
    plt.ylabel("Energy E(t)")
    plt.title("Energy vs time (RK4)")
    plt.grid(True)
    plt.savefig("results/energy_vs_time.png", dpi=200)

    # Convergence study: error vs dt
    for dt in dt_values:
        for name, step_func in methods.items():
            t_num, x_num, v_num = integrate(step_func, dt)
            x_ana_dt = analytic_solution(t_num)

            # L2 error in x over entire time history
            err = np.sqrt(np.mean((x_num - x_ana_dt)**2))
            errors[name].append(err)
            print(f"{name}: dt = {dt:8.4f},  L2 error = {err:.6e}")

    #  Error vs dt plot (log-log)
    plt.figure()
    for name in methods.keys():
        plt.loglog(dt_values, errors[name], "o-", label=name)

    plt.xlabel("dt")
    plt.ylabel("L2 error in x(t)")
    plt.title("Convergence of time integrators")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("results/error_vs_dt.png", dpi=200)

    print("Done. Plots saved in ./results/")


if __name__ == "__main__":
    main()
