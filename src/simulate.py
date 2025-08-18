import numpy as np
from scipy.integrate import odeint
import pandas as pd
import os
from typing import Iterable, Optional

def mass_spring_ode(y, t, omega):
    """ODE function for a simple mass-spring system: a = -omega^2 * x"""
    x, v = y
    dxdt = v
    dvdt = -omega**2 * x
    return [dxdt, dvdt]

def generate_mass_spring_data(
    omega: float = 2.0,
    x0: float = 1.0,
    v0: float = 0.0,
    t_end: float = 10.0,
    num_points: int = 1000,
    noise_std: float = 0.0,
    noise_on: Optional[Iterable[str]] = ("position", "velocity"),
    random_seed: Optional[int] = None,
):
    """Simulate a mass-spring system and (optionally) add Gaussian measurement noise.

    Parameters
    ----------
    omega : float
        Natural angular frequency.
    x0, v0 : float
        Initial position and velocity.
    t_end : float
        Simulation end time (seconds).
    num_points : int
        Number of time samples.
    noise_std : float
        Standard deviation of Gaussian noise to add to selected measured columns.
        Use 0.0 for no noise (default).
    noise_on : Iterable[str] | None
        Which columns to perturb (subset of {"position", "velocity", "acceleration"}).
        By default, noise is added to position and velocity only (common in experiments).
    random_seed : int | None
        Seed for reproducible noise.

    Returns
    -------
    pandas.DataFrame
        Columns: time, position, velocity, acceleration. Note: acceleration is
        computed from the *clean* velocity using a numerical derivative; if you
        want noisy acceleration too, include "acceleration" in `noise_on`.
    """
    t = np.linspace(0, t_end, num_points)
    y0 = [x0, v0]
    sol = odeint(mass_spring_ode, y0, t, args=(omega,))

    x_clean = sol[:, 0]
    v_clean = sol[:, 1]
    dt = t[1] - t[0]
    a_clean = np.gradient(v_clean, dt)

    # Start from clean signals
    x = x_clean.copy()
    v = v_clean.copy()
    a = a_clean.copy()

    if noise_std and noise_std > 0.0:
        rng = np.random.default_rng(random_seed)
        if noise_on:
            if "position" in noise_on:
                x = x + rng.normal(0.0, noise_std, size=x.shape)
            if "velocity" in noise_on:
                v = v + rng.normal(0.0, noise_std, size=v.shape)
            if "acceleration" in noise_on:
                a = a + rng.normal(0.0, noise_std, size=a.shape)

    data = pd.DataFrame({
        "time": t,
        "position": x,
        "velocity": v,
        "acceleration": a,
    })

    return data

def save_simulation(data, filename="mass_spring.csv", output_dir="data/simulated"):
    """Save the simulation data as a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    data.to_csv(path, index=False)
    print(f"Saved simulation to {path}")

if __name__ == "__main__":
    # Example: clean (no noise)
    clean = generate_mass_spring_data()
    save_simulation(clean, filename="mass_spring.csv")

    # Example: noisy measurements (position & velocity), reproducible
    noisy = generate_mass_spring_data(noise_std=0.01, random_seed=42)
    save_simulation(noisy, filename="mass_spring_noisy.csv")
