import numpy as np
from scipy.integrate import odeint
import pandas as pd
import os

def mass_spring_ode(y, t, omega):
    """ODE function for a simple mass-spring system: a = -omega^2 * x"""
    x, v = y
    dxdt = v
    dvdt = -omega**2 * x
    return [dxdt, dvdt]

def generate_mass_spring_data(omega=2.0, x0=1.0, v0=0.0, t_end=10.0, num_points=1000):
    """Simulate the motion of a mass-spring system."""
    t = np.linspace(0, t_end, num_points)
    y0 = [x0, v0]
    sol = odeint(mass_spring_ode, y0, t, args=(omega,))

    x = sol[:, 0]
    v = sol[:, 1]
    dt = t[1] - t[0]
    a = np.gradient(v, dt)

    data = pd.DataFrame({
        "time": t,
        "position": x,
        "velocity": v,
        "acceleration": a
    })

    return data

def save_simulation(data, filename="mass_spring.csv", output_dir="data/simulated"):
    """Save the simulation data as a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    data.to_csv(path, index=False)
    print(f"Saved simulation to {path}")

if __name__ == "__main__":
    data = generate_mass_spring_data()
    save_simulation(data)
