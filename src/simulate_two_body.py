# src/simulate_two_body.py
import numpy as np
from scipy.integrate import odeint
import pandas as pd
import os

def two_body_ode(state, t, mu=1.0):
    x, y, vx, vy = state
    r2 = x * x + y * y
    r = np.sqrt(r2)
    inv_r3 = 1.0 / (r2 * r + 1e-12)  # epsilon for safety if r≈0
    ax = -mu * x * inv_r3
    ay = -mu * y * inv_r3
    return [vx, vy, ax, ay]


def _ic_circular(mu=1.0, r0=1.0):
    """Return initial conditions for a circular orbit at radius r0."""
    x0, y0 = r0, 0.0
    v = np.sqrt(mu / r0)
    vx0, vy0 = 0.0, v
    return [x0, y0, vx0, vy0]


def _ic_elliptic(mu=1.0, r0=1.0, speed_scale=0.8):
    """Return ICs for a bound elliptical orbit (tangential speed below circular)."""
    x0, y0 = r0, 0.0
    v = np.sqrt(mu / r0) * speed_scale
    vx0, vy0 = 0.0, v
    return [x0, y0, vx0, vy0]


def generate_two_body_data(mu=1.0, r0=1.0, orbit="elliptic", t_end=30.0, num_points=6000):
    """
    Generate a single two-body trajectory.
    orbit: 'circular' or 'elliptic'
    """
    if orbit == "circular":
        y_init = _ic_circular(mu, r0)
    else:
        y_init = _ic_elliptic(mu, r0, speed_scale=0.8)

    t = np.linspace(0, t_end, num_points)
    sol = odeint(two_body_ode, y_init, t, args=(mu,))

    x, y, vx, vy = sol.T
    dt = t[1] - t[0]
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)

    df = pd.DataFrame({
        "time": t,
        "x": x, "y": y,
        "vx": vx, "vy": vy,
        "ax": ax, "ay": ay
    })
    return df


def generate_two_body_multi(mu=1.0, r0_list=(1.0, 1.5), speeds=(0.7, 0.85, 1.0),
                            t_end=30.0, num_points=6000):
    """
    Generate and stack multiple trajectories with different radii/speeds.
    This encourages SR to recover the 1/r^3 dependence.
    """
    frames = []
    for r0 in r0_list:
        for s in speeds:
            orbit = "circular" if abs(s - 1.0) < 1e-6 else "elliptic"
            # Build ICs explicitly so the speed difference is respected
            if orbit == "circular":
                y_init = _ic_circular(mu, r0)
            else:
                y_init = _ic_elliptic(mu, r0, speed_scale=s)

            t = np.linspace(0, t_end, num_points)
            sol = odeint(two_body_ode, y_init, t, args=(mu,))
            x, y, vx, vy = sol.T
            dt = t[1] - t[0]
            ax = np.gradient(vx, dt)
            ay = np.gradient(vy, dt)

            df = pd.DataFrame({
                "time": t,
                "x": x, "y": y,
                "vx": vx, "vy": vy,
                "ax": ax, "ay": ay,
                "r0": r0,
                "speed_scale": s,
                "orbit": orbit,
            })
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def save_two_body(filename="two_body.csv", multi=False, **kwargs):
    os.makedirs("data/simulated", exist_ok=True)
    if multi:
        df = generate_two_body_multi(**kwargs)
    else:
        df = generate_two_body_data(**kwargs)
    path = os.path.join("data/simulated", filename)
    df.to_csv(path, index=False)
    print(f"Saved two-body dataset to {path}")


if __name__ == "__main__":
    # Default to a single elliptical trajectory for stronger SR signal
    save_two_body(multi=False)