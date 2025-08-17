import os
import numpy as np
import pandas as pd
import sympy as sp
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt

PREDICTIONS_CSV = "results/predictions.csv"  # time, position, velocity, acceleration, predicted_acceleration
BEST_EQUATION_TXT = "results/equations/best_equation.txt"
PLOTS_DIR = "results/plots"


def _ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def _load_data():
    if not os.path.exists(PREDICTIONS_CSV):
        raise FileNotFoundError(
            f"Could not find {PREDICTIONS_CSV}. Run `python src/train_model.py` first.")
    df = pd.read_csv(PREDICTIONS_CSV)
    required_cols = {"time", "position", "velocity", "acceleration", "predicted_acceleration"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {PREDICTIONS_CSV}: {sorted(missing)}")
    return df


def _load_symbolic():
    if not os.path.exists(BEST_EQUATION_TXT):
        raise FileNotFoundError(
            f"Could not find {BEST_EQUATION_TXT}. Run `python src/run_symbolic.py` first.")
    with open(BEST_EQUATION_TXT, "r") as f:
        eq_str = f.read().strip()
    if not eq_str:
        raise ValueError("Best equation file is empty.")
    return eq_str


def _build_symbolic_fn(eq_str):
    # Normalize PySR variable names (x0/x1) to our canonical symbols (x/v)
    eq_norm = eq_str.replace("x0", "x").replace("x1", "v")
    x, v = sp.symbols("x v")
    expr = sp.sympify(eq_norm)
    fn = lambdify((x, v), expr, "numpy")
    return fn, expr


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2))
    # R^2 with protection against constant y_true
    denom = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else float("nan")
    return mse, r2


def main():
    _ensure_dirs()
    df = _load_data()
    eq_str = _load_symbolic()

    sym_fn, sym_expr = _build_symbolic_fn(eq_str)
    a_sym = sym_fn(df["position"].values, df["velocity"].values)
    a_sym = np.asarray(a_sym, dtype=float)

    a_true = df["acceleration"].values
    a_nn = df["predicted_acceleration"].values

    mse_sr, r2_sr = _metrics(a_true, a_sym)
    mse_nn, r2_nn = _metrics(a_true, a_nn)

    print("Best equation (from PySR):", eq_str)
    print(f"Symbolic vs True -> MSE: {mse_sr:.6g}, R^2: {r2_sr:.6g}")
    print(f"NN vs True       -> MSE: {mse_nn:.6g}, R^2: {r2_nn:.6g}")

    # Time-series plot
    t = df["time"].values
    plt.figure(figsize=(12, 5))
    plt.plot(t, a_true, label="True a(t)", linewidth=2)
    plt.plot(t, a_nn, "--", label="NN â(t)")
    plt.plot(t, a_sym, ":", label="SR ã(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.title("Acceleration: True vs NN vs Symbolic")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join(PLOTS_DIR, "acceleration_compare.png")
    plt.savefig(out1)
    plt.show()

    # Parity plot
    plt.figure(figsize=(6, 6))
    plt.scatter(a_true, a_nn, alpha=0.6, label="NN")
    plt.scatter(a_true, a_sym, alpha=0.6, label="Symbolic")
    lo = min(a_true.min(), a_nn.min(), np.min(a_sym))
    hi = max(a_true.max(), a_nn.max(), np.max(a_sym))
    diag = np.linspace(lo, hi, 200)
    plt.plot(diag, diag, "r", label="y=x")
    plt.xlabel("True a")
    plt.ylabel("Estimated a")
    plt.title("Parity Plot")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(PLOTS_DIR, "parity.png")
    plt.savefig(out2)
    plt.show()


if __name__ == "__main__":
    main()