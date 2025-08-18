# src/evaluate_two_body.py
import os, numpy as np, pandas as pd, sympy as sp
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt

INFILE = "results/predictions_two_body.csv"   # produced by train_two_body.py
AX_TXT = "results/equations_two_body/ax_best.txt"
AY_TXT = "results/equations_two_body/ay_best.txt"
PLOT_DIR = "results/plots_two_body"
os.makedirs(PLOT_DIR, exist_ok=True)

def make_fn(eq_str):
    # Normalize possible PySR names
    eq_str = (eq_str.replace("x0", "x")
                     .replace("x1", "y")
                     .replace("x2", "rsq")
                     .replace("x3", "r"))
    x, y, rsq, r = sp.symbols("x y rsq r")
    expr = sp.sympify(eq_str)
    fn = lambdify((x, y, rsq, r), expr, "numpy")
    return fn, expr

def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mse = float(np.mean((y_true - y_pred)**2))
    denom = np.sum((y_true - y_true.mean())**2)
    r2 = float(1 - np.sum((y_true - y_pred)**2) / denom) if denom > 0 else float("nan")
    return mse, r2

df = pd.read_csv(INFILE)
x, y = df["x"].values, df["y"].values
rsq = x*x + y*y
ax_true, ay_true = df["ax"].values, df["ay"].values
ax_nn, ay_nn     = df["ax_pred"].values, df["ay_pred"].values

with open(AX_TXT) as f: ax_str = f.read().strip()
with open(AY_TXT) as f: ay_str = f.read().strip()
ax_fn, ax_expr = make_fn(ax_str)
ay_fn, ay_expr = make_fn(ay_str)

ax_sr = np.asarray(ax_fn(x, y, rsq), dtype=float)
ay_sr = np.asarray(ay_fn(x, y, rsq), dtype=float)

mse_ax_sr, r2_ax_sr = metrics(ax_true, ax_sr)
mse_ay_sr, r2_ay_sr = metrics(ay_true, ay_sr)
mse_ax_nn, r2_ax_nn = metrics(ax_true, ax_nn)
mse_ay_nn, r2_ay_nn = metrics(ay_true, ay_nn)

print("Best ax:", ax_expr)
print("Best ay:", ay_expr)
print(f"SR vs True (ax): MSE={mse_ax_sr:.3e}, R2={r2_ax_sr:.6f}")
print(f"SR vs True (ay): MSE={mse_ay_sr:.3e}, R2={r2_ay_sr:.6f}")
print(f"NN vs True (ax): MSE={mse_ax_nn:.3e}, R2={r2_ax_nn:.6f}")
print(f"NN vs True (ay): MSE={mse_ay_nn:.3e}, R2={r2_ay_nn:.6f}")

# time series (short window for readability)
t = df["time"].values
idx = slice(0, min(1500, len(t)))  # show first part
plt.figure(figsize=(12,5))
plt.plot(t[idx], ax_true[idx], label="ax true", lw=2)
plt.plot(t[idx], ax_nn[idx], "--", label="ax NN")
plt.plot(t[idx], ax_sr[idx], ":", label="ax SR")
plt.xlabel("time"); plt.ylabel("ax"); plt.title("Two-Body ax: true vs NN vs SR"); plt.grid(True); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "ax_compare.png")); plt.show()

plt.figure(figsize=(12,5))
plt.plot(t[idx], ay_true[idx], label="ay true", lw=2)
plt.plot(t[idx], ay_nn[idx], "--", label="ay NN")
plt.plot(t[idx], ay_sr[idx], ":", label="ay SR")
plt.xlabel("time"); plt.ylabel("ay"); plt.title("Two-Body ay: true vs NN vs SR"); plt.grid(True); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "ay_compare.png")); plt.show()

plt.figure(figsize=(6,6))
plt.scatter(ax_true, ax_sr, s=6, alpha=0.5, label="SR ax")
plt.scatter(ax_true, ax_nn, s=6, alpha=0.5, label="NN ax")
lims = np.linspace(min(ax_true.min(), ax_sr.min(), ax_nn.min()),
                   max(ax_true.max(), ax_sr.max(), ax_nn.max()), 200)
plt.plot(lims, lims, 'r', label="y=x")
plt.xlabel("ax true"); plt.ylabel("ax est"); plt.title("Parity ax"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "parity_ax.png")); plt.show()

plt.figure(figsize=(6,6))
plt.scatter(ay_true, ay_sr, s=6, alpha=0.5, label="SR ay")
plt.scatter(ay_true, ay_nn, s=6, alpha=0.5, label="NN ay")
lims = np.linspace(min(ay_true.min(), ay_sr.min(), ay_nn.min()),
                   max(ay_true.max(), ay_sr.max(), ay_nn.max()), 200)
plt.plot(lims, lims, 'r', label="y=x")
plt.xlabel("ay true"); plt.ylabel("ay est"); plt.title("Parity ay"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "parity_ay.png")); plt.show()