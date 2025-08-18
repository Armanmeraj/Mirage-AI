# src/run_symbolic_two_body.py
import os, numpy as np, pandas as pd
from pysr import PySRRegressor

USE_PRED = True  # True: fit NN predictions; False: fit true accelerations
INFILE = "results/predictions_two_body.csv" if USE_PRED else "data/simulated/two_body.csv"
OUTDIR = "results/equations_two_body"
RUNS   = "results/runs_two_body"
os.makedirs(OUTDIR, exist_ok=True); os.makedirs(RUNS, exist_ok=True)

df = pd.read_csv(INFILE)
x, y = df["x"].values, df["y"].values
vx, vy = df["vx"].values, df["vy"].values
ax = df["ax_pred" if USE_PRED else "ax"].values
ay = df["ay_pred" if USE_PRED else "ay"].values

rsq = x*x + y*y
r = np.sqrt(rsq)
inv_r3 = 1.0 / (rsq * r + 1e-12)
# Features: x, y, rsq, r, inv_r3 (gives SR multiple ways to form r^-3)
X_feats = np.column_stack([x, y, rsq, r, inv_r3])
var_names = ["x", "y", "rsq", "r", "inv_r3"]

def fit_axis(y_target, axis_name):
    model = PySRRegressor(
        niterations=1500,  # more search
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        extra_sympy_mappings={"sqrt": lambda z: z**0.5, "inv": lambda z: 1/z},
        maxsize=32,
        constraints={"+": (5,5), "-": (5,5), "*": (5,5), "/": (5,5)},
        model_selection="best",
        progress=True,
        verbosity=1,
        output_directory=os.path.join(RUNS, axis_name),
    )
    print(f"\n=== Fitting {axis_name} ===")
    model.fit(X_feats, y_target, variable_names=var_names)
    expr = model.sympy()
    csv_path = os.path.join(OUTDIR, f"{axis_name}_equations.csv")
    txt_path = os.path.join(OUTDIR, f"{axis_name}_best.txt")
    model.equations_.to_csv(csv_path, index=False)
    with open(txt_path, "w") as f: f.write(str(expr))
    print(f"✅ Best {axis_name}: {expr}")
    print(f"Saved: {csv_path}, {txt_path}")
    print("Hint: look for x*inv_r3 and y*inv_r3 terms; coefficients should be ≈ -mu.")
    return expr

expr_ax = fit_axis(ax, "ax")
expr_ay = fit_axis(ay, "ay")
print("\nDone.")