import pandas as pd
from pysr import PySRRegressor
import os
import matplotlib.pyplot as plt

# === Config ===
USE_PREDICTIONS = True  # Set to False to use true acceleration
INPUT_FILE = "results/predictions.csv" if USE_PREDICTIONS else "data/simulated/mass_spring.csv"
OUTPUT_DIR = "results/equations"
RUNS_DIR = "results/runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# === Load Data ===
df = pd.read_csv(INPUT_FILE)
X = df[["position", "velocity"]].values
y = df["predicted_acceleration" if USE_PREDICTIONS else "acceleration"].values

# === Run Symbolic Regression ===
print("Running symbolic regression...")
model = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*"],
    unary_operators=[],
    constraints={"+": (5, 5), "-": (5, 5), "*": (5, 5)},
    maxsize=20,
    variable_names=["x", "v"],
    loss="loss(x, y) = (x - y)^2",
    model_selection="best",
    progress=True,
    verbosity=1,
    output_directory=RUNS_DIR,
)
model.fit(X, y)

# === Output Best Equation ===
best_eq = model.sympy()
print("\n\u2705 Best Equation Found:")
print(best_eq)

# === Save Results ===
model.equations_.to_csv(os.path.join(OUTPUT_DIR, "symbolic_equations.csv"), index=False)
with open(os.path.join(OUTPUT_DIR, "best_equation.txt"), "w") as f:
    f.write(str(best_eq))
