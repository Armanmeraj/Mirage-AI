import pandas as pd
from pysr import pysr, best
import os
import matplotlib.pyplot as plt

# === Config ===
USE_PREDICTIONS = True  # Set to False to use true acceleration
INPUT_FILE = "../results/predictions.csv" if USE_PREDICTIONS else "../data/simulated/mass_spring.csv"
OUTPUT_DIR = "../results/equations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Data ===
df = pd.read_csv(INPUT_FILE)
X = df[["position", "velocity"]].values
y = df["predicted_acceleration" if USE_PREDICTIONS else "acceleration"].values

# === Run Symbolic Regression ===
print("Running symbolic regression...")
equations = pysr(
    X,
    y,
    model_selection="best",  # Use the lowest loss model
    niterations=100,
    binary_operators=["+", "-", "*"],
    unary_operators=["square", "cube"],
    extra_sympy_mappings={"square": lambda x: x**2, "cube": lambda x: x**3},
    variable_names=["x", "v"],
    loss="loss(x, y) = (x - y)^2",
    progress=True,
    verbosity=1,
)

# === Output Best Equation ===
best_eq = best(equations)
print("\nBest Equation Found:")
print(best_eq)

# === Save Results ===
equations.to_csv(os.path.join(OUTPUT_DIR, "symbolic_equations.csv"), index=False)
with open(os.path.join(OUTPUT_DIR, "best_equation.txt"), "w") as f:
    f.write(str(best_eq))
