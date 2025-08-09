import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# === Config ===
CSV_PATH = "data/simulated/mass_spring.csv"
SAVE_MODEL_PATH = "results/model.pt"
SAVE_PREDICTIONS_PATH = "results/predictions.csv"
EPOCHS = 1000
LR = 0.01
HIDDEN = 64

# === MLP Model ===
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1)
        )

    def forward(self, x):
        return self.net(x)

# === Load and prepare data ===
df = pd.read_csv(CSV_PATH)
X = df[["position", "velocity"]].values.astype(np.float32)
y = df["acceleration"].values.astype(np.float32).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

# === Model setup ===
model = MLP()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === Training loop ===
losses = []
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# === Evaluation ===
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    test_loss = loss_fn(y_pred_test, y_test_tensor).item()
    print(f"Final Test Loss: {test_loss:.6f}")

# === Save model ===
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), SAVE_MODEL_PATH)

# === Save predictions ===
all_preds = model(torch.tensor(X).float()).detach().numpy()
df["predicted_acceleration"] = all_preds
df.to_csv(SAVE_PREDICTIONS_PATH, index=False)

# === Plot training loss ===
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/loss_curve.png")
plt.show()