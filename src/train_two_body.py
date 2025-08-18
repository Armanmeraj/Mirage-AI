# src/train_two_body.py
import torch, torch.nn as nn, torch.optim as optim
import pandas as pd, numpy as np, os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

CSV_PATH = "data/simulated/two_body.csv"
SAVE_MODEL_PATH = "results/model_two_body.pt"
SAVE_PRED_PATH  = "results/predictions_two_body.csv"
EPOCHS, LR, H = 800, 1e-3, 128

class MLP(nn.Module):
    def __init__(self, in_dim=4, out_dim=2, hidden=H):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

df = pd.read_csv(CSV_PATH)
X = df[["x","y","vx","vy"]].values.astype(np.float32)
Y = df[["ax","ay"]].values.astype(np.float32)

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)
Xtr, Xte = torch.tensor(Xtr), torch.tensor(Xte)
Ytr, Yte = torch.tensor(Ytr), torch.tensor(Yte)

model = MLP()
opt = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

losses = []
for e in range(EPOCHS):
    model.train(); opt.zero_grad()
    pred = model(Xtr)
    loss = loss_fn(pred, Ytr)
    loss.backward(); opt.step()
    losses.append(loss.item())
    if e % 100 == 0: print(f"Epoch {e}: loss={loss.item():.6e}")

model.eval()
with torch.no_grad():
    test_loss = loss_fn(model(Xte), Yte).item()
print(f"Final test loss: {test_loss:.6e}")

os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), SAVE_MODEL_PATH)

with torch.no_grad():
    all_preds = model(torch.tensor(X)).numpy()
df_out = df.copy()
df_out["ax_pred"], df_out["ay_pred"] = all_preds[:,0], all_preds[:,1]
df_out.to_csv(SAVE_PRED_PATH, index=False)

plt.figure(figsize=(8,4))
plt.plot(losses); plt.grid(True); plt.title("Two-Body Training Loss"); plt.xlabel("epoch"); plt.ylabel("MSE")
plt.tight_layout(); plt.savefig("results/loss_curve_two_body.png"); plt.show()
print(f"Saved {SAVE_PRED_PATH}")