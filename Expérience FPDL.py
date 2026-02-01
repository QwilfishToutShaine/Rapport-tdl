import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Data: MNIST binary (3 vs 5)
# ----------------------------
def load_mnist_binary(digit_a=3, digit_b=5, n_train=None, n_test=None, seed=0):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    def filter_ds(ds, n=None):
        idx = [i for i, (_, y) in enumerate(ds) if y in [digit_a, digit_b]]
        if n is not None:
            n = min(n, len(idx))
            rng = np.random.default_rng(seed)
            idx = rng.choice(idx, size=n, replace=False).tolist()
        return Subset(ds, idx)

    train = filter_ds(train, n_train)
    test  = filter_ds(test,  n_test)
    return train, test

def to_pm_one(y, digit_a=3, digit_b=5):
    # digit_a -> -1, digit_b -> +1
    y = y.clone()
    y[y == digit_a] = -1
    y[y == digit_b] = +1
    return y.float()

# ----------------------------
# Model: 2-layer MLP
# ----------------------------
class MLP(nn.Module):
    def __init__(self, width=256, act="tanh"):
        super().__init__()
        self.fc1 = nn.Linear(784, width)
        self.fc2 = nn.Linear(width, 1)
        self.act = act

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.tanh(x) if self.act == "tanh" else F.relu(x)
        x = self.fc2(x).squeeze(1)  # scalar per sample
        return x

def weight_norm(model):
    # sum of Frobenius norms of weight matrices (bias excluded)
    w1 = model.fc1.weight
    w2 = model.fc2.weight
    return (w1.norm(p='fro') + w2.norm(p='fro')).item()

@torch.no_grad()
def eval_metrics(model, loader, digit_a=0, digit_b=1, gammas=(0.0, 0.5, 1.0)):
    model.eval()
    all_margins = []
    all_correct = []
    for x, y in loader:
        x = x.to(DEVICE)
        y = to_pm_one(y.to(DEVICE), digit_a, digit_b)
        f = model(x)
        pred = torch.sign(f)
        correct = (pred == y).float()
        margins = y * f
        all_correct.append(correct.cpu())
        all_margins.append(margins.cpu())

    correct = torch.cat(all_correct).mean().item()
    margins = torch.cat(all_margins).numpy()
    err01 = 1.0 - correct
    q10 = float(np.quantile(margins, 0.10))
    med = float(np.median(margins))

    frac_le = {g: float(np.mean(margins <= g)) for g in gammas}
    return {"err01": err01, "margin_q10": q10, "margin_median": med, **{f"frac_margin_le_{g}": frac_le[g] for g in gammas}}

def train_one(width, weight_decay, seed=0, act="tanh", digit_a=0, digit_b=1,
              n_train=None, n_test=None, epochs=20, batch_size=128, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds, test_ds = load_mnist_binary(digit_a, digit_b, n_train=n_train, n_test=n_test, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = MLP(width=width, act=act).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # MSE loss with targets in {-1, +1}
    for _ in tqdm(range(epochs), leave=False):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = to_pm_one(y.to(DEVICE), digit_a, digit_b)
            f = model(x)
            loss = ((f - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    wn = weight_norm(model)
    train_m = eval_metrics(model, train_loader, digit_a, digit_b)
    test_m  = eval_metrics(model, test_loader,  digit_a, digit_b)

    return {
        "width": width,
        "weight_decay": weight_decay,
        "seed": seed,
        "act": act,
        "weight_norm": wn,
        "train_err01": train_m["err01"],
        "test_err01": test_m["err01"],
        "train_margin_q10": train_m["margin_q10"],
        "test_margin_q10": test_m["margin_q10"],
        "train_margin_median": train_m["margin_median"],
        "test_margin_median": test_m["margin_median"],
        "train_frac_margin_le_0.0": train_m["frac_margin_le_0.0"],
        "train_frac_margin_le_0.5": train_m["frac_margin_le_0.5"],
        "train_frac_margin_le_1.0": train_m["frac_margin_le_1.0"],
    }

if __name__ == "__main__":
    widths = [16, 64, 256, 1024]
    wds = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
    seeds = [0, 1, 2]  #60 trainings

    rows = []
    for width in widths:
        for wd in wds:
            for seed in seeds:
                row = train_one(width, wd, seed=seed, act="tanh",
                                digit_a=3, digit_b=5,
                                n_train=12000, n_test=2000,  
                                epochs=15)
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("mnist_bartlett_experiment.csv", index=False)
    print(df.groupby(["width", "weight_decay"])[["test_err01", "weight_norm", "train_margin_q10"]].mean())
