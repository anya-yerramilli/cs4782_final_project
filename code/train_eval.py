import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
from models import SimpleShot
from data_collector import get_datasets


import copy
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch


def train(model, train_ds, val_ds, device="cuda"):
    loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)

    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[45, 66], gamma=0.1)

    best_acc, best_state = 0.0, None

    for epoch in range(1, 91):
        # ---- supervised CE on 64 base classes ----
        model.train()
        hits, n, loss_sum = 0, 0, 0.0
        for x, y in tqdm(loader, leave=False, desc=f"Epoch {epoch:3d} train"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            loss_sum += loss.item()
            hits += logits.argmax(1).eq(y).sum().item()
            n += y.size(0)

        print(
            f"Epoch {epoch:3d} | train loss {loss_sum/len(loader):.3f} "
            f"| train acc {100*hits/n:5.2f}%"
        )

        # ---- few-shot validation every 5 epochs ----
        if epoch % 5 == 0 or epoch == 90:
            val_acc = evaluate_few_shot(
                model, val_ds, device=device, n_way=5, k_shot=1, n_tasks=600
            )
            print(f"Epoch {epoch:3d} | val 5-way/1-shot {val_acc:5.2f}%")
            if val_acc > best_acc:
                best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())

        sched.step()

    # restore best / compute global mean --------------------------------------
    model.load_state_dict(best_state)
    base_loader = DataLoader(train_ds, batch_size=256, shuffle=False)
    with torch.no_grad():
        feats = [
            model.feature_extraction(x.to(device)).view(x.size(0), -1).cpu()
            for x, _ in base_loader
        ]
    model.support_mean = torch.cat(feats).mean(0, keepdim=True)
    print(f"\nBest val acc {best_acc:5.2f}% – support_mean stored.")

    return model


def evaluate_few_shot(model, dataset, n_way=5, k_shot=1, n_tasks=600, device="cuda"):
    """
    CL2N + nearest-centroid evaluation.
    """
    model.eval()
    feats, labels = [], []

    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f = model.feature_extraction(x).view(x.size(0), -1)
            feats.append(f.cpu())
            labels.append(y)
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    classes = labels.unique().tolist()

    # ---------- pre-centred / L2-normed features ----------
    if not hasattr(model, "support_mean") or model.support_mean is None:
        raise RuntimeError("model.support_mean is not set – run training first.")
    feats = feats - model.support_mean  # centre
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)  # L2

    accs = []
    rng = torch.Generator().manual_seed(42)
    for _ in tqdm(range(n_tasks), leave=False):
        sampled = random.sample(classes, n_way)

        sup_f, sup_l, qry_f, qry_l = [], [], [], []
        for new_lbl, c in enumerate(sampled):
            idx = (labels == c).nonzero(as_tuple=False).squeeze()
            perm = idx[torch.randperm(idx.numel(), generator=rng)]
            sup_idx, qry_idx = perm[:k_shot], perm[k_shot : k_shot + 15]

            sup_f.append(feats[sup_idx])
            sup_l += [new_lbl] * k_shot
            qry_f.append(feats[qry_idx])
            qry_l += [new_lbl] * qry_idx.numel()

        sup_f = torch.cat(sup_f)
        qry_f = torch.cat(qry_f)
        sup_l = torch.tensor(sup_l)
        qry_l = torch.tensor(qry_l)

        # centroids
        centroids = torch.stack([sup_f[sup_l == i].mean(0) for i in range(n_way)])
        dists = ((qry_f[:, None, :] - centroids[None]) ** 2).sum(-1)  # (q, n_way)
        pred = dists.argmin(1)
        accs.append((pred == qry_l).float().mean().item() * 100)

    return float(np.mean(accs))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, test_ds = get_datasets()

    model = SimpleShot(input_dim=84, hidden_dim=64, num_classes=64, l2norm=True).to(
        device
    )

    model = train(model, train_ds, val_ds, device=device)

    # ------------------- final test scores -------------------
    for shots in (1, 5):
        acc = evaluate_few_shot(
            model, test_ds, n_way=5, k_shot=shots, n_tasks=10000, device=device
        )
        print(f"TEST 5-way {shots}-shot  :  {acc:5.2f}%")


if __name__ == "__main__":
    main()
