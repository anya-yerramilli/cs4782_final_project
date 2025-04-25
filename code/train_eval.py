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


def train(
    model,
    train_loader,
    val_dataset,  # NOTE: pass the *dataset*, not the loader
    epochs: int = 90,
    lr: float = 0.1,
    device: str = "cuda",
):
    """
    Supervised pre-training of the SimpleShot backbone on the 64 base classes.

    * Uses SGD with momentum 0.9, weight-decay 5e-4.
    * LR drops ×0.1 at epochs 45 and 66 (SimpleShot paper schedule).
    * Early-stops on 5-way / 1-shot validation accuracy (CL2N features).
    * Restores the best-performing weights at the end.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[45, 66], gamma=0.1
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):

        # ─────────────────────── training ────────────────────────────
        model.train()
        running_loss = 0.0
        running_hits = 0
        running_total = 0

        for inputs, targets in tqdm(
            train_loader, desc=f"Epoch {epoch:3d}/{epochs} – train", leave=False
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            running_total += targets.size(0)
            running_hits += preds.eq(targets).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * running_hits / running_total
        print(
            f"Epoch {epoch:3d} | train loss {train_loss:.3f} | "
            f"train acc {train_acc:5.2f}%"
        )

        # ─────────────────────── validation episodes ─────────────────
        val_acc = evaluate_few_shot(
            model,
            val_dataset,  # ← dataset, not loader
            n_way=5,
            k_shot=1,
            n_tasks=600,
            feature_transform="CL2N",  # Center + L2 (paper's best)
            device=device,
        )
        print(f"Epoch {epoch:3d} | val 5-way 1-shot acc {val_acc:5.2f}%")

        # ─────────────────────── checkpoint best ─────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            print(f"  🏆  new best model ({best_val_acc:5.2f}%) saved")

        scheduler.step()

    # ─────────────────── restore best weights ────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model with val acc {best_val_acc:5.2f}%")

    return model


def evaluate_few_shot(
    model,
    data_loader,
    n_way=5,
    k_shot=1,
    n_tasks=10000,
    feature_transform="CL2N",
    device="cuda",
):
    """
    Evaluate model using few-shot learning with nearest neighbor/centroid
    feature_transform: 'UN' (unnormalized), 'L2N' (L2-normalized), 'CL2N' (centered L2-normalized)
    """
    model.eval()
    accuracies = []

    # Get all features and labels from the dataset
    all_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Extracting features"):
            inputs = inputs.to(device)
            features = model.feature_extraction(inputs)
            features = torch.flatten(features, start_dim=1)
            all_features.append(features.cpu())
            all_labels.append(labels)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute support mean for centering if using CL2N
    if feature_transform == "CL2N":
        support_mean = torch.mean(all_features, dim=0, keepdim=True)
    else:
        support_mean = None

    # Sample n_tasks tasks
    classes = torch.unique(all_labels).tolist()

    for _ in tqdm(range(n_tasks), desc=f"Evaluating {n_way}-way {k_shot}-shot"):
        # Sample n_way classes
        sampled_classes = random.sample(classes, n_way)

        # For each sampled class, select k_shot examples as support and 15 examples as query
        support_features = []
        support_labels = []
        query_features = []
        query_labels = []

        for i, cls in enumerate(sampled_classes):
            # Get indices of all examples of this class
            cls_indices = torch.nonzero(all_labels == cls).squeeze()
            # Sample k_shot + 15 examples (or all available if less)
            num_examples = min(cls_indices.size(0), k_shot + 15)
            selected_indices = cls_indices[
                torch.randperm(cls_indices.size(0))[:num_examples]
            ]

            # First k_shot examples go to support set
            support_idx = selected_indices[:k_shot]
            support_feat = all_features[support_idx]
            support_features.append(support_feat)
            support_labels.extend(
                [i] * support_idx.size(0)
            )  # Use 0 to n_way-1 as labels

            # Remaining examples go to query set (up to 15)
            query_idx = selected_indices[k_shot : k_shot + 15]
            query_feat = all_features[query_idx]
            query_features.append(query_feat)
            query_labels.extend([i] * query_idx.size(0))  # Use 0 to n_way-1 as labels

        support_features = torch.cat(support_features, dim=0)
        support_labels = torch.tensor(support_labels)
        query_features = torch.cat(query_features, dim=0)
        query_labels = torch.tensor(query_labels)

        # Apply feature transformation
        if feature_transform == "L2N" or feature_transform == "CL2N":
            if feature_transform == "CL2N" and support_mean is not None:
                # Center features
                support_features = support_features - support_mean
                query_features = query_features - support_mean

            # L2-normalize features
            support_features = support_features / (
                torch.norm(support_features, dim=1, keepdim=True) + 1e-8
            )
            query_features = query_features / (
                torch.norm(query_features, dim=1, keepdim=True) + 1e-8
            )

        # Compute class centroids for support features (for multi-shot scenario)
        centroids = []
        for i in range(n_way):
            mask = support_labels == i
            class_features = support_features[mask]
            centroid = torch.mean(class_features, dim=0)
            centroids.append(centroid)
        centroids = torch.stack(centroids)

        # Compute distances and get predictions
        num_queries = query_features.size(0)
        num_centroids = centroids.size(0)
        distances = torch.zeros(num_queries, num_centroids)

        for i in range(num_queries):
            for j in range(num_centroids):
                distances[i, j] = torch.sum((query_features[i] - centroids[j]) ** 2)

        _, predicted = torch.min(distances, dim=1)
        accuracy = 100.0 * (predicted == query_labels).float().mean().item()
        accuracies.append(accuracy)

    return np.mean(accuracies)


def main():
    # Get datasets
    train_dataset, val_dataset, test_dataset = get_datasets()
    print("got data sets")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    print("created data loader")

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = SimpleShot(
        input_dim=84, hidden_dim=64, num_classes=64, l2norm=False, support=None
    )
    model = model.to(device)
    print("init model")

    # Train model
    model = train(model, train_loader, val_dataset, epochs=90, lr=0.1, device=device)
    print("done training!")

    # Evaluate on test set using 5-way 1-shot and 5-way 5-shot tasks
    print("Evaluating on test set...")

    # Try different feature transformations
    for transform in ["UN", "L2N", "CL2N"]:
        print(f"\nFeature transformation: {transform}")

        # 5-way 1-shot
        one_shot_acc = evaluate_few_shot(
            model,
            test_loader,
            n_way=5,
            k_shot=1,
            n_tasks=10000,
            feature_transform=transform,
            device=device,
        )
        print(f"5-way 1-shot accuracy: {one_shot_acc:.2f}%")

        # 5-way 5-shot
        five_shot_acc = evaluate_few_shot(
            model,
            test_loader,
            n_way=5,
            k_shot=5,
            n_tasks=10000,
            feature_transform=transform,
            device=device,
        )
        print(f"5-way 5-shot accuracy: {five_shot_acc:.2f}%")


if __name__ == "__main__":
    main()
