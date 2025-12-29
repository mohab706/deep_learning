import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


def get_class_weights(dataset, device):
    targets = [label for _, label in dataset.samples]
    counts = np.bincount(targets)
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    return weights.to(device)


def evaluate_model(model, loader, criterion, device):
    model.eval()
    losses, preds, labels = [], [], []

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            losses.append(loss.item())
            preds.extend(outputs.argmax(1).cpu().numpy())
            labels.extend(targets.cpu().numpy())

    acc = accuracy_score(labels, preds)
    return np.mean(losses), acc, labels, preds


def plot_confusion_matrix(labels, preds, class_names, save_path=None):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def save_model(model, path):
    torch.save(model.state_dict(), path)
