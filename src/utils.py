import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import warnings

warnings.filterwarnings("ignore")


def plot_ecg_with_background_cam(cam, ecg_signal):
    norm = Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(cam) + 1)
    y = [-0.1, 1.1]

    heatmap = np.tile(cam, (1, 1))
    mesh = ax.pcolormesh(x, y, heatmap, cmap="jet", shading="flat", alpha=0.8)

    ax.plot(ecg_signal, label="ECG Signal", color="black", linewidth=2)
    ax.set_title("ECG Signal with Grad-CAM Heatmap")
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")
    plt.colorbar(mesh, label="Grad-CAM Importance", ax=ax)
    ax.legend()
    plt.show()


def plot_ecg_with_line_cam(cam, ecg_signal):
    cmap = "jet"
    norm = Normalize(vmin=0, vmax=1)
    colors = get_cmap(cmap)(norm(cam))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(ecg_signal) - 1):
        plt.plot(
            [i, i + 1], [ecg_signal[i], ecg_signal[i + 1]], color=colors[i], linewidth=2
        )

    ax.set_title("ECG Signal with Grad-CAM Heatmap")
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")
    plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), label="Grad-CAM Importance", ax=ax
    )
    ax.legend()
    plt.show()


def validate(model, test_dataloader):
    acc, f1, r, p = 0, 0, 0, 0
    model.to("cpu")

    for x, y in test_dataloader:
        y_pred = model(x).argmax(dim=1).cpu()
        acc += accuracy_score(y, y_pred)
        f1 += f1_score(y, y_pred, average="weighted")
        r += recall_score(y, y_pred, average="weighted")
        p += precision_score(y, y_pred, average="weighted")

    print(f"Average Accuracy: {acc/len(test_dataloader)*100:.2f}%")
    print(f"Average F1-score: {f1/len(test_dataloader):.4f}")
    print(f"Average Recall: {r/len(test_dataloader):.4f}")
    print(f"Average precision: {p/len(test_dataloader):.4f}")
