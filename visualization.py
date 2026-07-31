import matplotlib.pyplot as plt
from config import *

plt.style.use(PLOT_STYLE)


# ==========================================================
# Save Figure
# ==========================================================
def save_figure(path):
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight"
    )
    plt.close()


# ==========================================================
# Plot Signal
# ==========================================================
def plot_signal(x, y, title, xlabel, ylabel, save_path, color="blue"):
    plt.figure(figsize=(12, 4))
    plt.plot(x, y, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    save_figure(save_path)


# ==========================================================
# Plot Comparison
# ==========================================================
def plot_compare(x, y1, y2, label1, label2, title, save_path):
    plt.figure(figsize=(12, 4))
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2)
    plt.legend()
    plt.grid(True)
    plt.title(title)
    save_figure(save_path)


# ==========================================================
# Histogram
# ==========================================================
def plot_histogram(data, bins, title, save_path):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins)
    plt.grid(True)
    plt.title(title)
    save_figure(save_path)
