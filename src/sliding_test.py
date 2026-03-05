from MNIST_snnTorch_pipeline import image_to_spikes
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# ────────────────────────────────────────────────────────────────
#  Load one random MNIST image
# ────────────────────────────────────────────────────────────────

mnist = datasets.MNIST(
    root='./mnist',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

idx = np.random.randint(0, len(mnist))           # random sample
img_tensor, label = mnist[idx]
print(f"Using MNIST digit: {label}  (index {idx})")

# ────────────────────────────────────────────────────────────────
#  Generate sliding data (we'll use placed_images for plotting)
# ────────────────────────────────────────────────────────────────

spikes, shifts, placed_images = image_to_spikes(img_tensor, direction='LR')

# ────────────────────────────────────────────────────────────────
#  Plot selected timesteps
# ────────────────────────────────────────────────────────────────

timesteps_to_show = [0, 13, 27, 40, 54]
n_plots = len(timesteps_to_show)

fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4.5), dpi=100)
fig.suptitle(f"MNIST digit sliding over 28×28 neuron grid  —  direction: Left → Right\n", y=.99)

for i, t in enumerate(timesteps_to_show):
    ax = axes[i]

    # Neuron grid (small black dots)
    yy, xx = np.meshgrid(np.arange(28), np.arange(28))
    ax.scatter(xx.ravel(), yy.ravel(), s=8, c='cyan', marker='o', alpha=0.5)

    # Placed image (slightly transparent)
    img_visible = placed_images[t]
    ax.imshow(img_visible, cmap='gray', alpha=0.7, origin='upper')

    ax.set_title(f"t = {t}\nshift = {shifts[t]}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    # Light grid lines for reference
    for pos in range(29):
        ax.axhline(pos - 0.5, color='gray', lw=0.4, alpha=0.3)
        ax.axvline(pos - 0.5, color='gray', lw=0.4, alpha=0.3)

plt.tight_layout()
plt.show()