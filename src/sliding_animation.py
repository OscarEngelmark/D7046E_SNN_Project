import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import datasets, transforms
from paths import PLOTS_DIR

from MNIST_snnTorch_pipeline import image_to_spikes

# ========================== CONFIG ==========================
DIRECTION = 'LR'          # Change to 'RL' for the opposite direction
FPS = 15                  # Frames per second (12–15 looks smooth)
GIF_NAME = f'{PLOTS_DIR}/mnist_sliding_{DIRECTION}.gif'
DIGIT_IDX = 5
# ===========================================================

# Load one random MNIST image
mnist = datasets.MNIST(
    root='./mnist',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

img_tensor, label = mnist[DIGIT_IDX]
print(f"Using MNIST digit {label} (index {DIGIT_IDX}) — Direction: {DIRECTION}")

# Generate the sliding frames (reuses your existing function)
spikes, shifts, placed_images = image_to_spikes(img_tensor, direction=DIRECTION)

# ================ Create the Animation ================
fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=140)
fig.suptitle(f"MNIST Digit Sliding Over 28×28 Receptor Grid — {DIRECTION}", fontsize=13, y=0.93)

# Static neuron grid (cyan dots — exactly like sliding_test.py)
yy, xx = np.meshgrid(np.arange(28), np.arange(28))

ax.scatter(
    xx.ravel(), yy.ravel(),
    s=12,
    c='cyan',
    marker='o',
    alpha=1,
)

# Dynamic image overlay (will be updated each frame)
# Use this (stretches contrast + makes it much more visible):
img_plot = ax.imshow(
    placed_images[0],
    cmap='gray',
    vmin=0, vmax=1,
    alpha=0.6,
    origin='upper'
)

# Light grid lines for reference
for pos in range(29):
    ax.axhline(pos - 0.5, color='gray', lw=0.35, alpha=0.25)
    ax.axvline(pos - 0.5, color='gray', lw=0.35, alpha=0.25)

ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')

# Title that updates with timestep
title_text = ax.set_title(f"t = 0  |  shift = {shifts[0]}", fontsize=12)

def animate(frame: int):
    frame_data = placed_images[frame]
    img_plot.set_data(frame_data)
    title_text.set_text(f"t = {frame}  |  shift = {shifts[frame]}")
    return img_plot, title_text

# Build the animation
anim = animation.FuncAnimation(
    fig, animate,
    frames=len(placed_images),      # 55 frames
    interval=1000 // FPS,
    blit=True,
    repeat=True
)

# Save as GIF (requires Pillow, which is installed by default with matplotlib)
print(f"Saving GIF → '{GIF_NAME}' ({len(placed_images)} frames @ {FPS} fps)...")
anim.save(GIF_NAME, writer='pillow', fps=FPS)
print("GIF created successfully!")

# Optional: show the last frame (or comment out if you only want the GIF)
plt.show()