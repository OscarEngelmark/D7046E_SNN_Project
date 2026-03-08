import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import datasets, transforms
from paths import PLOTS_DIR

from MNIST_snnTorch_pipeline import image_to_spikes_2D

# ========================== CONFIG ==========================
DIRECTION     = 'LR'          # 'LR' or 'RL'
FPS           = 8
DIGIT_IDX     = 5
GIF_NAME      = f'{PLOTS_DIR}/mnist_sliding_and_spikes_{DIRECTION}.gif'
AX_TITLE_FS   = 20
FIG_COLOR     = 'white'
AX_COLOR      = '#0a0e14'
DPI           = 180
# ===========================================================

# ── Load digit ────────────────────────────────────────────────
mnist = datasets.MNIST(
    root='./mnist',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

img_tensor, label = mnist[DIGIT_IDX]
print(f"Digit: {label} (idx {DIGIT_IDX})  Direction: {DIRECTION}")

# ── Generate data ─────────────────────────────────────────────
spikes, shifts, placed_images = image_to_spikes_2D(img_tensor, direction=DIRECTION)

# ── Figure with two subplots (stacked vertically) ─────────────
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8.5, 11), dpi=DPI)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.93, hspace=0.15)

fig.patch.set_facecolor(FIG_COLOR)

main_title = fig.suptitle(
    "",
    fontsize=18,           # larger → more visible
    fontweight='bold',
    color="black",
    y=0.96                 # move slightly down to avoid being cut off
)

# ── Shared settings ───────────────────────────────────────────
yy, xx = np.meshgrid(np.arange(28), np.arange(28))

# ── Top: Sliding digit ────────────────────────────────────────
ax_top.set_title("Input image sliding over receptor grid", fontsize=AX_TITLE_FS, pad=8, color="black")

img_plot = ax_top.imshow(placed_images[0], cmap='gray', vmin=0, vmax=1, alpha=0.9, origin='upper')

ax_top.scatter(xx.ravel(), yy.ravel(), s=12, c='cyan', marker='o', alpha=0.5, zorder=10)

ax_top.set_xticks([])
ax_top.set_yticks([])
ax_top.set_aspect('equal')

ax_top.set_facecolor(AX_COLOR)

# ── Bottom: Spiking activity ──────────────────────────────────
ax_bottom.set_title("Spiking receptor neurons (ON events)", fontsize=AX_TITLE_FS, pad=8, color="black")

# Resting neurons
ax_bottom.scatter(xx.ravel(), yy.ravel(), s=16, c='darkcyan', marker='o', alpha=0.45, zorder=5)

# Active spikes layer
active_scatter = ax_bottom.scatter([], [], s=75, c='lime', marker='o', alpha=0.95, zorder=10)

ax_bottom.set_xticks([])
ax_bottom.set_yticks([])
ax_bottom.set_aspect('equal')

ax_bottom.set_xlim(-0.5, 27.5)
ax_bottom.set_ylim(-0.5, 27.5)

ax_bottom.set_facecolor(AX_COLOR)

# ── Animation function ────────────────────────────────────────
def animate(frame: int):
    # Top: update image
    img_plot.set_data(placed_images[frame])

    # Bottom: update spikes
    spk = spikes[frame]
    active = spk > 0.5
    active_y, active_x = np.where(active)

    if len(active_x) > 0:
        offsets = np.column_stack((active_x, 27 - active_y))
    else:
        offsets = np.empty((0, 2), dtype=float)

    active_scatter.set_offsets(offsets)

    # Update title
    main_title.set_text(f"timestep = {frame}")

    return img_plot, active_scatter, main_title

# ── Create & save animation ───────────────────────────────────
anim = animation.FuncAnimation(
    fig, animate,
    frames=len(spikes),
    interval=1000 // FPS,
    blit=True,
    repeat=True
)

print(f"Saving → {GIF_NAME}  ({len(spikes)} frames @ {FPS} fps)")
anim.save(GIF_NAME, writer='pillow', fps=FPS, dpi=DPI)
print("GIF saved successfully.")