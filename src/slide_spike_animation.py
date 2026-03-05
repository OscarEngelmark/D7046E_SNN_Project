import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import datasets, transforms
from paths import PLOTS_DIR

from MNIST_snnTorch_pipeline import image_to_spikes

# ========================== CONFIG ==========================
DIRECTION     = 'LR'          # 'LR' or 'RL'
FPS           = 8
DIGIT_IDX     = 5
GIF_NAME      = f'{PLOTS_DIR}/mnist_sliding_and_spikes_{DIRECTION}.gif'
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
spikes, shifts, placed_images = image_to_spikes(img_tensor, direction=DIRECTION)

# ── Figure with two subplots (stacked vertically) ─────────────
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1,
    figsize=(7.2, 9.6),     # taller to fit both nicely
    dpi=140,
    gridspec_kw={'hspace': 0.12}
)

main_title = fig.suptitle(
    "",
    fontsize=15,
    y=0.98,
    color="white"
)

for ax in (ax_top, ax_bottom):
    for spine in ax.spines.values():
        spine.set_color('#6e7681')
        spine.set_linewidth(1.2)

# ── Shared settings ───────────────────────────────────────────
yy, xx = np.meshgrid(np.arange(28), np.arange(28))

# ── Top: Sliding digit ────────────────────────────────────────
ax_top.set_title("Input image sliding over receptor grid", fontsize=12, pad=6, color="white")

img_plot = ax_top.imshow(
    placed_images[0],
    cmap='gray',
    vmin=0, vmax=1,
    alpha=0.9,
    origin='upper'
)

ax_top.scatter(
    xx.ravel(), yy.ravel(),
    s=12,
    c='cyan',
    marker='o',
    alpha=0.5,
    zorder=10
)

for pos in range(29):
    ax_top.axhline(pos - 0.5, color='#8b949e', lw=0.45, alpha=0.35)
    ax_top.axvline(pos - 0.5, color='#8b949e', lw=0.45, alpha=0.35)

ax_top.set_xticks([])
ax_top.set_yticks([])
ax_top.set_aspect('equal')

# ── Bottom: Spiking activity ──────────────────────────────────
ax_bottom.set_title("Spiking receptor neurons (ON events)", fontsize=12, pad=6, color="white")

# Resting neurons
ax_bottom.scatter(
    xx.ravel(), yy.ravel(),
    s=16,
    c='darkcyan',
    marker='o',
    alpha=0.45,
    zorder=5
)

# Active spikes layer
active_scatter = ax_bottom.scatter(
    [], [],
    s=75,
    c='lime',
    marker='o',
    alpha=0.95,
    zorder=10
)

for pos in range(29):
    ax_bottom.axhline(pos - 0.5, color='#8b949e', lw=0.45, alpha=0.35)
    ax_bottom.axvline(pos - 0.5, color='#8b949e', lw=0.45, alpha=0.35)

ax_bottom.set_xticks([])
ax_bottom.set_yticks([])
ax_bottom.set_aspect('equal')
ax_bottom.set_facecolor('#0a0e14')
fig.patch.set_facecolor('#0a0e14')   # or keep white if preferred

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
    main_title.set_text(f"t = {frame}  |  shift = {shifts[frame]}")

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
anim.save(GIF_NAME, writer='pillow', fps=FPS, dpi=140)
print("GIF saved successfully.")