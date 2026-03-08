import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import datasets, transforms
import torch
from paths import PLOTS_DIR

def image_to_spikes(image_tensor, direction='LR', threshold=0.15):
    """
    Scroll a 28×28 image column-by-column and apply delta modulation.

    Returns
    -------
    spikes : np.ndarray, shape (28 timesteps, 28 receptors)
        1 where a receptor fires an ON-spike (intensity rose >= threshold).
    """
    img = image_tensor.squeeze().numpy()  # (28, 28)
    cols = range(28) if direction == 'LR' else range(27, -1, -1)
    cols = list(cols)

    spikes = np.zeros((28, 28), dtype=np.float32)  # (T, R)
    prev = np.zeros(28)
    for t, c in enumerate(cols):
        col_vals = img[:, c]
        delta = col_vals - prev
        spikes[t] = (delta >= threshold).astype(np.float32)
        prev = col_vals.copy()
    return spikes

# ========================== CONFIG ==========================
DIRECTION_TOP = 'RL'          # Top panel direction (change to 'RL' if you want)
FPS = 2
DIGIT_IDX = 7
GIF_NAME = f'{PLOTS_DIR}/mnist_column_spikes_dual_mirrored.gif'

AX_TITLE_FS = 18
FIG_COLOR = 'white'
AX_COLOR = '#0a0e14'
DPI = 180
CANVAS_WIDTH = 62
RECEPTOR_POS = CANVAS_WIDTH // 2
# ===========================================================

# ── Load digit ───────────────────────────────────────────────────────────────
mnist = datasets.MNIST(
    root='./mnist',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

img_tensor, label = mnist[DIGIT_IDX]
print(f"Digit: {label} (idx {DIGIT_IDX})")

img = img_tensor.squeeze().numpy()
img_flipped = np.fliplr(img).copy()          # horizontal mirror for bottom panel

# Generate spikes
spikes_top    = image_to_spikes(img_tensor, direction=DIRECTION_TOP)
direction_bottom = 'RL' if DIRECTION_TOP == 'LR' else 'LR'
spikes_bottom = image_to_spikes(torch.from_numpy(img_flipped)[None, None, ...], direction=direction_bottom)

# ── Precompute frames ────────────────────────────────────────────────────────
frames_top = []
frames_bottom = []

for t in range(28):
    # Top: original image
    canvas_top = np.zeros((28, CANVAS_WIDTH), dtype=np.float32)
    col_idx = t if DIRECTION_TOP == 'LR' else 27 - t
    left = RECEPTOR_POS - col_idx
    src_left = max(0, -left)
    dst_left = max(0, left)
    width = min(28 - src_left, CANVAS_WIDTH - dst_left)
    if width > 0:
        canvas_top[:, dst_left:dst_left + width] = img[:, src_left:src_left + width]
    frames_top.append(canvas_top)

    # Bottom: flipped image
    canvas_bottom = np.zeros((28, CANVAS_WIDTH), dtype=np.float32)
    col_idx_b = t if direction_bottom == 'LR' else 27 - t
    left_b = RECEPTOR_POS - col_idx_b
    src_left_b = max(0, -left_b)
    dst_left_b = max(0, left_b)
    width_b = min(28 - src_left_b, CANVAS_WIDTH - dst_left_b)
    if width_b > 0:
        canvas_bottom[:, dst_left_b:dst_left_b + width_b] = img_flipped[:, src_left_b:src_left_b + width_b]
    frames_bottom.append(canvas_bottom)

# ── Set up figure ────────────────────────────────────────────────────────────
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9.8, 11), dpi=DPI)
fig.subplots_adjust(left=0.06, right=0.94, bottom=0.05, top=0.91, hspace=0.01)
fig.patch.set_facecolor(FIG_COLOR)

main_title = fig.suptitle("", fontsize=21, fontweight='bold', color="black", y=0.96)

# ====================== TOP PANEL ======================
img_plot_top = ax_top.imshow(frames_top[0], cmap='gray', vmin=0, vmax=1, origin='upper', aspect='equal')

ax_top.scatter([RECEPTOR_POS]*28, np.arange(28), s=22, c='cyan', marker='o', alpha=0.55, zorder=11)

active_scatter_top = ax_top.scatter([], [], s=185, c='lime', marker='o',
                                    edgecolors='black', linewidth=1.1, alpha=0.97, zorder=15)

ax_top.set_xticks([])
ax_top.set_yticks([])
ax_top.set_facecolor(AX_COLOR)

# ====================== BOTTOM PANEL ======================
img_plot_bottom = ax_bottom.imshow(frames_bottom[0], cmap='gray', vmin=0, vmax=1, origin='upper', aspect='equal')

ax_bottom.scatter([RECEPTOR_POS]*28, np.arange(28), s=22, c='cyan', marker='o', alpha=0.55, zorder=11)

active_scatter_bottom = ax_bottom.scatter([], [], s=185, c='lime', marker='o',
                                          edgecolors='black', linewidth=1.1, alpha=0.97, zorder=15)

ax_bottom.set_xticks([])
ax_bottom.set_yticks([])
ax_bottom.set_facecolor(AX_COLOR)

# ── Animation function ───────────────────────────────────────────────────────
def animate(frame: int):
    img_plot_top.set_data(frames_top[frame])
    active_rows_top = np.where(spikes_top[frame] > 0.5)[0]
    if len(active_rows_top) > 0:
        offsets_top = np.column_stack((np.full_like(active_rows_top, RECEPTOR_POS), active_rows_top))
        active_scatter_top.set_offsets(offsets_top)
    else:
        active_scatter_top.set_offsets(np.empty((0, 2)))

    img_plot_bottom.set_data(frames_bottom[frame])
    active_rows_bottom = np.where(spikes_bottom[frame] > 0.5)[0]
    if len(active_rows_bottom) > 0:
        offsets_bottom = np.column_stack((np.full_like(active_rows_bottom, RECEPTOR_POS), active_rows_bottom))
        active_scatter_bottom.set_offsets(offsets_bottom)
    else:
        active_scatter_bottom.set_offsets(np.empty((0, 2)))

    return (img_plot_top, active_scatter_top, img_plot_bottom, active_scatter_bottom, main_title)

# ── Create and save animation ────────────────────────────────────────────────
print(f"Saving → {GIF_NAME}  (28 frames @ {FPS} fps)")

anim = animation.FuncAnimation(fig, animate, frames=28, interval=1000 // FPS, blit=False, repeat=True)

anim.save(GIF_NAME, writer='pillow', fps=FPS, dpi=DPI)
print("GIF saved successfully.")