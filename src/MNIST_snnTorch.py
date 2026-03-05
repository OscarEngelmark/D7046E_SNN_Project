import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

import snntorch as snn
from snntorch import surrogate

import numpy as np
import matplotlib.pyplot as plt

# ── Two-layer LIF SNN ─────────────────────────────────────────────────────────
class DirectionSNN(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(28, 64)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc2 = nn.Linear(64, 2)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        counts = torch.zeros(x.shape[0], 2)
        for t in range(x.shape[1]):  # 28 timesteps
            spk1, mem1 = self.lif1(self.fc1(x[:, t, :]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            counts += spk2
        return counts

def image_to_spikes(image_tensor, direction='LR', threshold=0.15):
    """
    Scroll image column-by-column (28 timesteps) across 28 receptors.
    Fire an ON-spike when pixel intensity increases by >= threshold.
    Returns ndarray shape (28 timesteps, 28 receptors).
    """
    img  = image_tensor.squeeze().numpy()
    cols = range(28) if direction == 'LR' else range(27, -1, -1)
    spikes = np.zeros((28, 28), dtype=np.float32)
    prev   = np.zeros(28)
    for t, c in enumerate(cols):
        delta      = img[:, c] - prev
        spikes[t]  = (delta >= threshold).astype(np.float32)
        prev       = img[:, c]
    return spikes

def main():
    # ── build balanced dataset ────────────────────────────────────────────────────
    mnist_data = datasets.MNIST(
        root='./mnist',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    N_PER_CLASS = 500  # samples per direction (L→R and R→L)
    N_TOTAL = N_PER_CLASS * 2

    X_data = np.zeros((N_TOTAL, 28, 28), dtype=np.float32)  # (N, T, R)
    y_data = np.zeros(N_TOTAL, dtype=np.int64)  # 0=LR, 1=RL

    img_pool = [mnist_data[i][0] for i in range(N_PER_CLASS)]  # first N images

    for i, img in enumerate(img_pool):
        X_data[i] = image_to_spikes(img, 'LR')
        y_data[i] = 0  # label: left→right

        X_data[i + N_PER_CLASS] = image_to_spikes(img, 'RL')
        y_data[i + N_PER_CLASS] = 1  # label: right→left

    # shuffle
    rng = np.random.default_rng(42)
    perm = rng.permutation(N_TOTAL)
    X_data, y_data = X_data[perm], y_data[perm]

    # train / test split (80 / 20)
    split = int(0.8 * N_TOTAL)
    X_train, X_test = X_data[:split], X_data[split:]
    y_train, y_test = y_data[:split], y_data[split:]

    torch.manual_seed(42)
    device = torch.device('cpu')

    X_tr = torch.tensor(X_train)
    y_tr = torch.tensor(y_train)
    X_te = torch.tensor(X_test)
    y_te = torch.tensor(y_test)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=32, shuffle=False)

    model = DirectionSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()

    # ── Training (20 epochs) ──────────────────────────────────────────────────────
    EPOCHS = 20
    train_losses, train_accs = [], []

    print(f"{'Epoch':>6}  {'Loss':>8}  {'Train Acc':>10}")
    print("-" * 30)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = correct = total = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        if epoch % 4 == 0 or epoch == 1:
            print(f"{epoch:>6}  {epoch_loss:>8.4f}  {epoch_acc:>9.1%}")
    print("Done.")

    import matplotlib.pyplot as plt

    # ── test-set evaluation ───────────────────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            all_preds.extend(out.argmax(1).numpy())
            all_labels.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = (all_preds == all_labels).mean()
    print(f"Test accuracy: {test_acc:.1%}  ({(all_preds == all_labels).sum()}/{len(all_labels)} correct)")

    # ── confusion matrix ──────────────────────────────────────────────────────────
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_labels, all_preds)
    except ImportError:
        cm = np.zeros((2, 2), int)
        for t, p in zip(all_labels, all_preds):
            cm[t, p] += 1

    # ── plots ─────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Training curves (use full list length, not hardcoded EPOCHS)
    ax = axes[0]
    ep = range(1, len(train_losses) + 1)
    ax.plot(ep, train_losses, 'b-o', markersize=4, label='Loss')
    ax2 = ax.twinx()
    ax2.plot(ep, [a * 100 for a in train_accs], 'r--s', markersize=4, label='Acc (%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss', color='b')
    ax2.set_ylabel('Accuracy %', color='r')
    ax.set_title('Training Curves')
    lines1, _ = ax.get_legend_handles_labels()
    lines2, _ = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, ['Loss', 'Accuracy'], loc='upper right')

    # 2. Confusion matrix
    ax = axes[1]
    im = ax.imshow(cm, cmap='Blues', vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=16, color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred L→R', 'Pred R→L'])
    ax.set_yticklabels(['True L→R', 'True R→L'])
    ax.set_title('Confusion Matrix (test set)')
    plt.colorbar(im, ax=ax)

    # 3. Example predictions
    ax = axes[2]
    correct_idx = np.where(all_preds == all_labels)[0][:5]
    incorrect_idx = np.where(all_preds != all_labels)[0][:5]
    show_idx = np.concatenate([correct_idx, incorrect_idx])[:8]

    labels_str = {0: 'L→R', 1: 'R→L'}
    bar_colors = ['green' if all_preds[i] == all_labels[i] else 'red' for i in show_idx]
    bar_labels = [f"T:{labels_str[all_labels[i]]}\nP:{labels_str[all_preds[i]]}" for i in show_idx]
    ax.bar(range(len(show_idx)), [1] * len(show_idx), color=bar_colors, edgecolor='k')
    ax.set_xticks(range(len(show_idx)))
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_yticks([])
    ax.set_title('Sample Predictions (green=correct, red=wrong)')

    plt.suptitle(f'SNN Motion Direction Classifier on MNIST  —  Test Accuracy: {test_acc:.1%}',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()


