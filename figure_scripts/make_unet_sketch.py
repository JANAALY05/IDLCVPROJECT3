import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_ENC = "#cfe0f5"
C_DEC = "#f8dcc0"
C_BOT = "#d9d2ef"
EDGE = "#333333"

ch = [32, 64, 128, 256, 512]
ph2 = ["192×256", "96×128", "48×64", "24×32", "12×16"]
drv = ["512×512", "256×256", "128×128", "64×64", "32×32"]

fig, ax = plt.subplots(figsize=(14.5, 7.6))
ax.set_xlim(-0.3, 14.6)
ax.set_ylim(-1.0, 7.4)
ax.axis("off")

W = 1.55
H = 0.95
ys = [6.0, 4.6, 3.2, 1.8, 0.4]          # level heights
x_enc = [1.1, 2.3, 3.5, 4.7]            # encoder x per level
x_bot = 6.95
x_dec = [12.4, 11.2, 10.0, 8.8]           # decoder x per level (level 0 rightmost)


def box(x, y, color, title, c, i):
    ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=EDGE, lw=1))
    ax.text(x, y + 0.22, f"{title}", ha="center", va="center", fontsize=9, weight="bold")
    ax.text(x, y - 0.05, f"{c} ch", ha="center", va="center", fontsize=8.5)
    ax.text(x, y - 0.29, f"{ph2[i]} | {drv[i]}", ha="center", va="center", fontsize=7.5, color="#444")


def arrow(x0, y0, x1, y1, color, style="-|>", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.4, linestyle=ls,
                                shrinkA=0, shrinkB=0))


# input / output
ax.text(x_enc[0] - 1.15, ys[0], "input\n3 ch", ha="center", va="center", fontsize=8.5)
arrow(x_enc[0] - 0.8, ys[0], x_enc[0] - W / 2, ys[0], "#2b6cb0")

for i in range(4):
    box(x_enc[i], ys[i], C_ENC, f"enc{i}", ch[i], i)
    box(x_dec[i], ys[i], C_DEC, f"dec{i}", ch[i], i)
    # skip connection
    arrow(x_enc[i] + W / 2, ys[i], x_dec[i] - W / 2, ys[i], "#888", ls="--")
    ax.text((x_enc[i] + x_dec[i]) / 2, ys[i] + 0.1, f"skip: concat {ch[i]}+{ch[i]} = {2*ch[i]} ch",
            ha="center", fontsize=7.5, color="#666")

box(x_bot, ys[4], C_BOT, "bottleneck", ch[4], 4)
box(x_bot + 1.6, ys[4], C_BOT, "", ch[4], 4)  # placeholder removed below
ax.patches[-1].remove()
for t in ax.texts[-3:]:
    t.remove()

# down arrows (max-pool)
for i in range(3):
    arrow(x_enc[i] + 0.2, ys[i] - H / 2, x_enc[i + 1] - 0.2, ys[i + 1] + H / 2, "#c0392b")
arrow(x_enc[3] + 0.3, ys[3] - H / 2, x_bot - 0.4, ys[4] + H / 2, "#c0392b")

# up arrows (transposed conv)
arrow(x_bot + 0.4, ys[4] + H / 2, x_dec[3] - 0.3, ys[3] - H / 2, "#27ae60")
for i in range(3, 0, -1):
    arrow(x_dec[i] + 0.2, ys[i] + H / 2, x_dec[i - 1] - 0.2, ys[i - 1] - H / 2, "#27ae60")

# output
ax.add_patch(FancyBboxPatch((x_dec[0] + 1.45 - 0.45, ys[0] - H / 2), 0.9, H,
                            boxstyle="round,pad=0.02,rounding_size=0.08", fc="#e8e8e8", ec=EDGE, lw=1))
ax.text(x_dec[0] + 1.45, ys[0] + 0.22, "output", ha="center", fontsize=9, weight="bold")
ax.text(x_dec[0] + 1.45, ys[0] - 0.05, "1 ch", ha="center", fontsize=8.5)
ax.text(x_dec[0] + 1.45, ys[0] - 0.29, "logits", ha="center", fontsize=7.5, color="#444")
arrow(x_dec[0] + W / 2, ys[0], x_dec[0] + 1.0, ys[0], "#8e44ad")

# legend
lx, ly = 0.3, -0.35
items = [("#333", "-", "DoubleConv: (3×3 conv → BN → ReLU) × 2, padding 1"),
         ("#c0392b", "-", "2×2 max-pool (½ size)"),
         ("#27ae60", "-", "2×2 transposed conv (×2 size, ½ channels)"),
         ("#888", "--", "skip connection"),
         ("#8e44ad", "-", "1×1 conv")]
pos = [(0.3, -0.3), (5.6, -0.3), (9.0, -0.3), (0.3, -0.75), (5.6, -0.75)]
for (c, ls, label), (x, ly) in zip(items, pos):
    if label.startswith("DoubleConv"):
        ax.add_patch(FancyBboxPatch((x, ly - 0.12), 0.3, 0.24, boxstyle="round,pad=0.01", fc=C_ENC, ec=EDGE, lw=0.8))
    else:
        ax.plot([x, x + 0.35], [ly, ly], color=c, lw=1.6, ls=ls)
    ax.text(x + 0.45, ly, label, va="center", fontsize=7.8)

ax.text(7.15, 7.15, "U-Net used for PH2 and DRIVE", ha="center", fontsize=13, weight="bold")
ax.text(7.15, 6.8, "each box: name · channels · feature-map size H×W for PH2 | DRIVE   (7.76 M parameters)",
        ha="center", fontsize=9, color="#444")

plt.savefig("figures/unet_sketch.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("figures/unet_sketch.pdf", bbox_inches="tight", facecolor="white")
print("saved")
