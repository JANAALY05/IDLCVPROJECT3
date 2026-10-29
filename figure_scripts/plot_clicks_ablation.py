import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
n = [1, 3, 5, 10, 25, 50]
res = {
 "random": {"dice": [0.813,0.840,0.875,0.877,0.891,0.883], "sens": [0.742,0.786,0.832,0.865,0.868,0.876], "spec": [0.963,0.962,0.969,0.952,0.965,0.953]},
 "margin": {"dice": [0.775,0.792,0.876,0.867,0.878,0.879], "sens": [0.649,0.687,0.871,0.859,0.880,0.891], "spec": [0.989,0.979,0.948,0.946,0.946,0.940]},
}
col = {"random": "#2a78d6", "margin": "#eb6834"}
mk = {"random": "o", "margin": "s"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, key, title in zip(axes, ["dice", "sens", "spec"], ["Dice", "Sensitivity", "Specificity"]):
    for s in ["random", "margin"]:
        ax.plot(n, res[s][key], color=col[s], lw=2, marker=mk[s], ms=7,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=f"{s} clicks")
    full = {"dice": 0.883, "sens": 0.829, "spec": 0.979}[key]
    ax.axhline(full, color="#52514e", lw=1.2, ls="--", label="full masks (U-Net + BCE)")
    ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels(n)
    ax.set_xlabel("clicks per image (each: positive and negative)", color="#52514e")
    ax.set_title(f"{title} on PH2 test set", loc="left", fontsize=11)
    ax.grid(axis="y", color="#e5e4e0", lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e")
axes[0].legend(frameon=False, fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("figures/clicks_ablation.png", dpi=200, facecolor="#fcfcfb")
plt.savefig("figures/clicks_ablation.pdf", facecolor="#fcfcfb")
print("saved")
