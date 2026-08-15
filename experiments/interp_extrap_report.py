"""
experiments/interp_extrap_report.py
====================================
Phase 3e: Dedicated interpolation vs extrapolation breakdown.

Reads the main sweep results and produces a focused comparison table and plot.
"""

import sys, os, csv, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(ROOT, "results", "figures")
TAB_DIR = os.path.join(ROOT, "results", "tables")

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def main():
    raw_path = os.path.join(TAB_DIR, "osc_sweep_raw.csv")
    if not os.path.exists(raw_path):
        print(f"ERROR: {raw_path} not found. Run damped_oscillator.py first.")
        return

    rows = load_csv(raw_path)
    noise_target = 0.05

    # Group by (method, bits) at noise=5%
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        if abs(float(r["noise_std"]) - noise_target) < 1e-6:
            groups[(r["method"], int(r["bits"]))].append(r)

    # Build comparison table
    report = []
    methods = ["A_StdNN", "B_PINN", "D_AdaPINN"]
    bits_list = [2, 4, 6, 8]

    print("\n" + "="*90)
    print("INTERPOLATION vs EXTRAPOLATION BREAKDOWN  (noise=5%, 5 seeds)")
    print(f"{'Method':14s} {'Bits':>5} {'Interp_mean':>12} {'Extrap_mean':>12} {'Ratio E/I':>10} {'Extrap_std':>11}")
    print("-"*90)

    for method in methods:
        for bits in bits_list:
            grp = groups.get((method, bits), [])
            if not grp:
                continue
            interp_vals = [float(r["mse_interp"]) for r in grp]
            extrap_vals = [float(r["mse_extrap"]) for r in grp]
            i_mean = np.mean(interp_vals)
            e_mean = np.mean(extrap_vals)
            e_std  = np.std(extrap_vals)
            ratio  = e_mean / max(i_mean, 1e-10)
            entry = dict(
                method=method, bits=bits,
                interp_mean=i_mean, interp_std=np.std(interp_vals),
                extrap_mean=e_mean, extrap_std=e_std,
                ratio=ratio,
            )
            report.append(entry)
            print(f"{method:14s} {bits:5d} {i_mean:12.6f} {e_mean:12.6f} {ratio:10.1f}x {e_std:11.6f}")

    print("="*90)

    # PINN improvement table
    print("\nPINN IMPROVEMENT over StdNN  (noise=5%)")
    print(f"{'Method':14s} {'Bits':>5} {'Interp_imp%':>12} {'Extrap_imp%':>12}")
    print("-"*50)
    for method in ["B_PINN", "D_AdaPINN"]:
        for bits in bits_list:
            std_row  = next((r for r in report if r["method"]=="A_StdNN" and r["bits"]==bits), None)
            pinn_row = next((r for r in report if r["method"]==method and r["bits"]==bits), None)
            if std_row and pinn_row:
                imp_i = (1 - pinn_row["interp_mean"] / max(std_row["interp_mean"], 1e-10)) * 100
                imp_e = (1 - pinn_row["extrap_mean"] / max(std_row["extrap_mean"], 1e-10)) * 100
                print(f"{method:14s} {bits:5d} {imp_i:12.1f} {imp_e:12.1f}")
    print()

    # Save
    csv_path = os.path.join(TAB_DIR, "interp_extrap_breakdown.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader(); w.writerows(report)
    print(f"Saved {csv_path}")

    json_path = os.path.join(TAB_DIR, "interp_extrap_breakdown.json")
    with open(json_path, "w") as jf:
        json.dump(report, jf, indent=2)

    # Grouped bar chart: interp vs extrap for each method at 8-bit
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-method interp vs extrap at 8-bit
    sub8 = [r for r in report if r["bits"] == 8]
    x = np.arange(len(sub8))
    w = 0.35
    names = [r["method"].replace("_", " ") for r in sub8]
    interps = [r["interp_mean"] for r in sub8]
    extraps = [r["extrap_mean"] for r in sub8]
    axes[0].bar(x - w/2, interps, w, label="Interpolation", color="steelblue", edgecolor="k")
    axes[0].bar(x + w/2, extraps, w, label="Extrapolation", color="tomato", edgecolor="k")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=9)
    axes[0].set_ylabel("MSE"); axes[0].set_title("Interp vs Extrap at 8-bit, 5% noise")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    # Right: extrap/interp ratio across bit-widths
    for method, color, marker in [("A_StdNN", "tomato", "s"), ("B_PINN", "seagreen", "o"), ("D_AdaPINN", "royalblue", "^")]:
        sub = [r for r in report if r["method"] == method]
        bs = [r["bits"] for r in sub]
        rs = [r["ratio"] for r in sub]
        axes[1].plot(bs, rs, f"{marker}-", color=color, lw=2, markersize=8,
                     label=method.replace("_", " "))
    axes[1].set_xlabel("Bit-width"); axes[1].set_ylabel("Extrap/Interp MSE ratio")
    axes[1].set_title("Extrapolation Degradation Ratio")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "interp_extrap_breakdown.png")
    plt.savefig(fig_path, dpi=200); plt.close()
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
