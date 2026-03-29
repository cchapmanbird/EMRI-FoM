import glob
import os
from dataclasses import dataclass

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Use the physrev style if available
# try:
plt.style.use('../physrev.mplstyle')
# except:
#     pass


@dataclass
class SpinDatum:
    name: str
    mass_million_solar: float
    spin_kind: str  # "bounded" or "lower_limit"
    spin_precision: float


# Precision definition used in this plot:
# - bounded measurement a = a0^{+u}_{-l}: precision = u + l (90% interval width)
# - lower limit a > a_min: precision proxy = 1 - a_min (distance to maximal spin)
# Smaller values indicate tighter constraints in both cases.
DATA = [
    SpinDatum("Mrk359", 1.1, "bounded", 0.30 + 0.54),
    SpinDatum("Ark564", 1.1, "lower_limit", 1.0 - 0.9),
    SpinDatum("Mrk766", 1.8, "lower_limit", 1.0 - 0.92),
    SpinDatum("NGC4051", 1.91, "lower_limit", 1.0 - 0.99),
    SpinDatum("NGC1365", 2.0, "lower_limit", 1.0 - 0.97),
    SpinDatum("1H0707-495", 2.3, "lower_limit", 1.0 - 0.94),
    SpinDatum("MCG--6-30-15", 2.9, "bounded", 0.06 + 0.07),
    SpinDatum("NGC5506", 5.0, "bounded", 2.0 * 0.04),
    SpinDatum("IRAS13224--3809", 6.3, "lower_limit", 1.0 - 0.975),
    SpinDatum("Tons180", 8.1, "lower_limit", 1.0 - 0.98),
    SpinDatum("ESO362--G18", 12.5, "lower_limit", 1.0 - 0.92),
    SpinDatum("Swift J2127.4+5654", 15.0, "bounded", 0.14 + 0.20),
    SpinDatum("Mrk335", 17.8, "lower_limit", 1.0 - 0.99),
    SpinDatum("Mrk110", 25.1, "lower_limit", 1.0 - 0.99),
    SpinDatum("NGC3783", 29.8, "lower_limit", 1.0 - 0.88),
    SpinDatum("1H0323+342", 34.0, "lower_limit", 1.0 - 0.9),
    SpinDatum("NGC 4151", 45.7, "lower_limit", 1.0 - 0.9),
    SpinDatum("Mrk79", 52.4, "lower_limit", 1.0 - 0.5),
    SpinDatum("PG1229+204", 57.0, "bounded", 0.06 + 0.02),
    SpinDatum("IRAS13197-1627", 64.0, "lower_limit", 1.0 - 0.7),
    SpinDatum("3C120", 69.0, "lower_limit", 1.0 - 0.95),
    SpinDatum("Mrk841", 79.0, "lower_limit", 1.0 - 0.52),
    SpinDatum("IRAS09149--6206", 100.0, "bounded", 0.02 + 0.07),
    SpinDatum("Ark120", 150.0, "lower_limit", 1.0 - 0.85),
    SpinDatum("RBS1124", 180.0, "lower_limit", 1.0 - 0.8),
    SpinDatum("RXS J1131-1231", 200.0, "bounded", 0.08 + 0.15),
    SpinDatum("Fairall 9", 255.0, "bounded", 0.19 + 0.15),
    SpinDatum("1H0419-577", 340.0, "lower_limit", 1.0 - 0.98),
    SpinDatum("PG0804+761", 550.0, "lower_limit", 1.0 - 0.97),
    SpinDatum("Q2237+305", 1000.0, "bounded", 0.06 + 0.03),
    SpinDatum("PG2112+059", 1000.0, "lower_limit", 1.0 - 0.83),
    SpinDatum("H1821+643", 4500.0, "lower_limit", 1.0 - 0.4),
]


def load_emri_spin_precision(
    spin_a: float = 0.99,
    tpl_val: float = 0.25,
    run_type: str = "circular",
    tolerance: float = 1e-6,
):
    """Load EMRI spin precision from inference files.

    Returns
    -------
    list[tuple[float, float, float]]
        Tuples of (m1 in million solar masses, absolute sigma_a, relative sigma_a/a).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(script_dir)
    inference_files = sorted(glob.glob(os.path.join(pipeline_dir, "inference_*/inference.h5")))

    emri_data = []
    for inf_file in inference_files:
        with h5py.File(inf_file, "r") as f:
            if run_type not in f:
                continue

            run_group = f[run_type]
            src_a = float(run_group["a"][()])
            src_tpl = float(np.round(run_group["Tpl"][()], decimals=5))
            if abs(src_a - spin_a) > tolerance or abs(src_tpl - tpl_val) > tolerance:
                continue

            param_names = np.array(run_group["param_names"][()], dtype=str).tolist()
            if "a" not in param_names:
                continue

            idx_a = param_names.index("a")
            detector_precision = run_group["detector_measurement_precision"][()]
            sigma_a = float(np.median(np.abs(detector_precision[:, idx_a])))
            if src_a == 0.0:
                continue
            rel_sigma_a = float(sigma_a / abs(src_a))
            absolute_sigma_a = sigma_a * 2 # to show 95 percent interval width for comparison with bounded measurements

            m1 = float(np.round(run_group["m1"][()], decimals=5))
            emri_data.append((m1 / 1e6, sigma_a, absolute_sigma_a))

    return emri_data


def main() -> None:
    bounded = [d for d in DATA if d.spin_kind == "bounded"]
    lower_lim = [d for d in DATA if d.spin_kind == "lower_limit"]

    emri_data = load_emri_spin_precision()

    # plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8/1.5, 4.5/1.5))
    # factor of 1e6 to convert from million solar masses to solar masses for the x-axis
    factor = 1e6
    ax.scatter(
        [d.mass_million_solar * factor for d in lower_lim],
        [d.spin_precision for d in lower_lim],
        s=55,
        marker="v",
        color="C0",
        alpha=0.85,
        label="Lower-limit constraints",
    )

    ax.scatter(
        [d.mass_million_solar * factor for d in bounded],
        [d.spin_precision for d in bounded],
        s=70,
        marker="o",
        color="C0",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9,
        label="Bounded constraints",
    )

    if emri_data:
        ax.scatter(
            [d[0] * factor for d in emri_data],
            [d[2] for d in emri_data],
            s=42,
            marker="s",
            color="C2",
            edgecolor="black",
            linewidth=0.35,
            alpha=0.85,
            label="EMRI constraints LISA 3 months",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Black hole mass $[M_\odot]$")
    ax.set_ylabel("Spin measurement precision")
    ax.grid(True, which="both", alpha=0.25)

    all_precisions = [d.spin_precision for d in DATA]
    if emri_data:
        all_precisions.extend([d[2] for d in emri_data])
    positive_precisions = [p for p in all_precisions if p > 0.0]
    if positive_precisions:
        ax.set_ylim(min(positive_precisions) * 0.7, max(positive_precisions) * 1.3)
        ax.set_xlim(4e4, 6e9)

    median_bounded = sorted(d.spin_precision for d in bounded)[len(bounded) // 2]
    median_lower = sorted(d.spin_precision for d in lower_lim)[len(lower_lim) // 2]

    # ax.text(
    #     0.02,
    #     0.98,
    #     (
    #         f"N = {len(DATA)} objects with mass estimates\\n"
    #         f"Median bounded width: {median_bounded:.3f}\\n"
    #         f"Median lower-limit proxy: {median_lower:.3f}\\n"
    #         f"EMRI points (a={0.99:.2f}, Tpl={0.25:.2f}): {len(emri_data)}"
    #     ),
    #     transform=ax.transAxes,
    #     va="top",
    #     ha="left",
    #     fontsize=9,
    #     bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    # )

    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    out_path = "./spin_precision_vs_mass.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
