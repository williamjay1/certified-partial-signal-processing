"""Publication-size quantitative figures from unchanged, executed study outputs.

Style-only inheritance: SciencePlots (MIT) and Baker figure QA (Apache-2.0).
No fitting, model inference, filtering of executions, or new experiments.
"""
from pathlib import Path
import os, sys, json, csv, hashlib

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache" / "matplotlib_nature")
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "temp")
for folder in (
    r"D:\MLWork\ASC_SelectiveReplanning_20260930\env_core",
    r"D:\MLWork\ASC_SelectiveReplanning_20260930\env_packages",
):
    sys.path.insert(0, folder)
VENDOR = ROOT / "scripts" / "vendor" / "nature_design"
sys.path.insert(0, str(VENDOR / "baker" / "scripts"))
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.style.use([str(VENDOR / n) for n in
               ("science.mplstyle", "no-latex.mplstyle", "nature.mplstyle")])
mpl.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "mathtext.fontset": "dejavusans",
    "text.usetex": False,
    "axes.linewidth": 0.55,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.top": False,
    "ytick.right": False,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "axes.grid": False,
    "axes.labelpad": 4,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "path.simplify": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": None,
})
BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#D55E00"
INK = "#24282C"
GRAY = "#7B8187"
LIGHT = "#C9CDD1"
OUT = ROOT / "artwork" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {}
RECORDS = []
RASTER_DPI = 1000

def source(path):
    p = ROOT / "data" / path
    SOURCES[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def read(path):
    return json.loads(source(path).read_text(encoding="utf-8"))

def rows(path):
    with source(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def export_csv(name, fields, records):
    with (OUT / (name + "_source.csv")).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

def canvas(height_mm):
    return plt.figure(figsize=(7.204724409, height_mm / 25.4), dpi=300)

def heading(fig, ax, letter, title, ygap=.035):
    rect = ax.get_position()
    fig.text(rect.x0 - .037, rect.y1 + ygap, letter,
             fontsize=8, fontweight="bold", color=INK, va="bottom", ha="left")
    fig.text(rect.x0, rect.y1 + ygap, title, fontsize=7,
             color=INK, va="bottom", ha="left")

def text_geometry(fig):
    """Blocking text/text and page-clipping check at the physical canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        b = text.get_window_extent(renderer)
        if b.width < .1 or b.height < .1:
            continue
        boxes.append((text, b))
    findings = []
    page = fig.bbox
    px_tol = fig.dpi / 72 * .35
    for t, b in boxes:
        if b.x0 < -px_tol or b.y0 < -px_tol or b.x1 > page.x1 + px_tol or b.y1 > page.y1 + px_tol:
            findings.append({"type": "CLIPPED", "text": t.get_text()})
    for i, (t, b) in enumerate(boxes):
        for u, c in boxes[i + 1:]:
            overlap_x = min(b.x1, c.x1) - max(b.x0, c.x0)
            overlap_y = min(b.y1, c.y1) - max(b.y0, c.y0)
            if overlap_x > px_tol and overlap_y > px_tol:
                findings.append({"type": "TEXT_OVERLAP", "texts": [t.get_text(), u.get_text()]})
    return {"status": "PASS" if not findings else "FAIL",
            "text_objects": len(boxes), "findings": findings}

def save(fig, name, question, scope, data_values):
    geometry = text_geometry(fig)
    (OUT / (name + "_text_geometry.json")).write_text(
        json.dumps(geometry, indent=2), encoding="utf-8")
    if geometry["findings"]:
        raise RuntimeError(name + ": " + str(geometry["findings"]))
    alignment = require_matplotlib_panel_alignment(
        fig, json_out=OUT / (name + "_alignment.json"),
        overlay_svg=OUT / (name + "_alignment_overlay.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=1000)
    fig.savefig(OUT / f"{name}_preview.png", dpi=300)
    RECORDS.append({
        "name": name, "question": question, "scope": scope,
        "width_mm": 183, "height_mm": float(fig.get_figheight() * 25.4),
        "raster_dpi": RASTER_DPI, "editable_vector": True,
        "data": data_values, "text_geometry": geometry["status"],
        "panel_alignment": alignment.get("status"),
    })
    plt.close(fig)
    print("EXPORTED", name, flush=True)

DATA = read("results/final_analysis/har_locked_v1/heldout_analysis.json")
LATENCY = rows("results/review2_latency_analysis/saved_output_v1/paired_call_data.csv")
PROFILE = read("results/review2_latency_analysis/component_profile_v1/component_profile.json")
HYDRAULIC = read("results/hydraulic_external_confirmation/20261002T061642Z_3208a13f/executed_results.json")

def pooled(name, delta=None):
    return next(x for x in DATA["pooled"] if x["name"] == name and x["delta"] == delta)

# Figure 2: subject heterogeneity with distinct timed and all-test denominators.
sub = DATA["subject_primary"]
positions = np.arange(len(sub))
fig = canvas(94)
grid = fig.add_gridspec(2, 1, left=.12, right=.965, bottom=.13,
                       top=.89, hspace=.8)
axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
axes[0].bar(positions, [100 * x["mean_time_reduction"] for x in sub],
            width=.58, color=BLUE, edgecolor="none")
axes[0].set_ylim(0, 15)
axes[0].set_yticks([0, 5, 10, 15])
axes[0].set_ylabel("Mean time reduction (%)")
axes[0].set_xticks(positions, [str(x["subject"]) for x in sub])
axes[0].tick_params(axis="x", bottom=False, labelbottom=False)
heading(fig, axes[0], "a", "Processing cost · 100 timed windows per subject")
axes[1].vlines(positions, 0, [1000 * x["full_risk_excess"] for x in sub],
               color=LIGHT, linewidth=.65)
axes[1].plot(positions, [1000 * x["full_risk_excess"] for x in sub],
             color=BLUE, linestyle="none", marker="o", markersize=3.5,
             markeredgecolor="white", markeredgewidth=.45)
axes[1].set_ylim(0, 8)
axes[1].set_yticks([0, 2, 4, 6, 8])
axes[1].set_ylabel("Mean loss increase (×10⁻³)")
axes[1].set_xlabel("Unseen subject")
axes[1].set_xticks(positions, [str(x["subject"]) for x in sub])
heading(fig, axes[1], "b", "Probability fidelity · all released windows per subject")
for ax in axes:
    ax.set_xlim(-.6, len(sub) - .4)
export_csv("figure2_subject_variation", list(sub[0]), sub)
save(fig, "figure2_subject_variation",
     "How do cost savings and probability deterioration vary by unseen subject?",
     "Nine people; three frozen systems averaged. Top: 100 timed windows/person. "
     "Bottom: all 2,947 released windows. Descriptive means, no false independent-system interval.",
     sub)

# Figure 3: complete time and the individual guarantee are separate outcomes.
methods = [
    ("tight_minimax", .025, "Simplex minimax"),
    ("coordinate_minimax", .025, "Coordinate minimax"),
    ("tight_midpoint", .025, "Certified midpoint"),
    ("uniform_A_tight_minimax", .025, "Uniform checkpoints"),
    ("score_separation_argmax", None, "Class preservation"),
    ("fixed_prefix_mean_completion", None, "Fixed prefix"),
    ("entropy_mean_completion", None, "Entropy"),
]
controls = [(pooled(n, d), label) for n, d, label in methods]
yy = np.array([0, 1, 2, 3, 4.7, 5.7, 6.7])
fig = canvas(98)
grid = fig.add_gridspec(1, 2, left=.255, right=.932, bottom=.20,
                       top=.84, wspace=.55)
axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
axes[0].barh(yy, [x["complete_mean_ms"] for x, _ in controls],
             height=.58, color=[BLUE] * 4 + [GRAY] * 3, edgecolor="none")
axes[1].barh(yy, [100 * x["delta025_every_label_violations"] / 8841
                  for x, _ in controls], height=.58,
             color=[BLUE] * 4 + [ORANGE] * 3, edgecolor="none")
axes[1].scatter([0] * 4, yy[:4], s=9, facecolor="white", edgecolor=BLUE,
                linewidth=.75, zorder=4, clip_on=False)
reference_ms = float(np.mean([
    next(m["complete_mean_ms"] for m in s["methods"] if m["name"] == "compiled_full")
    for s in DATA["systems"]]))
axes[0].axvline(reference_ms, color=INK, linestyle=(0, (3, 2)), linewidth=.65)
axes[0].set_xlim(0, 1.25)
axes[0].set_xticks([0, .4, .8, 1.2], ["0", "0.4", "0.8", "1.2"])
axes[1].set_xlim(0, 55)
axes[1].set_xticks([0, 20, 40], ["0", "20", "40"])
axes[0].set_yticks(yy, [label for _, label in controls])
axes[1].set_yticks(yy, [""] * len(yy))
for ax in axes:
    ax.set_ylim(7.4, -.6)
    ax.tick_params(axis="y", length=0, pad=5)
axes[0].set_xlabel("Complete mean time (ms)")
axes[1].set_xlabel("Tolerance exceedances (%)")
heading(fig, axes[0], "a", "Execution cost")
heading(fig, axes[1], "b", "Individual fidelity")
for y, (record, _) in zip(yy, controls):
    axes[0].text(1.035, y, f'{record["complete_mean_ms"]:.3f}',
                 transform=axes[0].get_yaxis_transform(), fontsize=6.5,
                 va="center", ha="left")
    rate = 100 * record["delta025_every_label_violations"] / 8841
    axes[1].text(1.035, y, f"{rate:.2f}",
                 transform=axes[1].get_yaxis_transform(), fontsize=6.5,
                 va="center", ha="left")
fig.legend([Line2D([], [], color=INK, linestyle=(0, (3, 2)), linewidth=.65)],
           ["Compiled full reference"], loc="lower center",
           bbox_to_anchor=(.59, .025), handlelength=2.5)
control_values = [{
    "method": label, "complete_mean_ms": x["complete_mean_ms"],
    "all_test_system_windows": 8841,
    "every_label_tolerance_exceedances": x["delta025_every_label_violations"],
    "exceedance_pct": 100 * x["delta025_every_label_violations"] / 8841,
} for x, label in controls]
export_csv("figure3_stopping_controls", list(control_values[0]), control_values)
save(fig, "figure3_stopping_controls",
     "Does a lower mean cost also preserve the individual probability tolerance?",
     "Left: same 900 windows; right: 2,947 windows × three frozen systems = 8,841 outputs. "
     "Delta=0.025; all seven controls retained; fixed-prefix and entropy have identical outputs.",
     {"reference_ms": reference_ms, "controls": control_values})

# Figure 4: all tolerance settings, with actual marginal HAR bootstrap intervals.
tolerances = [0, .01, .025, .05, .1]
har = [pooled("tight_minimax", d) for d in tolerances]
hyd = [next(x for x in HYDRAULIC["all_methods"]
            if x["method"] == "tight_minimax" and x["delta"] == d)
       for d in tolerances]
hyd_full = next(x for x in HYDRAULIC["all_methods"]
                if x["method"] == "compiled_full")["complete_ms"]["mean"]
markers = ["s", "^", "o", "D", "v"]
fig = canvas(89)
grid = fig.add_gridspec(1, 2, left=.105, right=.97, bottom=.20,
                       top=.79, wspace=.35)
axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
values = []
for panel, (ax, records) in enumerate(zip(axes, (har, hyd))):
    xx = [1000 * (r["timed_cohort_risk_excess"] if panel == 0
                  else r["mean_reference_loss_excess"]) for r in records]
    y = [(1 - r["time_reduction"]) if panel == 0
         else r["complete_ms"]["mean"] / hyd_full for r in records]
    assert np.all(np.isfinite(xx)) and np.all(np.isfinite(y))
    ax.plot(xx, y, color=BLUE, linewidth=.9, zorder=2)
    for i, (d, x, v, record) in enumerate(zip(tolerances, xx, y, records)):
        item = {"workload": "HAR" if panel == 0 else "Hydraulic",
                "delta": d, "mean_half_brier_increase": x / 1000,
                "complete_mean_time_ratio": v}
        if panel == 0:
            xl, xu = record["timed_cohort_risk_excess_95ci"]
            sl, su = record["time_reduction_95ci"]
            yl, yu = 1 - su, 1 - sl
            assert xl <= x / 1000 <= xu and yl <= v <= yu
            ax.errorbar(x, v, xerr=[[x - 1000 * xl], [1000 * xu - x]],
                        yerr=[[v - yl], [yu - v]], fmt="none",
                        ecolor=LIGHT, elinewidth=.65, capsize=1.6,
                        capthick=.65, zorder=1)
            item.update({"loss_increase_ci_low": xl, "loss_increase_ci_high": xu,
                         "time_ratio_ci_low": yl, "time_ratio_ci_high": yu})
        ax.plot(x, v, marker=markers[i], markersize=4 if d == .025 else 3.4,
                markerfacecolor=BLUE if d == .025 else "white",
                markeredgecolor=BLUE, markeredgewidth=.85,
                linestyle="none", zorder=4)
        values.append(item)
    ax.axhline(1, color=LIGHT, linestyle=(0, (2, 2)), linewidth=.65, zorder=0)
    ax.plot(0, 1, marker="x", color=INK, markersize=4, markeredgewidth=.85)
    ax.set_xlim(-.65, 22)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_ylim(.55, 1.25)
    ax.set_yticks([.6, .8, 1, 1.2])
    ax.set_xlabel("Observed mean loss increase (×10⁻³)")
    ax.set_ylabel("Complete mean time / full")
heading(fig, axes[0], "a", "HAR · same 900 windows")
heading(fig, axes[1], "b", "Hydraulic · 435 cycles")
handles = [Line2D([], [], marker=m, linestyle="none", markersize=3.5,
                  markerfacecolor=BLUE if d == .025 else "white",
                  markeredgecolor=BLUE, markeredgewidth=.8)
           for m, d in zip(markers, tolerances)]
handles.append(Line2D([], [], marker="x", color=INK, linestyle="none", markersize=4))
fig.legend(handles, [f"δ = {d:g}" for d in tolerances] + ["Compiled full"],
           ncol=6, loc="upper center", bbox_to_anchor=(.53, .975),
           handlelength=1.1, handletextpad=.45, columnspacing=1.25)
export_csv("figure4_fidelity_time", sorted(set().union(*(x.keys() for x in values))), values)
save(fig, "figure4_fidelity_time",
     "What measured complete-time/fidelity trade-off follows from the requested tolerance?",
     "All five tolerances, original contemporaneous full references. HAR intervals: "
     "marginal 95% shared-subject bootstrap, not a joint region. Hydraulic descriptive; no CI fabricated.",
     values)

# Figure 5: all calls and a complementary paired quantile view; no tail censoring.
assert len(LATENCY) == 8100
fig = canvas(132)
grid = fig.add_gridspec(2, 3, left=.105, right=.97, bottom=.155,
                       top=.89, hspace=.88, wspace=.40)
axes = [[fig.add_subplot(grid[r, c]) for c in range(3)] for r in range(2)]
quantile_records = []
for c, seed in enumerate((17, 29, 43)):
    selected = [r for r in LATENCY if int(r["seed"]) == seed]
    assert len(selected) == 2700
    arrays = {}
    for key, label, color, linestyle in (
        ("full_ms", "Compiled full", GRAY, (0, (3, 2))),
        ("partial_ms", "Certified partial", BLUE, "-"),
    ):
        a = np.sort(np.array([float(r[key]) for r in selected]))
        assert np.all(a > 0)
        arrays[key] = a
        surv = 100 * (len(a) - np.arange(len(a))) / len(a)
        axes[0][c].step(a, surv, where="pre", color=color,
                        linestyle=linestyle, linewidth=.85)
        axes[0][c].plot(a[-1], surv[-1], marker="o", color=color,
                        markersize=2, linestyle="none")
    assert len(arrays["full_ms"]) == len(arrays["partial_ms"]) == 2700
    upper = axes[0][c]
    upper.set_xscale("log")
    upper.set_yscale("log")
    upper.set_xlim(.5, 100)
    upper.set_ylim(.03, 110)
    upper.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    upper.set_yticks([.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    upper.minorticks_off()
    upper.set_xlabel("Execution time (ms)")
    if c == 0:
        upper.set_ylabel("Upper tail calls (%)")
    else:
        upper.tick_params(labelleft=False)
    heading(fig, upper, "abc"[c], f"All calls · system {seed}", ygap=.025)
    lower = axes[1][c]
    for j, q in enumerate((.5, .95, .99)):
        full_q = float(np.quantile(arrays["full_ms"], q))
        partial_q = float(np.quantile(arrays["partial_ms"], q))
        lower.plot([full_q, partial_q], [j - .1, j + .1],
                   color=LIGHT, linewidth=.7, zorder=1)
        lower.plot(full_q, j - .1, marker="o", markerfacecolor="white",
                   markeredgecolor=GRAY, markeredgewidth=.8,
                   markersize=3.4, linestyle="none", zorder=3)
        lower.plot(partial_q, j + .1, marker="o", markerfacecolor=BLUE,
                   markeredgecolor=BLUE, markersize=3.4,
                   linestyle="none", zorder=3)
        quantile_records.append({"system": seed, "calls_per_method": 2700,
                                 "quantile": q, "full_ms": full_q,
                                 "partial_ms": partial_q})
    lower.set_xlim(.75, 2.8)
    lower.set_xticks([1, 1.5, 2, 2.5], ["1", "1.5", "2", "2.5"])
    lower.set_ylim(2.5, -.5)
    lower.set_yticks([0, 1, 2], ["P50", "P95", "P99"])
    lower.tick_params(axis="y", length=0)
    lower.set_xlabel("Quantile time (ms)")
    heading(fig, lower, "def"[c], f"Quantiles · system {seed}", ygap=.025)
fig.legend([
    Line2D([], [], color=GRAY, linestyle=(0, (3, 2)), marker="o",
           markerfacecolor="white", markeredgecolor=GRAY, markersize=3.4),
    Line2D([], [], color=BLUE, linewidth=.85, marker="o", markersize=3.4),
], ["Compiled full", "Certified partial"], ncol=2, loc="lower center",
           bbox_to_anchor=(.53, .025), columnspacing=2)
export_csv("figure5_latency_tails", list(quantile_records[0]), quantile_records)
export_csv("figure5_latency_tails_all_calls", list(LATENCY[0]), LATENCY)
save(fig, "figure5_latency_tails",
     "Do lower mean costs coexist with adverse execution-time tails?",
     "Original uninstrumented calls; 2,700 calls/method/system, all retained, "
     "including the 72.1447 ms full call. Upper row exact empirical survival, lower row raw-call quantiles.",
     {"quantiles": quantile_records,
      "maximum_full_ms": max(float(r["full_ms"]) for r in LATENCY),
      "maximum_partial_ms": max(float(r["partial_ms"]) for r in LATENCY)})

# Figure 6: complete diagnostic decomposition and magnified decision components.
keys = ["feature_ms", "leaf_enclosure_ms", "probability_interval_ms",
        "output_certificate_ms", "exact_fallback_ms", "control_ms",
        "state_setup_ms", "unassigned_instrumentation_residual_ms"]
labels = ["Feature operators", "Leaf/score enclosure", "Probability intervals",
          "Output/certificate", "Exact prediction", "Control", "Setup", "Residual"]
colors = [BLUE, "#56B4E9", TEAL, ORANGE, "#8E7B6B", "#E6AF73", GRAY, LIGHT]
prows = [next(x for x in PROFILE["aggregate"] if x["seed"] == seed and x["method"] == method)
         for seed in (17, 29, 43)
         for method in ("instrumented_compiled_full", "instrumented_tight_primary")]
fig = canvas(123)
grid = fig.add_gridspec(2, 1, left=.20, right=.97, bottom=.165,
                       top=.855, height_ratios=[2.1, 1], hspace=1.1)
axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
positions = np.array([0, 1, 2.6, 3.6, 5.2, 6.2])
left = np.zeros(6)
for key, label, color in zip(keys, labels, colors):
    values = np.array([r["component_mean_ms"][key] for r in prows])
    assert np.all(values >= 0), "Negative signed residual requires a diverging plot."
    axes[0].barh(positions, values, left=left, height=.58, label=label,
                 color=color, linewidth=0)
    left += values
assert np.allclose(left, [r["complete_ms"]["mean"] for r in prows], atol=2e-12)
axes[0].set_xlim(0, 1)
axes[0].set_xticks([0, .2, .4, .6, .8, 1], ["0", "0.2", "0.4", "0.6", "0.8", "1"])
axes[0].set_ylim(6.9, -.7)
axes[0].set_yticks(positions, [f"{s}  {m}" for s in (17, 29, 43) for m in ("Full", "Partial")])
axes[0].tick_params(axis="y", length=0, pad=5)
axes[0].set_xlabel("Instrumented complete mean time (ms)")
heading(fig, axes[0], "a", "All measured components", ygap=.024)
small = [("Probability intervals", ["probability_interval_ms"]),
         ("Output + certificate", ["output_certificate_ms"]),
         ("Control", ["control_ms"]),
         ("Setup + residual", ["state_setup_ms", "unassigned_instrumentation_residual_ms"])]
for method, color, shift, label in (
    ("instrumented_compiled_full", GRAY, -.15, "Full"),
    ("instrumented_tight_primary", BLUE, .15, "Partial"),
):
    records = [r for r in prows if r["method"] == method]
    vals = [1000 * float(np.mean([sum(r["component_mean_ms"][key] for key in kk)
                                  for r in records])) for _, kk in small]
    axes[1].barh(np.arange(4) + shift, vals, height=.23, color=color, label=label)
axes[1].set_yticks(np.arange(4), [label for label, _ in small])
axes[1].tick_params(axis="y", length=0, pad=5)
axes[1].set_ylim(3.6, -.6)
axes[1].set_xlim(0, 16)
axes[1].set_xticks([0, 4, 8, 12, 16])
axes[1].set_xlabel("Mean component time (µs)")
heading(fig, axes[1], "b", "Small components · pooled across systems", ygap=.024)
axes[1].legend(loc="upper right", bbox_to_anchor=(1, -.43), ncol=2,
               borderaxespad=0, handlelength=1.5, columnspacing=1.5)
fig.legend(*axes[0].get_legend_handles_labels(), ncol=4,
           loc="upper center", bbox_to_anchor=(.53, .998),
           columnspacing=1.2, handlelength=1.1, handletextpad=.5, fontsize=6.5)
component_values = [{"system": r["seed"], "method": r["method"],
                     "complete_ms": r["complete_ms"]["mean"],
                     **r["component_mean_ms"]} for r in prows]
export_csv("figure6_component_cost", list(component_values[0]), component_values)
save(fig, "figure6_component_cost",
     "How much diagnostic time belongs to signal work and to the certification decision?",
     "Separate instrumented 36-window diagnostic shared by three systems and three repeats: "
     "324 calls/method. All eight nonoverlapping components and signed residual retained. "
     "Small-component panel magnifies the same records; no substitution for main timing.",
     component_values)

manifest = {
    "stage": "NATURE_STYLE_FROM_UNCHANGED_EXECUTED_OUTPUTS",
    "style_inheritance": {
        "SciencePlots_commit": "b9b16959570bd2fbc9ff5118bacc423c3bddd592",
        "Baker_skill_commit": "ee5aaa47060ed116d7ae2d2429761ae6ab07f9e0",
        "scope": "Style-only inheritance and geometry QA; no claim of Nature certification.",
    },
    "raster_dpi": RASTER_DPI,
    "sources": [{"path": p, "sha256": h} for p, h in SOURCES.items()],
    "figures": RECORDS,
    "model_fits": 0, "model_predictions": 0, "timing_calls": 0,
}
(OUT / "quantitative_figure_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8")
print("COMPLETE", len(RECORDS), "quantitative figures", flush=True)
