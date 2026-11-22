"""Presentation figures (style: one marker per material, dashed lines between computed points,
reference TiN in black, paper-standard formats). Output: 04_summary/figures/*.png (400 dpi).
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "04_summary", "figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
NAVY, ORANGE, STEEL = "#1F3864", "#E0622B", "#5B8BD0"
FINAL = ["TiN", "AlN", "HfN", "ZrN", "TaN_cubic", "NbN_cubic", "TiC"]
LABEL = {"TaN_cubic": "TaN", "NbN_cubic": "NbN", "ZrO2_mono": "m-ZrO$_2$", "ZrO2_tet": "t-ZrO$_2$"}
MK = {"TiN": "o", "AlN": "s", "HfN": "D", "ZrN": "P", "TaN_cubic": "^", "NbN_cubic": "v", "TiC": "X"}
COL = {"TiN": "black", "AlN": "black"}
lab = lambda n: LABEL.get(n, n.replace("2", "$_2$").replace("3", "$_3$").replace("4", "$_4$").replace("5", "$_5$"))

# 1) funnel ---------------------------------------------------------------
steps = [("N0  all compounds\n(1-2 metals + N/C/B/O)", 449), ("N1  E$_{hull}$<0.05 or ICSD", 297),
         ("N2  no reaction with\nTiN / ZrO$_2$", 223), ("N3  EOT penalty\n≤ TiN (2.34 nm)", 215),
         ("N4a ALD process\nreported", 130), ("N4b binary,\ndeposited phase", 17), ("Final candidates", 5)]
fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=400)
y = np.arange(len(steps))[::-1]
ax.barh(y, [s[1] for s in steps], color=[STEEL] * 6 + [NAVY], height=0.6)
for yi, (t, n) in zip(y, steps):
    ax.text(n + 6, yi, str(n), va="center", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([s[0] for s in steps], fontsize=7)
ax.set_xlabel("number of compositions"); ax.set_xlim(0, 500)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_funnel.png")); plt.close(fig)

# 2) validation: M3GNet vs MP-PBE volume ---------------------------------
vp = json.load(open(os.path.join(ROOT, "04_summary", "volume_parity.json")))
fig, ax = plt.subplots(figsize=(3.6, 3.4), dpi=400)
x = np.array([v["V_MP_per_atom"] for v in vp.values()]); yv = np.array([v["V_M3GNet_per_atom"] for v in vp.values()])
lo, hi = 8.3, 16.2
ax.plot([lo, hi], [lo, hi], color="black", lw=0.8)
ax.fill_between([lo, hi], [lo * 0.97, hi * 0.97], [lo * 1.03, hi * 1.03], color="0.9", zorder=0, label="±3%")
for (n, v) in vp.items():
    fin = n in FINAL
    ax.scatter(v["V_MP_per_atom"], v["V_M3GNet_per_atom"], marker=MK.get(n, "o"), s=26 if fin else 14,
               color=COL.get(n, NAVY if fin else "0.55"), zorder=3)
    if fin or n == "SiO2":
        ax.annotate(lab(n), (v["V_MP_per_atom"], v["V_M3GNet_per_atom"]), xytext=(3, -8), textcoords="offset points", fontsize=6)
mae = np.mean(np.abs(yv / x - 1)) * 100
ax.set_xlabel("MP (PBE DFT) volume (Å$^3$/atom)"); ax.set_ylabel("M3GNet volume (Å$^3$/atom)")
ax.set_title(f"{len(vp)} structures: mean |error| vs DFT {mae:.1f}%", fontsize=8)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.legend(fontsize=6, loc="upper left", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_validation_volume.png")); plt.close(fig)

# 3) NEB profiles (mechanism that sets the barrier) -----------------------
summ = pd.read_csv(os.path.join(ROOT, "04_summary", "summary_table.csv")).set_index("material")
neb = json.load(open(os.path.join(ROOT, "03_neb", "results.json")))
ov = json.load(open(os.path.join(ROOT, "03_neb", "results_overrides.json")))
fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=400)
for n in FINAL:
    f = LABEL.get(n, n)
    mech = "Oi" if summ.loc[f, "mechanism"] == "O_i" else "ON_VN"
    r = dict(neb[n][mech]); r.update(ov.get(n, {}).get(mech, {}))
    E = np.array(r["E_profile_rel_eV"]); s = np.linspace(0, 1, len(E))
    ax.plot(s, E, ls="--", lw=0.9, marker=MK[n], ms=4, color=COL.get(n, NAVY),
            mfc=COL.get(n, NAVY) if n in COL else "white",
            label=f"{lab(n)}  {summ.loc[f, 'Ea']:.2f} eV ({'O$_i$' if mech == 'Oi' else 'O$_N$+V$_N$'})")
ax.set_xlabel("reaction coordinate (normalised)"); ax.set_ylabel("E − E$_{initial}$ (eV)")
ax.legend(fontsize=6.5, frameon=False, loc="upper left")
ax.set_title("O migration barrier, M3GNet CI-NEB (216–300 atoms)", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_neb_profiles.png")); plt.close(fig)

# 4) CTE (QHA-valid only) -------------------------------------------------
cte = json.load(open(os.path.join(ROOT, "05_cte", "results.json")))
cs = pd.read_csv(os.path.join(ROOT, "05_cte", "cte_summary.csv")).set_index("material")
fig, ax = plt.subplots(figsize=(4.4, 3.3), dpi=400)
for n in ["TiN", "AlN", "HfN", "ZrN", "TiC"]:
    T = np.array(cte[n]["alpha_T"]["T"]); a = np.array(cte[n]["alpha_T"]["alpha_1e-6"])
    ax.plot(T, a, ls="--", lw=0.9, marker=MK[n], ms=4, color=COL.get(n, NAVY),
            mfc=COL.get(n, NAVY) if n in COL else "white", label=lab(n))
ax.set_xlabel("temperature (K)"); ax.set_ylabel(r"linear CTE $\alpha$ (10$^{-6}$ K$^{-1}$)")
ax.legend(fontsize=7, frameon=False)
ax.set_title("QHA (phonopy), relative comparison only;\nTaN, NbN, ZrO$_2$: QHA not valid (soft / imaginary modes)", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_cte.png")); plt.close(fig)
# 5) selection map, slide version (larger fonts, legend explained on the slide) -----------
d = pd.read_csv(os.path.join(ROOT, "04_summary", "summary_table.csv")).dropna(subset=["Ea", "EOT_penalty_nm"])
fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=400)
mks = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "p", "*", "d", "H", "8", "o", "s", "^"]
OFF = {"TaN": (6, 4), "TaC": (-26, 6), "Ta2O5": (6, 4), "TiC": (6, -11), "TiN": (6, 4), "TiO2": (6, -4),
       "ZrN": (-14, 9), "HfN": (7, -3), "NbN": (6, -11), "NbC": (8, 2), "Y2O3": (-14, -16), "Nb2O5": (6, -12),
       "HfO2": (5, 2), "Al2O3": (5, 2), "AlN": (5, 3), "Si3N4": (5, 2), "SiC": (5, 2), "SiO2": (5, 2)}
for i, r in d.reset_index(drop=True).iterrows():
    ref = r.cls.startswith("reference")
    col = "black" if ref else (NAVY if r.conductive else ORANGE)
    face = col if not isinstance(r.flag, str) or not r.flag else "white"
    ax.scatter([r.Ea], [r.EOT_penalty_nm], marker=mks[i % len(mks)], s=55, color=face, edgecolors=col, linewidths=1.2, zorder=3)
    if r.EOT_is_bound:
        ax.annotate("", xy=(r.Ea, r.EOT_penalty_nm - 0.25), xytext=(r.Ea, r.EOT_penalty_nm), arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
    ax.annotate(lab(r.material) + (" (k?)" if r.EOT_is_bound else ""), (r.Ea, r.EOT_penalty_nm),
                xytext=OFF.get(r.material, (5, 3)), textcoords="offset points", fontsize=8,
                fontweight="bold" if r.material in ("HfN", "ZrN") else "normal")
ax.axvline(d[d.material == "TiN"].Ea.iloc[0], color="black", ls="--", lw=0.8)
ax.set_xlabel("O migration barrier $E_m$ (eV)"); ax.set_ylabel("EOT penalty (nm, 1 nm interlayer oxidised)")
ax.set_xlim(-0.05, 2.4); ax.set_ylim(-0.08, 2.55)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_selection_map_slide.png")); plt.close(fig)
print("figures written to", OUT)

# 6) YSZ validation: Arrhenius of O diffusion, v1 (friction 0.01/fs) and v2 (0.001/fs) ----------------
from scipy.stats import linregress
fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=400)
for fn, mk, nm in [("arrhenius.json", "o", "Langevin 0.01 fs$^{-1}$"), ("arrhenius_v2.json", "s", "Langevin 0.001 fs$^{-1}$")]:
    dd = json.load(open(os.path.join(ROOT, "06_ysz_md", fn)))
    xs, ys = [], []
    for T, v in dd["D_all"].items():
        for D in v: xs.append(1000 / float(T)); ys.append(D)
    rr = linregress(np.array(xs) / (1000 * 8.617333e-5), np.log(ys))
    xf = np.linspace(min(xs), max(xs), 20)
    col = NAVY if mk == "o" else ORANGE
    ax.scatter(xs, ys, marker=mk, s=28, color=col, zorder=3)
    ax.plot(xf, np.exp(rr.intercept + rr.slope * xf / (1000 * 8.617333e-5)), ls="--", lw=0.9, color=col,
            label=f"{nm}: E$_a$ = {-rr.slope:.2f} ± {rr.stderr:.2f} eV")
ax.set_yscale("log"); ax.set_xlabel("1000 / T (K$^{-1}$)"); ax.set_ylabel("D$_O$ (cm$^2$/s)")
ax.set_title("8YSZ (316 atoms), M3GNet MD, 2 seeds per T\nexperiment E$_a$ ≈ 0.85 (high T) – 1.1 eV (low T)", fontsize=7.5)
ax.legend(fontsize=6.5, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_ysz_arrhenius.png")); plt.close(fig)
print("fig6 written")
