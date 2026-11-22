"""Collect step 1-3 results into one table + selection map.

Barrier per material = lowest converged migration barrier over mechanisms (Oi, ON_VN).
For a path between inequivalent end points (dE_end != 0), the long-range barrier is the
larger of forward/reverse (the defect must also leave the lower site).
results_overrides.json (split-path re-runs, unconverged flags) takes precedence.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "04_summary")
neb = json.load(open(os.path.join(ROOT, "03_neb", "results.json")))
ov = json.load(open(os.path.join(ROOT, "03_neb", "results_overrides.json")))
el = json.load(open(os.path.join(ROOT, "02_relax_elastic", "results.json")))
n3 = pd.read_csv(os.path.join(ROOT, "01_mp_funnel", "N3_table_v2.csv")).set_index("formula")   # v2 (2026-11-22 audit)

# name in NEB / elastic -> (formula in MP table, class, flag)
MATS = {
    "TiN": ("TiN", "reference (conductive)", ""), "AlN": ("AlN", "reference (insulating)", ""),
    "TaN_cubic": ("TaN", "conductive nitride", "rocksalt (metastable, E_hull 0.33)"),
    "NbN_cubic": ("NbN", "conductive nitride", "rocksalt (metastable, E_hull 0.20)"),
    "HfN": ("HfN", "conductive nitride", ""), "ZrN": ("ZrN", "conductive nitride", ""),
    "TiC": ("TiC", "conductive carbide", "oxide = Ti oxide as on TiN (A5 degradation); lit. k 80 not credible"), "TaC": ("TaC", "conductive carbide", ""),
    "NbC": ("NbC", "conductive carbide", ""),
    "Si3N4": ("Si3N4", "insulating nitride/carbide", "ALD film amorphous; c-channel path"),
    "SiC": ("SiC", "insulating nitride/carbide", ""),
    "Al2O3": ("Al2O3", "oxide", "ALD film amorphous"), "HfO2": ("HfO2", "oxide", "M3GNet misses O-O dimer"),
    "SiO2": ("SiO2", "oxide", "low confidence: open I-42d polymorph"), "TiO2": ("TiO2", "oxide", "C2/m, ALD film usually anatase"),
    "Ta2O5": ("Ta2O5", "oxide", "ALD film amorphous"), "Nb2O5": ("Nb2O5", "oxide", "ALD film amorphous"),
    "Y2O3": ("Y2O3", "oxide", ""),
}


def mech_barrier(name, mech):
    r = dict(neb.get(name, {}).get(mech, {}))
    r.update(ov.get(name, {}).get(mech, {}))
    if not r or not r.get("converged", False) or r.get("barrier_forward_eV") is None:
        return None, r
    f, b = r["barrier_forward_eV"], r.get("barrier_reverse_eV", r["barrier_forward_eV"])
    return max(f, b) if abs(f - b) > 0.05 else f, r


rows = []
for name, (formula, cls, flag) in MATS.items():
    bO, rO = mech_barrier(name, "Oi")
    bV, rV = mech_barrier(name, "ON_VN")
    cand = [(b, m) for b, m in ((bO, "O_i"), (bV, "O_X+V_X")) if b is not None]
    best = min(cand) if cand else (None, None)
    e = el.get(name, el.get(formula, {}))
    t = n3.loc[formula] if formula in n3.index else None
    rows.append(dict(material=formula, cls=cls,
                     Ea_Oi=bO, Ea_OXVX=bV, Ea=best[0], mechanism=best[1],
                     EOT_penalty_nm=None if t is None else t["EOT_penalty_v2"],
                     EOT_is_bound=None if t is None else bool(t["EOT_is_bound"]),
                     conductive=None if t is None else bool(t["conductive"]),
                     oxide=None if t is None else t["oxide_products"],
                     B=e.get("B_VRH"), G=e.get("G_VRH"), E=e.get("E"), BG=e.get("Pugh_B_over_G"),
                     flag=flag))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "summary_table.csv"), index=False)
print(df[["material", "cls", "Ea_Oi", "Ea_OXVX", "Ea", "mechanism", "EOT_penalty_nm", "E", "flag"]].round(3).to_string(index=False))

# ---- selection map (figure style: one marker per material, reference in black, paper format)
markers = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "p", "*", "d", "H", "8", "o", "s", "^"]
OFFS = {"TaN": (6, -12), "TaC": (-26, -12), "Ta2O5": (6, 6), "TiC": (-10, 8), "TiN": (6, 4), "TiO2": (6, 6),
        "ZrN": (-30, 8), "HfN": (6, 4), "NbN": (6, 4), "NbC": (6, 4), "Y2O3": (6, -4), "Nb2O5": (6, -4)}
fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=200)
d = df.dropna(subset=["Ea", "EOT_penalty_nm"]).reset_index(drop=True)
for i, r in d.iterrows():
    ref = r.cls.startswith("reference")
    col = "black" if ref else ("#1F3864" if r.conductive else "#E0622B")
    face = col if not r.flag else "white"
    ax.scatter([r.Ea], [r.EOT_penalty_nm], marker=markers[i % len(markers)], s=70, color=face, edgecolors=col,
               linewidths=1.3, zorder=3, label=r.material)
    if r.EOT_is_bound:   # k of the oxide not in MP: plotted at the k = 3.9 upper bound, arrow = true value is lower
        ax.annotate("", xy=(r.Ea, r.EOT_penalty_nm - 0.25), xytext=(r.Ea, r.EOT_penalty_nm),
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
    ax.annotate(r.material + (" (k?)" if r.EOT_is_bound else ""), (r.Ea, r.EOT_penalty_nm),
                xytext=OFFS.get(r.material, (5, 5)), textcoords="offset points", fontsize=7)
tin = d[d.material == "TiN"].iloc[0]
ax.axvline(tin.Ea, color="black", ls="--", lw=0.8)
ax.set_xlabel("O migration barrier $E_m$ (eV)  [M3GNet CI-NEB, lower of O$_i$ / O$_X$+V$_X$]")
ax.set_ylabel("EOT penalty, 1 nm interlayer fully oxidised (nm)")
ax.set_title("Selection map: right = slower O diffusion, down = less capacitance loss", fontsize=9)
ax.set_xlim(-0.05, 3.05)
fig.text(0.01, 0.01, "filled: conductive (navy) / insulating (orange) / reference (black);  open: low-confidence structure;  "
         "(k?) + arrow: oxide k not in MP, plotted at k = 3.9 upper bound\n"
         "EOT penalty: literature thin-film k x Pilling-Bedworth expansion; TiN = measured EOT increase 2.34 nm (Miao 2026);  dashed line: TiN barrier",
         ha="left", va="bottom", fontsize=6.2)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(HERE, "selection_map.png"))
print("saved selection_map.png")
