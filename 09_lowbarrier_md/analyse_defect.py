"""Step 7 analysis: hop counting and collective MSD -> Arrhenius E_a(MD), compared with the NEB E_m.

Positions are block-averaged over 0.5 ps (10 frames) to remove vibration.  A hop is counted when an atom's
block-averaged position leaves its current rest site by more than half the NEB site-to-site hop distance
(Al2O3 hop is only 1.08 A, so a fixed 1.0 A threshold would miss hops; changed before the first analysis).
Primary E_a(MD) = Arrhenius fit of the collective-MSD diffusivity; the hop-rate fit is reported alongside.  Anion hops
(O, N) give the hop rate Gamma (1/ps); cation hops are reported only as a check that the host stayed intact.
Collective MSD: sum over anions of |r(t0+t) - r(t0)|^2 (time origins every 1 ps), slope over lag 5-25 ps.
E_a = -k_B d ln X / d(1/T) over 1000/1200/1400 K, X = collective D (primary) or Gamma (only if every T has >= 1 hop).
"""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
KB = 8.617333e-5; BLOCK = 10
HOP = {k: v['Oi']['hop_distance_A'] for k, v in json.load(open(os.path.join(ROOT, '03_neb', 'results.json'))).items() if 'Oi' in v}
summ = pd.read_csv(os.path.join(ROOT, "04_summary", "summary_table.csv")).set_index("material")
MK = {"Al2O3": "o", "HfO2": "s", "SiO2": "^", "Si3N4": "D", "Nb2O5": "v"}


def hops(c, THR):
    n = np.zeros(c.shape[1], int); rest = c[0].copy()
    for k in range(1, len(c)):
        m = np.linalg.norm(c[k] - rest, axis=1) > THR
        n[m] += 1; rest[m] = c[k][m]
    return n


res = {}
for d in sorted(glob.glob(os.path.join(HERE, "*_T*"))):
    if not os.path.exists(os.path.join(d, "state.npz")):
        continue
    s = np.load(os.path.join(d, "state.npz")); name, T = os.path.basename(d).rsplit("_T", 1); T = int(T)
    if int(s["step"]) < 55_000:
        print("incomplete:", os.path.basename(d), int(s["step"])); continue
    X = s["hist"].astype(float); t = s["thist"]; dt = float(t[1] - t[0]); tprod = float(t[-1] - t[0])
    sym = s["symbols"]; an = np.isin(sym, ["O", "N"])
    nb = len(X) // BLOCK; c = X[: nb * BLOCK].reshape(nb, BLOCK, *X.shape[1:]).mean(1)
    h = hops(c, 0.5 * HOP[name])
    step = int(round(1.0 / dt)); lags = np.arange(1, len(X) // 2); msd = np.zeros(len(lags))
    for k, L in enumerate(lags):
        dd = X[L::step][: len(X[:-L:step])][:, an] - X[:-L:step][:, an]
        msd[k] = (dd ** 2).sum(-1).sum(-1).mean()
    lag = lags * dt; w = (lag >= 5) & (lag <= 25)
    slope = np.polyfit(lag[w], msd[w], 1)[0]
    r = {"T": T, "t_prod_ps": tprod, "anion_hops": int(h[an].sum()), "cation_hops": int(h[~an].sum()),
         "atoms_that_hopped": int((h[an] > 0).sum()), "Gamma_per_ps": float(h[an].sum() / tprod),
         "collective_D_A2_per_ps": float(slope / 6)}
    res.setdefault(name, {})[T] = r
    print(f"{name} {T} K: anion hops {r['anion_hops']} ({r['atoms_that_hopped']} atoms), cation hops "
          f"{r['cation_hops']}, Gamma {r['Gamma_per_ps']:.3f}/ps, D_coll {r['collective_D_A2_per_ps']:.3f} A2/ps")

out = {}
fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=400)
for name, rr in res.items():
    Ts = np.array(sorted(rr)); G = np.array([rr[T]["Gamma_per_ps"] for T in Ts])
    o = {"by_T": rr, "E_m_NEB_eV": float(summ.loc[name, "Ea"])}
    Dc = np.array([rr[T]["collective_D_A2_per_ps"] for T in Ts])
    o["E_a_MD_hops_eV"] = float(-np.polyfit(1 / Ts, np.log(G), 1)[0] * KB) if len(Ts) >= 2 and (G > 0).all() else None
    if len(Ts) >= 2 and (Dc > 0).all():
        p = np.polyfit(1 / Ts, np.log(Dc), 1); o["E_a_MD_eV"] = float(-p[0] * KB)
        ax.plot(1000 / Ts, Dc, ls="--", lw=0.9, marker=MK.get(name, "o"), ms=4, color="#1F3864", mfc="white",
                label=f"{name}: MD {o['E_a_MD_eV']:.2f} / NEB {o['E_m_NEB_eV']:.2f} eV")
    else:
        o["E_a_MD_eV"] = None
    out[name] = o
ax.set_yscale("log"); ax.set_xlabel("1000 / T (K$^{-1}$)"); ax.set_ylabel(r"collective D (Å$^2$/ps)")
ax.legend(fontsize=6, frameon=False); ax.set_title("O$_i$ hop rate, M3GNet MD (50 ps per T)", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "lowbarrier_arrhenius.png"))
json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
for n, o in out.items():
    print(f"{n}: E_a(MD, MSD) {o['E_a_MD_eV']} | E_a(MD, hops) {o['E_a_MD_hops_eV']} | E_m(NEB) {o['E_m_NEB_eV']:.3f} eV")
