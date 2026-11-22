"""Step 7 analysis v2 (2026-11-22 check of v1 = analyse_defect.py, kept for the record).

Problem found in v1: the hop counter compared the 0.5 ps block-averaged position with a moving "rest" position and
counted every excursion beyond 0.5 x NEB hop distance.  When that threshold is close to the vibrational / librational
amplitude (Si3N4 threshold 0.65 A, SiO2 0.80 A) an atom that oscillates around the threshold is counted again and
again (Si3N4: 32-58 cation "hops" while no cation moved > 1.01 A between start and end; SiO2: 1319-1872 anion "hops").

v2 counts site-to-site transitions instead (site-based jump analysis with a residence criterion, as in
de Klerk, van der Maas & Wagemaker, ACS Appl. Energy Mater. (2018), doi 10.1021/acsaem.8b00457):
  - reference sites = anion positions averaged over the first 1 ps of production (lattice sites + the O_i site);
  - every 0.5 ps block-averaged anion position is assigned to its nearest reference site (minimum image);
  - a jump is counted only when the new site is held for >= 2 consecutive blocks (1 ps);
  - host check: cation and anion displacement between the first and last 1 ps.
  - a jump straight back to the site the atom just left is a return (it adds no long-range transport) and is
    excluded; E_a(MD) is fitted to the net (non-return) jump rate, weighted by the Poisson counts;
  - host intact = every cation moved < 1.2 A and the anion sublattice moved < 0.4 A on average (start vs end).
E_a(MD) from the net jump rate only if the host is intact and every T has >= 10 net jumps (Poisson error <= 32 %); otherwise "not determinable".
The summed anion MSD is reported as in v1 but only used where the host anion sublattice is intact.
"""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
KB = 8.617333e-5; BLOCK = 10; HOLD = 2; MIN_JUMPS = 10
summ = pd.read_csv(os.path.join(ROOT, "04_summary", "summary_table.csv")).set_index("material")
MK = {"Al2O3": "o", "HfO2": "s", "SiO2": "^", "Si3N4": "D", "Nb2O5": "v"}


def nearest(pos, ref, cell, inv):
    d = pos[:, None, :] - ref[None, :, :]
    f = d @ inv; f -= np.round(f); d = f @ cell
    return np.argmin((d ** 2).sum(-1), axis=1)


res = {}
for d in sorted(glob.glob(os.path.join(HERE, "*_T*"))):
    s = np.load(os.path.join(d, "state.npz")); name, T = os.path.basename(d).rsplit("_T", 1); T = int(T)
    X = s["hist"].astype(float); t = s["thist"]; dt = float(t[1] - t[0]); tprod = float(t[-1] - t[0])
    cell = s["cell"]; inv = np.linalg.inv(cell); an = np.isin(s["symbols"], ["O", "N"])
    nb = len(X) // BLOCK; c = X[: nb * BLOCK].reshape(nb, BLOCK, *X.shape[1:]).mean(1)[:, an]
    ref = c[:2].mean(0)
    site = np.array([nearest(p, ref, cell, inv) for p in c])          # (blocks, anions)
    jumps = 0; returns = 0; movers = set(); cur = site[0].copy(); prev = np.full(len(cur), -1)
    for k in range(1, len(site) - HOLD + 1):
        held = np.all(site[k:k + HOLD] == site[k], axis=0) & (site[k] != cur)
        for i in np.where(held)[0]:
            jumps += 1; returns += int(site[k][i] == prev[i]); movers.add(i); prev[i] = cur[i]; cur[i] = site[k][i]
    net = jumps - returns
    a = X[: int(1 / dt)].mean(0); b = X[-int(1 / dt):].mean(0); dr = np.linalg.norm(b - a, axis=1)
    step = int(round(1.0 / dt)); lags = np.arange(1, len(X) // 2); msd = np.zeros(len(lags))
    for k, L in enumerate(lags):
        dd = X[L::step][: len(X[:-L:step])][:, an] - X[:-L:step][:, an]
        msd[k] = (dd ** 2).sum(-1).sum(-1).mean()
    lag = lags * dt; w = (lag >= 5) & (lag <= 25)
    r = {"T": T, "t_prod_ps": tprod, "site_jumps": jumps, "return_jumps": returns, "net_jumps": net, "anions_that_jumped": len(movers),
         "net_jump_rate_per_ps": net / tprod, "summed_anion_D_A2_per_ps": float(np.polyfit(lag[w], msd[w], 1)[0] / 6),
         "cation_disp_mean_A": float(dr[~an].mean()), "cation_disp_max_A": float(dr[~an].max()),
         "anion_disp_mean_A": float(dr[an].mean()), "anions_disp_gt1A": int((dr[an] > 1).sum()), "n_anions": int(an.sum())}
    res.setdefault(name, {})[T] = r
    print(f"{name} {T} K: site jumps {jumps}, returns {returns}, net {net} ({len(movers)} anions) | cation disp mean {r['cation_disp_mean_A']:.2f} "
          f"max {r['cation_disp_max_A']:.2f} A | anion disp mean {r['anion_disp_mean_A']:.2f} A, >1 A: {r['anions_disp_gt1A']}/{r['n_anions']}")

out = {}
fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=400)
for name, rr in res.items():
    Ts = np.array(sorted(rr)); J = np.array([rr[T]["net_jumps"] for T in Ts]); G = np.array([rr[T]["net_jump_rate_per_ps"] for T in Ts])
    host_ok = all(rr[T]["cation_disp_max_A"] < 1.2 and rr[T]["anion_disp_mean_A"] < 0.4 for T in Ts)
    o = {"by_T": rr, "E_m_NEB_eV": float(summ.loc[name, "Ea"]), "host_intact": host_ok}
    if not host_ok:
        o["E_a_MD_jumps_eV"] = None
        o["reason"] = "host not intact (cation moved > 1.2 A or anion sublattice mean displacement > 0.4 A)"
    elif (J >= MIN_JUMPS).all():
        p, cov = np.polyfit(1 / Ts, np.log(G), 1, w=np.sqrt(J), cov="unscaled")
        o["E_a_MD_jumps_eV"] = float(-p[0] * KB); o["E_a_MD_jumps_err_eV"] = float(np.sqrt(cov[0, 0]) * KB)
        ax.errorbar(1000 / Ts, G, yerr=G / np.sqrt(J), ls="--", lw=0.9, marker=MK.get(name, "o"), ms=4, color="#1F3864",
                    mfc="white", capsize=2, label=f"{name}: MD {o['E_a_MD_jumps_eV']:.2f} ± {o['E_a_MD_jumps_err_eV']:.2f} / NEB {o['E_m_NEB_eV']:.2f} eV")
    else:
        o["E_a_MD_jumps_eV"] = None
        o["reason"] = f"fewer than {MIN_JUMPS} net jumps at some T: " + ", ".join(f"{T} K {j}" for T, j in zip(Ts, J))
    out[name] = o
ax.set_yscale("log"); ax.set_xlabel("1000 / T (K$^{-1}$)"); ax.set_ylabel("net anion site-jump rate (ps$^{-1}$)")
ax.legend(fontsize=6, frameon=False); ax.set_title("O$_i$-driven anion jumps, M3GNet MD (50 ps per T, 1 seed)", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "lowbarrier_arrhenius_v2.png"))
json.dump(out, open(os.path.join(HERE, "results_v2.json"), "w"), indent=1)
for n, o in out.items():
    e = f"{o['E_a_MD_jumps_eV']:.3f} ± {o['E_a_MD_jumps_err_eV']:.3f}" if o["E_a_MD_jumps_eV"] is not None else "n/a (" + o["reason"] + ")"
    print(f"{n}: E_a(MD, site jumps) {e} | E_m(NEB) {o['E_m_NEB_eV']:.3f} eV | host intact {o['host_intact']}")
