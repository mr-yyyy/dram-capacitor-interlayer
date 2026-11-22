"""YSZ oxygen diffusion: MSD (multiple time origins) -> D (Einstein, 3D) -> Arrhenius E_a.

MSD(t) = < |r_i(t0 + t) - r_i(t0)|^2 >  averaged over all O atoms and time origins t0 (every 1 ps)
D      = slope / 6, linear fit of MSD over lag 10-50 ps (production run of 100 ps, frames every 50 fs)
E_a    = -k_B * d ln D / d(1/T), fit over the seed-averaged D at 800/1000/1200 K
Figures follow the project style: one marker per temperature, dashed lines between computed points.
"""
import os, sys, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
KB = 8.617333e-5
LAG_FIT = (10.0, 50.0)
V2 = len(sys.argv) > 1 and sys.argv[1] == "v2"      # v2 = weak-coupling re-run (md_v2.py)
SUF = "_v2" if V2 else ""
NEED = 100_000 if V2 else 110_000


def msd(O, dt_ps, origin_every_ps=1.0):
    n = len(O); step = int(round(origin_every_ps / dt_ps))
    lags = np.arange(1, n // 2)
    out = np.zeros(len(lags))
    for k, L in enumerate(lags):
        d = O[L::step][: len(O[:-L:step])] - O[:-L:step]
        out[k] = (d ** 2).sum(-1).mean()
    return lags * dt_ps, out


res = {}
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200)
mk = {800: "o", 1000: "s", 1200: "^"}
for d in sorted(x for x in glob.glob(os.path.join(HERE, "T*_s*" + SUF)) if x.endswith("_v2") == V2):
    s = np.load(os.path.join(d, "state.npz"))
    if int(s["step"]) < NEED:
        print("incomplete:", os.path.basename(d), int(s["step"])); continue
    T = int(os.path.basename(d).split("_")[0][1:]); seed = int(os.path.basename(d).split("_")[1][1:])
    O = s["Ohist"].astype(float); t = s["thist"]; dt = float(t[1] - t[0])
    lag, m = msd(O, dt)
    w = (lag >= LAG_FIT[0]) & (lag <= LAG_FIT[1])
    slope, icpt = np.polyfit(lag[w], m[w], 1)
    D = slope / 6 * 1e-16 / 1e-12        # A^2/ps -> cm^2/s
    res.setdefault(T, []).append(D)
    ax[0].plot(lag, m, ls="--", lw=0.8, label=f"{T} K, seed {seed}")
    print(f"T {T} K seed {seed}: D = {D:.3e} cm2/s | MSD(50 ps) = {np.interp(50, lag, m):.2f} A^2")
ax[0].set_xlabel("lag time (ps)"); ax[0].set_ylabel("O MSD (Å$^2$)"); ax[0].legend(fontsize=6)
Ts = np.array(sorted(res)); Dm = np.array([np.mean(res[T]) for T in Ts])
if len(Ts) >= 2:
    x, y = 1 / Ts, np.log(Dm)
    p = np.polyfit(x, y, 1); Ea = -p[0] * KB
    yfit = np.polyval(p, x); r2 = 1 - ((y - yfit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    for T in Ts:
        ax[1].scatter([1000 / T] * len(res[T]), res[T], marker=mk.get(int(T), "o"), color="#1F3864", zorder=3)
    ax[1].plot(1000 / Ts, Dm, ls="--", color="#1F3864", lw=0.8)
    ax[1].set_yscale("log"); ax[1].set_xlabel("1000 / T (K$^{-1}$)"); ax[1].set_ylabel("D$_O$ (cm$^2$/s)")
    ax[1].set_title(f"8YSZ, M3GNet MD: E$_a$ = {Ea:.2f} eV (R$^2$ = {r2:.3f})", fontsize=8)
    out = {"T_K": Ts.tolist(), "D_cm2s_mean": Dm.tolist(), "D_all": {int(k): v for k, v in res.items()},
           "Ea_eV": Ea, "R2": r2, "fit_lag_ps": LAG_FIT}
    json.dump(out, open(os.path.join(HERE, f"arrhenius{SUF}.json"), "w"), indent=1)
    print(f"E_a = {Ea:.3f} eV, R2 = {r2:.4f}")
fig.tight_layout(); fig.savefig(os.path.join(HERE, f"ysz_msd_arrhenius{SUF}.png"))
