"""Two-segment NEB for split interstitials: A -> M (relaxed midpoint) ; barrier = max(A->M, M->B)."""
import sys, warnings, json, numpy as np; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import neb
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE
name = sys.argv[1]
A = read(f"{name}/Oi/init.xyz"); B = read(f"{name}/Oi/final.xyz")
A.calc = neb.calc(); B.calc = neb.calc()
tmp = [A] + [A.copy() for _ in range(7)] + [B]
NEB(tmp).interpolate(method="idpp", mic=True)
M = tmp[4].copy(); neb.relax(M, steps=3000)
eA, eM = A.get_potential_energy(), M.get_potential_energy()
o = len(M) - 1; d = M.get_distances(o, range(o), mic=True); nn = np.argsort(d)[:4]
print(name, "M: dE(M-A) =", round(eM - eA, 3), "| O nn", [(M[j].symbol, round(d[j], 2)) for j in nn], flush=True)
def seg(X, Y, tag):
    imgs = [X] + [X.copy() for _ in range(7)] + [Y]
    for im in imgs: im.calc = neb.calc()
    n = NEB(imgs, climb=False, method="improvedtangent"); n.interpolate(method="idpp", mic=True)
    FIRE(n, maxstep=0.1, logfile=None).run(fmax=0.2, steps=1000)
    n.climb = True
    c = FIRE(n, maxstep=0.1, logfile=None).run(fmax=0.05, steps=2000)
    E = np.array([im.get_potential_energy() for im in imgs]) - eA
    f = float(np.linalg.norm(n.get_forces().reshape(7, -1, 3), axis=2).max())
    write(f"{name}/Oi/band_{tag}.xyz", imgs)
    print(name, tag, "conv", c, "| profile vs A", E.round(3).tolist(), "| max force", round(f, 3), flush=True)
    return E.max(), c
e1, c1 = seg(A, M, "A_M")
e2, c2 = seg(M, B, "M_B")
ea = float(max(e1, e2))
print(name, "O_i barrier (from A) =", round(ea, 3), "| converged", c1 and c2)
ov = json.load(open("results_overrides.json"))
ov.setdefault(name, {})["Oi"] = {"barrier_forward_eV": round(ea, 3), "converged": bool(c1 and c2),
    "note": f"two-segment A->M->B, M relaxed midpoint dE={eM-eA:.3f} eV; bands band_A_M.xyz, band_M_B.xyz"}
json.dump(ov, open("results_overrides.json", "w"), indent=1)
