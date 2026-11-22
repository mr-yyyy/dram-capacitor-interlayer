"""Validation vs experiment (audit 2026-11-22): M3GNet relaxed rocksalt lattice constant vs experimental a.
Experimental a: Holec et al., Phys. Rev. B 85, 064101 (2012) [arXiv:1111.2737v1], Table 3, a_exp from ICDD PDF-2 (2007):
TiN 03-065-0565, ZrN 00-035-0753, HfN 00-033-0592, NbN 03-065-5011, TaN 03-065-9404."""
import os, json
from ase.io import read
H = os.path.dirname(os.path.abspath(__file__))
EXP = {"TiN": 4.241, "ZrN": 4.578, "HfN": 4.525, "NbN_cubic": 4.389, "TaN_cubic": 4.358}
out = {}
for n, ae in EXP.items():
    a = read(os.path.join(H, f"{n}_relaxed.cif")); L = a.cell.cellpar()
    ac = (a.get_volume() / len(a) * 8) ** (1 / 3)        # conventional rocksalt cell has 8 atoms
    out[n] = {"a_M3GNet": round(ac, 4), "a_exp": ae, "err_pct": round((ac / ae - 1) * 100, 2)}
    print(f"{n:10s} M3GNet {ac:.3f} | exp {ae:.3f} | {out[n]['err_pct']:+.2f} %")
json.dump(out, open(os.path.join(H, "lattice_vs_exp.json"), "w"), indent=1)
