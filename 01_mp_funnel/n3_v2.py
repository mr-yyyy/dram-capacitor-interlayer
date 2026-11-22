"""N3 v2 (2026-11-22 audit): EOT penalty with literature k, oxide volume expansion and a consistent worst case.

Fixes against N3 v1 (n3_table.py, kept for the record):
 1. k of the oxide = literature value for ALD/thin-film dielectrics, as the user decided on 2026-11-20
    ("산화물 k는 문헌값"); v1 used MP DFPT k of the crystalline polymorph (e.g. Ta2O5 Pmmn, theoretical, k 51.8).
      SiO2 3.9, Si3N4 7, Al2O3 9, Ta2O5 22, TiO2 80, ZrO2 25, HfO2 25, Y2O3 15
        (Robertson, Eur. Phys. J. Appl. Phys. 28, 265 (2004), Table 1, as reproduced on globalsino.com page 2311)
      Nb2O5 41  ("the value of the dielectric constant given in the literature is 41": Fitzner & Stepien,
        Materials 16, 798 (2023), Introduction)
    Phases without a literature value: MP DFPT k as in v1; none at all -> k = 3.9 bound (flagged).
 2. Oxide thickness: 1 nm of interlayer converts to PBR x 1 nm of oxide, PBR = V(oxide products) / V(interlayer)
    per the same metal content (Pilling & Bedworth, J. Inst. Met. 29, 529 (1923)); volumes from MP structures.
    v1 assumed 1 nm of oxide.
 3. Insulating non-oxide interlayer: oxidised fraction f gives penalty (1-f)*3.9/k_self + f*PBR*3.9/k_ox, linear
    in f, so the worst case is max(self, oxide) -- v1 added both (a 1 nm layer cannot be fully intact and fully
    oxidised at the same time).
 4. TiN reference = measured EOT increase 2.339 nm (A5, Miao et al. APL 128, 022905 (2026), Fig. 3a), the N3 threshold.
    v1 showed TiN on the selection map at 0.110 nm (crystalline TiO2 C2/m, MP k 35.5), contradicting that decision.
Conductive interlayer: penalty = PBR * 3.9 / k_ox_eff.  Oxide interlayer: penalty = 3.9 / k_self (not oxidised further).
"""
import os, sys, json, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from funnel import get_entries

K_LIT = {"SiO2": 3.9, "Si3N4": 7.0, "Al2O3": 9.0, "Ta2O5": 22.0, "TiO2": 80.0, "ZrO2": 25.0, "HfO2": 25.0,
         "Y2O3": 15.0, "Nb2O5": 41.0}
TIN_REF = 2.339
v1 = pd.read_csv(os.path.join(HERE, "N3_table.csv"))


def products(mpr, f):
    c = Composition(f)
    keep = {el: a for el, a in c.get_el_amt_dict().items() if el not in ("N", "C")}
    els = sorted(set(keep) | {"O"})
    entries = [e for e in get_entries(mpr, list(c.get_el_amt_dict()) + ["O"])
               if set(e.composition.chemical_system.split("-")) <= set(els)]
    base = Composition(keep); tot = base + Composition({"O": 50 * base.num_atoms})
    dec = PhaseDiagram(entries).get_decomposition(tot)
    return [(str(e.data["material_id"]), e.composition.reduced_formula, fr * tot.num_atoms) for e, fr in dec.items()
            if e.composition.reduced_formula not in ("O2", "O3")]


with MPRester() as mpr:
    prods = {f: products(mpr, f) for f in v1.formula}
    ids = sorted({m for v in prods.values() for m, _, _ in v} | set(v1.material_id))
    vol = {str(d.material_id): d.volume / d.nsites for d in
           mpr.materials.summary.search(material_ids=ids, fields=["material_id", "volume", "nsites"])}

kmp = {}   # MP DFPT k by formula from the v1 table (fallback)
for _, r in v1.iterrows():
    for name, k in zip(str(r.oxide_products).split(" + "), str(r.oxide_k_MP).split(" / ")):
        if name != "nan" and k not in ("NA", "nan"):
            kmp[name.split("(")[0]] = float(k)

rows = []
for _, r in v1.iterrows():
    f = r.formula; comp = Composition(f)
    is_oxide = "O" in comp and not ({"N", "C", "B"} & set(comp.get_el_amt_dict()))
    k_self = K_LIT.get(f, None if pd.isna(r.k_self_MP) else float(r.k_self_MP))
    if f == "AlN":
        k_self = float(r.k_self_MP)   # no verified literature value; MP DFPT 8.7 (textbook ~9)
    pr = prods[f]
    V_self = vol[r.material_id] * comp.num_atoms
    V_ox = sum(vol[m] * n for m, _, n in pr)
    pbr = V_ox / V_self if pr else np.nan
    ks, src, bound = [], [], False
    for m, p, n in pr:
        if p in K_LIT:
            ks.append(K_LIT[p]); src.append(f"{p}:{K_LIT[p]:g}(lit)")
        elif p in kmp:
            ks.append(kmp[p]); src.append(f"{p}:{kmp[p]:g}(MP)")
        else:
            ks.append(3.9); src.append(f"{p}:3.9(bound)"); bound = True
    phi = np.array([vol[m] * n for m, _, n in pr]); phi = phi / phi.sum() if pr else phi
    k_ox = 1 / np.sum(phi / np.array(ks)) if pr else np.nan
    ins = not bool(r.conductive)
    eot_ox = pbr * 3.9 / k_ox if pr else np.nan
    if ins and is_oxide:
        sb = k_self is None
        pen = 3.9 / (k_self or 3.9); bound = sb; eot_ox = 0.0
    elif ins:
        sb = k_self is None
        pen = max(3.9 / (k_self or 3.9), eot_ox); bound = bound or sb
    else:
        pen = eot_ox
    if f == "TiN":
        pen = TIN_REF
    rows.append(dict(formula=f, material_id=r.material_id, conductive=r.conductive,
                     k_self=k_self, oxide_products=" + ".join(f"{p}({m})" for m, p, _ in pr), oxide_k_used=" / ".join(src),
                     k_ox_eff=round(k_ox, 2), PBR=round(pbr, 3), EOT_ox_per_nm=round(eot_ox, 3),
                     EOT_self_per_nm=None if not ins else round(3.9 / (k_self or 3.9), 3),
                     EOT_penalty_v2=round(pen, 3), EOT_is_bound=bool(bound),
                     EOT_penalty_v1=r.EOT_penalty_used, pass_N3=bool(pen <= TIN_REF + 1e-9),
                     note="measured A5 (whole TiN degradation)" if f == "TiN" else ""))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "N3_table_v2.csv"), index=False)
print("N3 v2:", int(df.pass_N3.sum()), "/", len(df), "pass (<= TiN 2.339 nm) | k bound used:", int(df.EOT_is_bound.sum()))
show = ["TiN", "AlN", "HfN", "ZrN", "TaN", "NbN", "TiC", "TaC", "NbC", "SiC", "Si3N4", "Al2O3", "HfO2", "SiO2",
        "TiO2", "Ta2O5", "Nb2O5", "Y2O3"]
print(df.set_index("formula").loc[[s for s in show if s in set(df.formula)],
      ["PBR", "oxide_k_used", "EOT_penalty_v1", "EOT_penalty_v2", "EOT_is_bound"]].to_string())
