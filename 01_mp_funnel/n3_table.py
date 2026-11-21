"""N3 table: metallic or not, O-rich oxidation products, their PBE band gap and MP DFPT k,
EOT penalty per nm (no threshold applied here; threshold is a user decision).

Rules (recorded in the condition-change log):
  * N and C are assumed to leave as N2 / CO2 gas: products = O-rich equilibrium of the
    metal(+B) part in the (metal, B)-O system (first version kept nitrate phases such as
    Ti(NO3)4 from the full-system O-rich limit -> changed 2026-11-20)
  * k_poly = MP DFPT e_total (served as polycrystalline average); None if MP has no DFPT entry
  * several products: volume-fraction-weighted series combination 1/k_eff = sum(phi_i / k_i)
  * EOT penalty per nm:  insulating interlayer (MP band gap > 0): 3.9/k_self + 3.9/k_ox_eff
                         conductive interlayer (band gap = 0):    3.9/k_ox_eff
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition

HERE = os.path.dirname(os.path.abspath(__file__))
d1 = pd.read_csv(os.path.join(HERE, "N1.csv"))
d2 = pd.read_csv(os.path.join(HERE, "N2.csv"))
props = json.load(open(os.path.join(HERE, "N3_props.json")))
GAS_ONLY = ({"N", "O"}, {"C", "O"}, {"N"}, {"O"}, {"C"})

import sys
sys.path.insert(0, HERE)
from funnel import get_entries
from pymatgen.analysis.phase_diagram import PhaseDiagram


def oxidation_products_no_NC(mpr, f):
    """O-rich products assuming N and C leave as N2 / CO2 gas: equilibrium of the
    metal(+B) part with excess O in the (metal, B)-O system. Avoids nitrate/carbonate phases."""
    c = Composition(f)
    keep = {el: amt for el, amt in c.get_el_amt_dict().items() if el not in ("N", "C")}
    if not keep:
        return []
    els = sorted(set(keep) | {"O"})
    entries = [e for e in get_entries(mpr, list(c.get_el_amt_dict()) + ["O"])
               if set(e.composition.chemical_system.split("-")) <= set(els)]
    pd_ = PhaseDiagram(entries)
    base = Composition(keep)
    dec = pd_.get_decomposition(base + Composition({"O": 50 * base.num_atoms}))
    return [(e.data["material_id"], e.composition.reduced_formula) for e in dec
            if e.composition.reduced_formula not in ("O2", "O3")]


with MPRester() as _m:
    prods_all = {f: oxidation_products_no_NC(_m, f) for f in props["products"]}
ids = sorted({str(m) for v in prods_all.values() for m, _ in v})

with MPRester() as mpr:
    sdocs = {str(d.material_id): d for d in mpr.materials.summary.search(
        material_ids=ids, fields=["material_id", "band_gap", "volume", "nsites", "formula_pretty"])}
    # k of the candidate itself: use its own lowest-E_hull N1-passing polymorph
    own = (d1[d1.pass_N1].sort_values("e_hull").drop_duplicates("formula").set_index("formula"))
    diel = {str(d.material_id): d for d in mpr.materials.dielectric.search(
        material_ids=sorted(set(ids) | set(own.material_id)), fields=["material_id", "e_total"])}


def kpoly(mid):
    d = diel.get(mid)
    if d is None or d.e_total is None:
        return None
    e = np.array(d.e_total, dtype=float)
    return float(e) if e.ndim == 0 else float(np.mean(np.linalg.eigvalsh(e)))   # MP serves e_total as poly average


rows = []
for f in sorted(d2[d2.pass_N2].formula):
    if f not in own.index:          # passed N2 only via the 0.10 set
        continue
    o = own.loc[f]
    gap = float(o.band_gap)
    prods = prods_all.get(f, [])
    # volume fractions of the product phases from the decomposition stoichiometry (approx: per formula unit volume)
    vols, ks, gaps, names = [], [], [], []
    for m, p in prods:
        s = sdocs.get(str(m))
        vols.append(s.volume / s.nsites if s else np.nan)
        ks.append(kpoly(str(m))); gaps.append(s.band_gap if s else None); names.append(f"{p}({m})")
    if prods and all(k is not None for k in ks):
        phi = np.array(vols) / np.nansum(vols)
        k_eff = 1 / np.sum(phi / np.array(ks))
    else:
        k_eff = None
    k_self = kpoly(o.material_id)
    ins = gap > 0
    eot_self = (3.9 / k_self if k_self else None) if ins else 0.0
    is_oxide = "O" in Composition(f).get_el_amt_dict() and not ({"N", "C", "B"} & set(Composition(f).get_el_amt_dict()))
    eot_ox = 0.0 if (is_oxide and ins) else (3.9 / k_eff if k_eff else None)   # an oxide interlayer is not oxidized further
    rows.append(dict(formula=f, material_id=o.material_id, spacegroup=o.spacegroup, e_hull=round(o.e_hull, 3),
                     theoretical=o.theoretical, band_gap_PBE=round(gap, 2), conductive=not ins,
                     k_self_MP=None if k_self is None else round(k_self, 1),
                     erxn_TiN=round(d2.set_index("formula").loc[f, "erxn_TiN"], 3),
                     erxn_ZrO2=round(d2.set_index("formula").loc[f, "erxn_ZrO2"], 3),
                     oxide_products=" + ".join(names),
                     oxide_gap_min_PBE=None if not gaps or None in gaps else round(min(gaps), 2),
                     oxide_k_MP=" / ".join("NA" if k is None else f"{k:.1f}" for k in ks),
                     k_ox_eff=None if k_eff is None else round(k_eff, 1),
                     EOT_self_per_nm=None if eot_self is None else round(eot_self, 3),
                     EOT_ox_per_nm=None if eot_ox is None else round(eot_ox, 3)))
df = pd.DataFrame(rows)
df["EOT_penalty_per_nm"] = df.EOT_self_per_nm + df.EOT_ox_per_nm
df.to_csv(os.path.join(HERE, "N3_table.csv"), index=False)
print(len(df), "compositions | conductive", int(df.conductive.sum()), "| k missing (self or oxide):",
      int(df.EOT_penalty_per_nm.isna().sum()))
