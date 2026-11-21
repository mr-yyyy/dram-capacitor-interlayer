"""MP funnel for O-barrier interlayer candidates (TiN / interlayer / ZrO2).

Thermo type: GGA_GGA+U (MP2020-corrected PBE GGA/GGA+U hull) for ALL energies.
API key: environment variable MP_API_KEY (never stored in this repo).

Stages (results cached as CSV / JSON in this folder):
  N0  all compounds with exactly one anion X in {N, C, B, O} and 1-2 metals from
      {Al, Ti, Zr, Hf, Nb, Ta, Si, Y}; E_hull recomputed here from GGA_GGA+U entries
  N1  E_hull < 0.05 eV/atom  OR  experimentally observed (MP theoretical == False)
      (pass flag at 0.10 eV/atom also stored for the sensitivity check)
  N2  InterfacialReactivity(use_hull_energy=True) min reaction energy with TiN and
      with ZrO2 > -0.1 eV/atom   (per composition, not per polymorph)
  N3  data for EOT penalty: metallic or not, O-rich oxidation products,
      their PBE band gap and MP DFPT dielectric constant (threshold set later)
Run:  python funnel.py
"""
import os, json, pickle, itertools, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.interface_reactions import InterfacialReactivity
from pymatgen.core import Composition

THERMO = "GGA_GGA+U"
CRIT = {"thermo_types": [THERMO]}
METALS = ["Al", "Ti", "Zr", "Hf", "Nb", "Ta", "Si", "Y"]
ANIONS = ["N", "C", "B", "O"]
EHULL_N1, EHULL_SENS, ERXN_N2 = 0.05, 0.10, -0.10
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)


def chemsys(els):
    return "-".join(sorted(set(els)))


def get_entries(mpr, els):
    """GGA_GGA+U entries of a chemsys and all its subsystems, cached on disk."""
    fn = os.path.join(CACHE, f"entries_{chemsys(els)}.pkl")
    if os.path.exists(fn):
        with open(fn, "rb") as fh:
            return pickle.load(fh)
    es = mpr.get_entries_in_chemsys(sorted(set(els)), additional_criteria=CRIT)
    with open(fn, "wb") as fh:
        pickle.dump(es, fh)
    return es


def stage_n0(mpr):
    fn = os.path.join(HERE, "N0_all.csv")
    if os.path.exists(fn):
        return pd.read_csv(fn)
    rows = []
    for X in ANIONS:
        entries = get_entries(mpr, METALS + [X])
        systems = [[m, X] for m in METALS] + [[a, b, X] for a, b in itertools.combinations(METALS, 2)]
        for sysel in systems:
            sub = [e for e in entries if set(e.composition.chemical_system.split("-")) <= set(sysel)]
            pd_ = PhaseDiagram(sub)
            for e in sub:
                els = set(e.composition.chemical_system.split("-"))
                if els != set(sysel):          # each compound handled in its own chemsys only
                    continue
                rows.append(dict(material_id=e.data["material_id"], formula=e.composition.reduced_formula,
                                 chemsys=chemsys(els), anion=X, n_metals=len(els) - 1,
                                 e_hull=pd_.get_e_above_hull(e), e_form=pd_.get_form_energy_per_atom(e)))
        print(f"N0 anion {X}: {sum(r['anion'] == X for r in rows)} entries", flush=True)
    df = pd.DataFrame(rows).sort_values("e_hull").drop_duplicates("material_id")
    docs = mpr.materials.summary.search(
        material_ids=df.material_id.tolist(),
        fields=["material_id", "symmetry", "band_gap", "is_metal", "theoretical", "nsites", "energy_above_hull"])
    meta = {str(d.material_id): d for d in docs}
    df["spacegroup"] = [meta[m].symmetry.symbol if m in meta else None for m in df.material_id]
    df["band_gap"] = [meta[m].band_gap if m in meta else None for m in df.material_id]
    df["is_metal"] = [meta[m].is_metal if m in meta else None for m in df.material_id]
    df["theoretical"] = [meta[m].theoretical if m in meta else None for m in df.material_id]
    df["nsites"] = [meta[m].nsites if m in meta else None for m in df.material_id]
    df["e_hull_mp_default"] = [meta[m].energy_above_hull if m in meta else None for m in df.material_id]
    df = df.sort_values(["anion", "chemsys", "e_hull"]).reset_index(drop=True)
    df.to_csv(fn, index=False)
    return df


def stage_n1(df):
    df = df.copy()
    exp = df.theoretical == False  # noqa: E712
    df["pass_N1"] = (df.e_hull < EHULL_N1) | exp
    df["pass_N1_sens010"] = (df.e_hull < EHULL_SENS) | exp
    df.to_csv(os.path.join(HERE, "N1.csv"), index=False)
    return df


def min_rxn(mpr, comp, partner):
    pd_ = PhaseDiagram(get_entries(mpr, list(comp.as_dict()) + list(partner.as_dict())))
    ir = InterfacialReactivity(comp, partner, pd_, norm=True, use_hull_energy=True)
    x, e = ir.minimum
    return e, x


def stage_n2(mpr, df1, col="pass_N1_sens010"):
    fn = os.path.join(HERE, "N2.csv")
    done = pd.read_csv(fn) if os.path.exists(fn) else pd.DataFrame(columns=["formula"])
    formulas = sorted(set(df1[df1[col]].formula) - set(done.formula))
    rows = done.to_dict("records")
    for i, f in enumerate(formulas):
        c = Composition(f)
        r = {"formula": f}
        for name, p in (("TiN", Composition("TiN")), ("ZrO2", Composition("ZrO2"))):
            try:
                e, x = min_rxn(mpr, c, p)
                r[f"erxn_{name}"], r[f"x_{name}"] = e, x
            except Exception as ex:          # record, do not hide
                r[f"erxn_{name}"], r[f"err_{name}"] = None, repr(ex)[:200]
        rows.append(r)
        print(f"N2 {i+1}/{len(formulas)} {f}: TiN {r.get('erxn_TiN')}, ZrO2 {r.get('erxn_ZrO2')}", flush=True)
        pd.DataFrame(rows).to_csv(fn, index=False)
    d2 = pd.DataFrame(rows)
    d2["pass_N2"] = (d2.erxn_TiN > ERXN_N2) & (d2.erxn_ZrO2 > ERXN_N2)
    d2.to_csv(fn, index=False)
    return d2


def oxidation_products(mpr, f):
    """Stable phases at the O-rich limit (composition + large O excess), gases removed."""
    c = Composition(f)
    els = list(c.as_dict()) + ["O"]
    pd_ = PhaseDiagram(get_entries(mpr, els))
    big = c + Composition({"O": 50 * c.num_atoms})
    dec = pd_.get_decomposition(big)
    prods = [e for e in dec if e.composition.reduced_formula not in ("O2", "N2", "O3")]
    return [(e.data["material_id"], e.composition.reduced_formula) for e in prods]


def stage_n3(mpr, d2, df1):
    fn = os.path.join(HERE, "N3_data.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for f in sorted(d2[d2.pass_N2].formula):
        if f in out:
            continue
        out[f] = {"oxide_products": oxidation_products(mpr, f)}
        print("N3", f, out[f]["oxide_products"], flush=True)
        json.dump(out, open(fn, "w"), indent=1)
    ids = sorted({m for v in out.values() for m, _ in v["oxide_products"]} | set(df1[df1.formula.isin(out)].material_id))
    gaps = {str(d.material_id): d.band_gap for d in mpr.materials.summary.search(material_ids=ids, fields=["material_id", "band_gap"])}
    diel = {str(d.material_id): dict(e_total=d.e_total, e_ionic=d.e_ionic, e_electronic=d.e_electronic)
            for d in mpr.materials.dielectric.search(material_ids=ids, fields=["material_id", "e_total", "e_ionic", "e_electronic"])}
    json.dump({"thermo_type": THERMO, "band_gap": gaps, "dielectric": diel, "products": out},
              open(os.path.join(HERE, "N3_props.json"), "w"), indent=1, default=str)
    return out, gaps, diel


if __name__ == "__main__":
    with MPRester() as mpr:
        df0 = stage_n0(mpr)
        df1 = stage_n1(df0)
        print(f"N0 {len(df0)} | N1(0.05 or ICSD) {df1.pass_N1.sum()} | N1(0.10 or ICSD) {df1.pass_N1_sens010.sum()}")
        d2 = stage_n2(mpr, df1)
        print(f"N2 compositions tested {len(d2)} | pass {d2.pass_N2.sum()}")
        stage_n3(mpr, d2, df1)
        print("done")
