"""Step 6: doped TiN, Ti(1-x)M(x)N -- substitution energy and O migration barrier next to the dopant.

Host: the same 216-atom TiN supercell (3x3x3 conventional, relaxed lattice from step 2, cell fixed) as step 3.
Dopant site: one Ti replaced by M (x = 1/108). The Ti chosen is the one nearest to the migrating atom at the
             saddle point of the undoped TiN band (step 3), so the hop passes the dopant directly.
Paths:       the converged undoped TiN endpoints (03_neb/TiN/<mech>/init.xyz, final.xyz) with Ti -> M; both
             mechanisms (Oi = split N-O dumbbell hop, ON_VN). Endpoints re-relaxed, then the same two-stage
             CI-NEB as step 3 (neb.py: 7 images, IDPP, improved tangent, fmax 0.05 eV/A, midpoint fallback).
Quantities:
  E_sub   = E(Ti107 M N108) - E(Ti108 N108) + mu_Ti - mu_M     (mu = M3GNet energy/atom of the MP ground-state
            elemental solid, cell relaxed)
  E_mix   = E(Ti107 M N108) + E(TiN f.u.) - E(Ti108 N108) - E(MN f.u.)   (only where the MN nitride was relaxed
            in step 2: AlN, ZrN, HfN, NbN_cubic, TaN_cubic). > 0: the dopant prefers to phase-separate.
  dE_O    = [E(O defect next to M) - E(doped bulk)] - [E(O defect, TiN) - E(TiN bulk)]  (< 0: O is bound by M)
  E_m     = max(E_band) - min(E_init, E_final)     (barrier out of the lower end point)
Dopants: Al, Si, Cr, Zr, Hf, Nb, Ta (Si: Micron patent US 11,289,487 doped TiN; Al: TiAlN; the rest: group 4-6
         nitride formers).  B was not included because B does not substitute on the Ti site.
Usage: python doped.py [M ...]        (DOPED_FETCH_ONLY=1: download elemental structures and stop)
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ase.io import read, write

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOPANTS = sys.argv[1:] or ["Al", "Si", "Cr", "Zr", "Hf", "Nb", "Ta"]
ELEM_FN = os.path.join(HERE, "elements_mp.json")
NITRIDE = {"Al": "AlN", "Zr": "ZrN", "Hf": "HfN", "Nb": "NbN_cubic", "Ta": "TaN_cubic"}

# ---- elemental reference structures (MP ground state, downloaded once) ----------------------------------
if not os.path.exists(ELEM_FN):
    from mp_api.client import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor
    out = {}
    with MPRester(os.environ["MP_API_KEY"]) as m:
        for el in ["Ti"] + ["Al", "Si", "Cr", "Zr", "Hf", "Nb", "Ta"]:
            docs = m.materials.summary.search(chemsys=el, fields=["material_id", "energy_above_hull", "structure"])
            d = min(docs, key=lambda x: x.energy_above_hull)
            a = AseAtomsAdaptor.get_atoms(d.structure)
            out[el] = {"mp_id": str(d.material_id), "symbols": a.get_chemical_symbols(),
                       "positions": a.positions.tolist(), "cell": a.cell.array.tolist()}
            print(el, d.material_id, len(a), flush=True)
    json.dump(out, open(ELEM_FN, "w"), indent=1)
if os.environ.get("DOPED_FETCH_ONLY"):
    sys.exit(0)

sys.path.insert(0, os.path.join(ROOT, "03_neb"))
import neb                                   # loads M3GNet-PES-MatPES-PBE-2025.2 on the GPU
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
neb.HERE = HERE                              # run_neb writes <HERE>/<name>/<mech>/
res_fn = os.path.join(HERE, "results.json")
res = json.load(open(res_fn)) if os.path.exists(res_fn) else {}


def energy(a):
    a.calc = neb.calc(); return float(a.get_potential_energy())


def mu(el):
    if "mu" in res and el in res["mu"]:
        return res["mu"][el]
    d = json.load(open(ELEM_FN))[el]
    a = Atoms(d["symbols"], positions=d["positions"], cell=d["cell"], pbc=True); a.calc = neb.calc()
    FIRE(FrechetCellFilter(a), logfile=None).run(fmax=0.01, steps=2000)
    res.setdefault("mu", {})[el] = float(a.get_potential_energy() / len(a)); save()
    return res["mu"][el]


def save():
    cur = json.load(open(res_fn)) if os.path.exists(res_fn) else {}
    cur.update(res); json.dump(cur, open(res_fn, "w"), indent=1)


unit = read(os.path.join(ROOT, "02_relax_elastic", "TiN_relaxed.cif"))
rep = neb.choose_repeat(unit); host = unit.repeat(rep)
E_host = energy(host.copy())
fu = lambda name: (lambda a: energy(a) / (len(a) / 2))(read(os.path.join(ROOT, "02_relax_elastic", f"{name}_relaxed.cif")))
E_TiN_fu = E_host / (len(host) / 2)
ref = json.load(open(os.path.join(ROOT, "03_neb", "results.json")))["TiN"]

paths = {}
for mech in ["Oi", "ON_VN"]:
    d = os.path.join(ROOT, "03_neb", "TiN", mech)
    A0, B0 = read(os.path.join(d, "init.xyz")), read(os.path.join(d, "final.xyz"))
    band = read(os.path.join(d, "band.xyz"), ":")
    ts = band[int(np.argmax([im.get_potential_energy() for im in band]))]
    mov = len(A0) - 1 if mech == "Oi" else int(ref[mech]["moving_index"])
    ti = [i for i, s in enumerate(A0.get_chemical_symbols()) if s == "Ti"]
    dd = ts.get_distances(mov, ti, mic=True)
    i_sub = ti[int(np.argmin(dd))]
    # O-defect energy in pure TiN relative to its own bulk (O_i: +O; ON_VN: N -> O and one V_N)
    paths[mech] = {"A": A0, "B": B0, "i_sub": i_sub, "d_saddle_A": float(dd.min()),
                   "E_def_TiN": float(A0.get_potential_energy() if A0.calc else energy(A0.copy())) - E_host}

if os.environ.get("DOPED_DRY"):              # set-up check only
    print({m: {k: v for k, v in p.items() if k not in ("A", "B")} for m, p in paths.items()}, E_host, mu("Ti"), mu("Al"))
    sys.exit(0)
for M in DOPANTS:
    key = f"Ti107{M}N108"
    if key in res and all(m in res[key] for m in ("Oi", "ON_VN")):
        print("skip", key, flush=True); continue
    t0 = time.time()
    # dopant bulk: same Ti site as the Oi path (all Ti sites are equivalent in the pure host)
    db = host.copy(); db[paths["Oi"]["i_sub"]].symbol = M
    neb.relax(db, fmax=0.02, steps=3000); E_db = float(db.get_potential_energy())
    r = res.get(key, {})
    r.update({"x": 1 / 108, "E_sub_eV": E_db - E_host + mu("Ti") - mu(M),
              "E_mix_eV": (E_db + E_TiN_fu - E_host - fu(NITRIDE[M])) if M in NITRIDE else None,
              "mu_ref_mp_id": json.load(open(ELEM_FN))[M]["mp_id"]})
    res[key] = r; save()
    for mech in ["Oi", "ON_VN"]:
        if mech in r:
            continue
        p = paths[mech]
        A, B = p["A"].copy(), p["B"].copy()
        A[p["i_sub"]].symbol = M; B[p["i_sub"]].symbol = M
        info = {"dopant_site_to_saddle_A": p["d_saddle_A"], "repeat": list(rep)}
        out = neb.run_neb(key, mech, A, B, info)
        E = np.array(out["E_profile_rel_eV"])
        out["E_m_eV"] = float(E.max() - min(0.0, E[-1]))
        out["dE_O_vs_TiN_eV"] = (out["E_init"] - E_db) - p["E_def_TiN"]
        r[mech] = out; res[key] = r; save()
        print(f"{key} {mech}: E_m {out['E_m_eV']:.3f} eV (fwd {out['barrier_forward_eV']:.3f}, rev "
              f"{out['barrier_reverse_eV']:.3f}) | dE_O {out['dE_O_vs_TiN_eV']:+.3f} | conv {out['converged']}", flush=True)
    r["E_m_eV"] = min(r[m]["E_m_eV"] for m in ("Oi", "ON_VN") if r[m]["converged"]) if any(
        r[m]["converged"] for m in ("Oi", "ON_VN")) else None
    r["minutes"] = (time.time() - t0) / 60; res[key] = r; save()
    print(f"{key}: E_sub {r['E_sub_eV']:+.3f} eV | E_mix {r['E_mix_eV']} | E_m {r['E_m_eV']} eV | "
          f"{r['minutes']:.1f} min", flush=True)
print("DOPED DONE", flush=True)
