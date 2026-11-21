"""Step 5: thermal expansion by phonopy quasi-harmonic approximation (QHA), M3GNet-PES-MatPES-PBE-2025.2.

Per material: relaxed cell from 02_relax_elastic/<NAME>_relaxed.cif
  7 volumes V0 x {0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03}; at each volume the cell shape and atoms are
  relaxed at constant volume (FrechetCellFilter(constant_volume=True), fmax 0.001 eV/A)
  phonopy finite displacements 0.01 A, supercell >= 10 A per axis, mesh 24x24x24 (scaled by cell length)
  thermal properties 0-1200 K; PhonopyQHA (Vinet EOS) -> volumetric beta(T); linear alpha = beta/3
  (for non-cubic cells alpha is the orientation average).
Imaginary modes (min mesh frequency < -0.1 THz) at any volume are flagged: QHA is not valid there.
CTE is used only for relative comparison (HANDOFF).
Usage: python cte.py NAME [NAME ...]
"""
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, torch, matgl
from ase.io import read
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from matgl.ext.ase import PESCalculator
from phonopy import Phonopy, PhonopyQHA
from phonopy.structure.atoms import PhonopyAtoms

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEV)
POT = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2").to(DEV)
CALC = PESCalculator(POT, stress_unit="eV/A3")
SCALES = [0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03]
T = np.arange(0, 1210, 10)


def to_ph(a):
    return PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=a.cell.array, scaled_positions=a.get_scaled_positions())


def phonons(a):
    n = [int(np.ceil(10.0 / L)) for L in a.cell.lengths()]
    ph = Phonopy(to_ph(a), supercell_matrix=np.diag(n))
    ph.generate_displacements(distance=0.01)
    forces = []
    from ase import Atoms
    for sc in ph.supercells_with_displacements:
        s = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        s.calc = CALC
        f = s.get_forces()
        forces.append(f - f.mean(axis=0))      # remove drift
    ph.forces = forces
    ph.produce_force_constants()
    mesh = [max(4, int(round(60 / L))) for L in a.cell.lengths()]   # ~ 60 A / a k-points per axis
    ph.run_mesh(mesh)
    fmin = float(ph.get_mesh_dict()["frequencies"].min())
    ph.run_thermal_properties(t_min=T[0], t_max=T[-1], t_step=10)
    tp = dict(ph.get_thermal_properties_dict())
    # phonopy reports thermal properties per PRIMITIVE cell; E and V are per the input unit cell -> rescale
    k = len(a) / len(ph.primitive)
    for key in ("free_energy", "entropy", "heat_capacity"):
        tp[key] = np.asarray(tp[key]) * k
    return tp, fmin, n, mesh


def run(name):
    t0 = time.time()
    a0 = read(os.path.join(ROOT, "02_relax_elastic", f"{name}_relaxed.cif"))
    V0 = a0.get_volume()
    cache = os.path.join(HERE, f"cache_{name}.json")
    if os.path.exists(cache):
        c = json.load(open(cache))
    else:
        c = {"vols": [], "E": [], "F": [], "S": [], "CV": [], "fmins": []}
        for s in SCALES:
            a = a0.copy()
            a.set_cell(a0.cell.array * s ** (1 / 3), scale_atoms=True)
            a.calc = CALC
            FIRE(FrechetCellFilter(a, constant_volume=True), logfile=None).run(fmax=0.001, steps=2000)
            tp, fmin, n, mesh = phonons(a)
            c["vols"].append(a.get_volume()); c["E"].append(float(a.get_potential_energy()))
            c["F"].append(tp["free_energy"].tolist()); c["S"].append(tp["entropy"].tolist())
            c["CV"].append(tp["heat_capacity"].tolist()); c["fmins"].append(fmin)
            c["T"] = tp["temperatures"].tolist(); c["supercell"] = n; c["mesh"] = mesh
        json.dump(c, open(cache, "w"))
    vols, E, fmins, n, mesh = c["vols"], c["E"], c["fmins"], c["supercell"], c["mesh"]
    F, S, CV = np.array(c["F"]).T, np.array(c["S"]).T, np.array(c["CV"]).T       # (n_T, n_V)
    qha = PhonopyQHA(volumes=vols, electronic_energies=E, temperatures=c["T"],
                     free_energy=F, cv=CV, entropy=S, eos="vinet", t_max=1000)
    Tq = np.array(c["T"][: len(qha.thermal_expansion)])
    beta = np.array(qha.thermal_expansion)
    alpha = beta / 3
    a300 = float(np.interp(300, Tq, alpha))
    m = (Tq >= 300) & (Tq <= 900)
    res = {"name": name, "natoms_unit": len(a0), "supercell": n, "mesh": mesh, "V0": V0, "volumes": vols,
           "E_V": E, "min_freq_THz_per_volume": fmins, "imaginary_modes": bool(min(fmins) < -0.1),
           "alpha_300K_1e-6": a300 * 1e6, "alpha_mean_300_900K_1e-6": float(alpha[m].mean() * 1e6),
           "alpha_T": {"T": Tq[::10].tolist(), "alpha_1e-6": (alpha[::10] * 1e6).round(3).tolist()},
           "B0_GPa_static": float(np.ravel(qha.bulk_modulus)[0]),
           "minutes": (time.time() - t0) / 60}
    return res


if __name__ == "__main__":
    fn = os.path.join(HERE, "results.json")
    for name in sys.argv[1:]:
        r = run(name)
        out = json.load(open(fn)) if os.path.exists(fn) else {}
        out[name] = r
        json.dump(out, open(fn, "w"), indent=1)
        print(f"{name}: alpha(300K) {r['alpha_300K_1e-6']:.2f} | <alpha>300-900K {r['alpha_mean_300_900K_1e-6']:.2f} x1e-6/K | "
              f"min freq {min(r['min_freq_THz_per_volume']):.2f} THz | imaginary {r['imaginary_modes']} | {r['minutes']:.1f} min",
              flush=True)
