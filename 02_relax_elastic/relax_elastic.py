"""Step 2: relaxation check + elastic constants with M3GNet-PES-MatPES-PBE-2025.2.

Relax: FIRE + FrechetCellFilter, fmax 0.001 eV/A (cell + positions).
Elastic: 6 Voigt strains x {-1, -0.5, +0.5, +1 %}, internal positions relaxed at fixed cell
(fmax 0.001 eV/A); C_ij = d(sigma_i)/d(eps_j) by linear fit; VRH averages via pymatgen.
Small cells -> CPU (desktop benchmark: CPU faster below ~12 atoms).
Usage: python relax_elastic.py NAME [NAME ...]   (names defined in MATERIALS below)
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, matgl
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.io import write
from matgl.ext.ase import PESCalculator
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.io.ase import AseAtomsAdaptor

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cpu"
torch.set_default_device(DEV)
EV_A3_TO_GPA = 160.21766208
STRAINS = [-0.01, -0.005, 0.005, 0.01]
FMAX = 0.001


def mp_structure(mid):
    from mp_api.client import MPRester
    with MPRester() as m:
        s = m.get_structure_by_material_id(mid, conventional_unit_cell=True)
    return AseAtomsAdaptor.get_atoms(s)


# name -> (builder, reference note for comparison)
MATERIALS = {
    "TiN":        (lambda: bulk("TiN", "rocksalt", a=4.24, cubic=True), "rocksalt Fm-3m (mp-492)"),
    "AlN":        (lambda: bulk("AlN", "wurtzite", a=3.11, c=4.98), "wurtzite P6_3mc (mp-661)"),
    "ZrO2_mono":  (lambda: mp_structure("mp-2858"), "monoclinic P2_1/c (mp-2858)"),
    "ZrO2_tet":   (lambda: mp_structure("mp-2574"), "tetragonal P4_2/nmc (mp-2574)"),
    # conductive candidates in the deposited rocksalt phase (starting a is only an initial guess)
    "TaN_cubic":  (lambda: bulk("TaN", "rocksalt", a=4.35, cubic=True), "rocksalt Fm-3m (mp-251)"),
    "NbN_cubic":  (lambda: bulk("NbN", "rocksalt", a=4.40, cubic=True), "rocksalt Fm-3m (mp-1580)"),
    "HfN":        (lambda: bulk("HfN", "rocksalt", a=4.52, cubic=True), "rocksalt Fm-3m (mp-2828)"),
    "ZrN":        (lambda: bulk("ZrN", "rocksalt", a=4.58, cubic=True), "rocksalt Fm-3m (mp-1352)"),
}
# N4 representatives (01_mp_funnel/N4_representatives.csv): MP conventional cells
import csv as _csv
_rep = os.path.join(os.path.dirname(HERE), "01_mp_funnel", "N4_representatives.csv")
if os.path.exists(_rep):
    for _r in _csv.DictReader(open(_rep)):
        if _r["formula"] not in ("AlN", "HfN", "ZrN", "TaN", "NbN"):      # already defined above
            MATERIALS[_r["formula"]] = ((lambda mid=_r["material_id"]: mp_structure(mid)),
                                        f"{_r['spacegroup']} ({_r['material_id']})")


def voigt_strain_matrix(j, e):
    """Symmetric strain tensor for Voigt component j (0..5) with engineering shear e."""
    eps = np.zeros((3, 3))
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    a, b = pairs[j]
    if a == b:
        eps[a, a] = e
    else:
        eps[a, b] = eps[b, a] = e / 2
    return eps


def stress_voigt(atoms):
    s = atoms.get_stress(voigt=True)  # eV/A3, ASE order xx yy zz yz xz xy, tensile positive
    return s * EV_A3_TO_GPA


def run(name, calc):
    atoms = MATERIALS[name][0]()
    atoms.calc = calc
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    opt.run(fmax=FMAX, steps=3000)
    write(os.path.join(HERE, f"{name}_relaxed.cif"), atoms)
    cellpar = atoms.cell.cellpar()
    res = {"name": name, "ref": MATERIALS[name][1], "natoms": len(atoms),
           "relax_steps": opt.nsteps, "cellpar": [round(x, 4) for x in cellpar],
           "volume_per_atom": atoms.get_volume() / len(atoms),
           "energy_per_atom": atoms.get_potential_energy() / len(atoms),
           "residual_stress_GPa": stress_voigt(atoms).round(3).tolist()}
    cell0, pos0 = atoms.cell.array.copy(), atoms.get_scaled_positions().copy()
    C = np.zeros((6, 6))
    for j in range(6):
        sig = []
        for e in STRAINS:
            a = atoms.copy(); a.calc = calc
            F = np.eye(3) + voigt_strain_matrix(j, e)
            a.set_cell(cell0 @ F.T, scale_atoms=False)
            a.set_scaled_positions(pos0)
            FIRE(a, logfile=None).run(fmax=FMAX, steps=2000)
            sig.append(stress_voigt(a))
        sig = np.array(sig)
        for i in range(6):
            C[i, j] = np.polyfit(STRAINS, sig[:, i], 1)[0]
    C = 0.5 * (C + C.T)
    et = ElasticTensor.from_voigt(C)
    B, G = et.k_vrh, et.g_vrh
    res.update({"C_GPa": C.round(1).tolist(), "eig_C_min": float(np.linalg.eigvalsh(C).min()),
                "B_VRH": B, "G_VRH": G, "E": 9 * B * G / (3 * B + G), "Pugh_B_over_G": B / G,
                "poisson": (3 * B - 2 * G) / (2 * (3 * B + G))})
    return res


if __name__ == "__main__":
    pot = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2").to(DEV)
    calc = PESCalculator(pot, stress_unit="eV/A3")   # default GPa is wrong for ASE cell relax
    out_fn = os.path.join(HERE, "results.json")
    out = json.load(open(out_fn)) if os.path.exists(out_fn) else {}
    for name in sys.argv[1:]:
        r = run(name, calc)
        out[name] = r
        json.dump(out, open(out_fn, "w"), indent=1)
        print(f"{name}: cell {r['cellpar']} | B {r['B_VRH']:.0f} G {r['G_VRH']:.0f} E {r['E']:.0f} GPa "
              f"| B/G {r['Pugh_B_over_G']:.2f} | min eig(C) {r['eig_C_min']:.0f}", flush=True)
