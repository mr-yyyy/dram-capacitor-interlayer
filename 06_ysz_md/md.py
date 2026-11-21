"""Step 4: method validation -- oxygen diffusion in 8 mol% YSZ by M3GNet MD (ASE + matgl).

Structure: fluorite ZrO2 3x3x3 conventional (108 cations, 216 O); 16 Zr -> Y and 8 O removed
           (8 mol% Y2O3: Y/(Zr+Y) = 0.148; charge balance 2 Y : 1 V_O); Y and V_O placed at random (seed).
           Relaxed at 0 K (cell + positions, fmax 0.02 eV/A) -> 316 atoms.
MD:        timestep 1 fs; 10 ps NPT (Berendsen, P = 0, taut 100 fs, taup 1000 fs) to set the lattice at T,
           then 100 ps NVT (Langevin, friction 0.01 /fs) production. Total 110 ps per run.
Output:    unwrapped O positions every 50 fs during production -> MSD -> D (Einstein) -> Arrhenius (analyse.py).
Checkpoint every 5 ps (state.npz); rerunning the same command resumes.
Usage: python md.py T_K SEED
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, matgl
from ase import units
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.nptberendsen import NPTBerendsen
from ase.md.langevin import Langevin
from matgl.ext.ase import PESCalculator

T, SEED = float(sys.argv[1]), int(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, f"T{int(T)}_s{SEED}"); os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEV)
CALC = PESCalculator(matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2").to(DEV), stress_unit="eV/A3")
DT = 1.0 * units.fs
N_NPT, N_NVT, REC, CKPT = 10_000, 100_000, 50, 5_000
if os.environ.get("MD_SMOKE_TEST"):          # short functional test only
    N_NPT, N_NVT, CKPT = 200, 300, 100
state_fn = os.path.join(OUT, "state.npz")
log = open(os.path.join(OUT, "md.log"), "a")


def build():
    rng = np.random.default_rng(SEED)
    a = bulk("ZrO2", "fluorite", a=5.12, cubic=True).repeat((3, 3, 3))
    zr = [i for i, s in enumerate(a.get_chemical_symbols()) if s == "Zr"]
    o = [i for i, s in enumerate(a.get_chemical_symbols()) if s == "O"]
    for i in rng.choice(zr, 16, replace=False):
        a[i].symbol = "Y"
    del a[sorted(rng.choice(o, 8, replace=False), reverse=True)]
    a.calc = CALC
    FIRE(FrechetCellFilter(a), logfile=None).run(fmax=0.02, steps=1500)
    return a


if os.path.exists(state_fn):
    s = np.load(state_fn, allow_pickle=True)
    from ase import Atoms
    atoms = Atoms(symbols=list(s["symbols"]), positions=s["positions"], cell=s["cell"], pbc=True)
    atoms.set_momenta(s["momenta"]); atoms.calc = CALC
    step = int(s["step"]); Ohist = list(s["Ohist"]); thist = list(s["thist"])
    print(f"resume at step {step}", file=log, flush=True)
else:
    atoms = build()
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=np.random.default_rng(SEED)); Stationary(atoms)
    step, Ohist, thist = 0, [], []
    json.dump({"T": T, "seed": SEED, "natoms": len(atoms), "formula": atoms.get_chemical_formula(),
               "a_relaxed_A": (atoms.get_volume() ** (1 / 3) / 3)}, open(os.path.join(OUT, "info.json"), "w"))
iO = np.array([i for i, x in enumerate(atoms.get_chemical_symbols()) if x == "O"])


def save():
    np.savez(state_fn, symbols=np.array(atoms.get_chemical_symbols()), positions=atoms.get_positions(),
             momenta=atoms.get_momenta(), cell=atoms.cell.array, step=step,
             Ohist=np.array(Ohist), thist=np.array(thist))


t0 = time.time()
npt = NPTBerendsen(atoms, DT, temperature_K=T, pressure_au=0.0, taut=100 * units.fs, taup=1000 * units.fs,
                   compressibility_au=0.8)          # ~1/B, B ~ 1.25 eV/A^3 (200 GPa)
nvt = None
while step < N_NPT + N_NVT:
    if step < N_NPT:
        npt.run(REC)
    else:
        if nvt is None:
            nvt = Langevin(atoms, DT, temperature_K=T, friction=0.01 / units.fs,
                           rng=np.random.default_rng(SEED + 1000 + step))
        nvt.run(REC)
        Ohist.append(atoms.get_positions()[iO].astype(np.float32)); thist.append((step + REC - N_NPT) * 1e-3)  # ps
    step += REC
    if step % CKPT == 0:
        save()
        Tinst = atoms.get_temperature()
        print(f"step {step} ({step/1000:.0f} ps) T {Tinst:.0f} K | V/atom {atoms.get_volume()/len(atoms):.3f} | "
              f"{(time.time()-t0)/60:.1f} min", file=log, flush=True)
save()
print("DONE", file=log, flush=True)
