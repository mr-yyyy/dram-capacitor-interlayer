"""Step 4 v2: YSZ production re-run with weak thermostat coupling.

Why: v1 production used Langevin friction 0.01 /fs (tau = 100 fs). Strong velocity-randomising coupling damps
transport (Basconi & Shirts, J. Chem. Theory Comput. 9, 2887 (2013)). v2 continues from the end state of each v1
run (already equilibrated at T: 10 ps NPT + 100 ps NVT, cell from NPT) and runs a new 100 ps production with
Langevin friction 0.001 /fs (tau = 1 ps). Everything else as v1 (1 fs, O positions every 50 fs, checkpoints).
Output: T<T>_s<SEED>_v2/state.npz -> analyse.py v2
Usage: python md_v2.py T_K SEED
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, matgl
from ase import units, Atoms
from ase.md.langevin import Langevin
from matgl.ext.ase import PESCalculator

T, SEED = float(sys.argv[1]), int(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, f"T{int(T)}_s{SEED}", "state.npz")
OUT = os.path.join(HERE, f"T{int(T)}_s{SEED}_v2"); os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEV)
CALC = PESCalculator(matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2").to(DEV), stress_unit="eV/A3")
DT, N_NVT, REC, CKPT, FRICTION = 1.0 * units.fs, 100_000, 50, 5_000, 0.001
if os.environ.get("MD_SMOKE_TEST"):
    N_NVT, CKPT = 300, 100
state_fn = os.path.join(OUT, "state.npz")
log = open(os.path.join(OUT, "md.log"), "a")
s = np.load(state_fn if os.path.exists(state_fn) else SRC, allow_pickle=True)
atoms = Atoms(symbols=list(s["symbols"]), positions=s["positions"], cell=s["cell"], pbc=True)
atoms.set_momenta(s["momenta"]); atoms.calc = CALC
if os.path.exists(state_fn):
    step, Ohist, thist = int(s["step"]), list(s["Ohist"]), list(s["thist"])
    print(f"resume at step {step}", file=log, flush=True)
else:
    step, Ohist, thist = 0, [], []
    json.dump({"T": T, "seed": SEED, "start": f"end of T{int(T)}_s{SEED} (v1)", "friction_per_fs": FRICTION},
              open(os.path.join(OUT, "info.json"), "w"))
iO = np.array([i for i, x in enumerate(atoms.get_chemical_symbols()) if x == "O"])


def save():
    np.savez(state_fn, symbols=np.array(atoms.get_chemical_symbols()), positions=atoms.get_positions(),
             momenta=atoms.get_momenta(), cell=atoms.cell.array, step=step,
             Ohist=np.array(Ohist), thist=np.array(thist))


t0 = time.time()
md = Langevin(atoms, DT, temperature_K=T, friction=FRICTION / units.fs, rng=np.random.default_rng(SEED + 2000 + step))
while step < N_NVT:
    md.run(REC); step += REC
    Ohist.append(atoms.get_positions()[iO].astype(np.float32)); thist.append(step * 1e-3)
    if step % CKPT == 0:
        save()
        print(f"step {step} ({step/1000:.0f} ps) T {atoms.get_temperature():.0f} K | {(time.time()-t0)/60:.1f} min",
              file=log, flush=True)
save()
print("DONE", file=log, flush=True)
