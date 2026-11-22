"""Step 7: MD cross-check of the low NEB barriers (E_m <= 0.8 eV): Al2O3, HfO2, SiO2, Si3N4, Nb2O5 (all O_i).

Start:  the relaxed NEB initial state 03_neb/<NAME>/Oi/init.xyz (host supercell + one O interstitial, neutral).
MD:     timestep 1 fs; 5 ps NPT (Berendsen, P = 0, taut 100 fs, taup 1000 fs, compressibility 1/B with B from
        step 2) to set the lattice at T, then 50 ps NVT (Langevin) production (thermostat: see next line).
        T = 1000, 1200, 1400 K (hop rate at E_m 0.3-0.5 eV gives >~5 hops in 50 ps).
        Langevin friction 0.001 /fs (weak coupling, tau = 1 ps): strong Langevin coupling damps transport
        (Basconi & Shirts, J. Chem. Theory Comput. 9, 2887 (2013)); changed from 0.01 /fs before the first run.
        Waits while 06_ysz_md/v2_running.flag exists (GPU is shared with the YSZ re-run).
Output: unwrapped positions of all atoms every 50 fs during production (state.npz; checkpoint every 5 ps,
        rerunning resumes).  Hops and MSD are counted by analyse_defect.py.
Usage:  python md_defect.py NAME T_K
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, matgl
from ase import units, Atoms
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.nptberendsen import NPTBerendsen
from ase.md.langevin import Langevin
from matgl.ext.ase import PESCalculator

NAME, T = sys.argv[1], float(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, f"{NAME}_T{int(T)}"); os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEV)
CALC = PESCalculator(matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2").to(DEV), stress_unit="eV/A3")
DT = 1.0 * units.fs
N_NPT, N_NVT, REC, CKPT = 5_000, 50_000, 50, 5_000
if os.environ.get("MD_SMOKE_TEST"):
    N_NPT, N_NVT, CKPT = 100, 200, 100
B_GPa = float(pd.read_csv(os.path.join(ROOT, "04_summary", "summary_table.csv")).set_index("material").loc[NAME, "B"])
state_fn = os.path.join(OUT, "state.npz")
log = open(os.path.join(OUT, "md.log"), "a")
SEED = 7
while os.path.exists(os.path.join(ROOT, "06_ysz_md", "v2_running.flag")):
    time.sleep(60)

if os.path.exists(state_fn):
    s = np.load(state_fn, allow_pickle=True)
    atoms = Atoms(symbols=list(s["symbols"]), positions=s["positions"], cell=s["cell"], pbc=True)
    atoms.set_momenta(s["momenta"]); atoms.calc = CALC
    step = int(s["step"]); hist = list(s["hist"]); thist = list(s["thist"])
    print(f"resume at step {step}", file=log, flush=True)
else:
    atoms = read(os.path.join(ROOT, "03_neb", NAME, "Oi", "init.xyz")); atoms.calc = CALC
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=np.random.default_rng(SEED)); Stationary(atoms)
    step, hist, thist = 0, [], []
    json.dump({"name": NAME, "T": T, "natoms": len(atoms), "formula": atoms.get_chemical_formula(),
               "B_GPa": B_GPa, "start": f"03_neb/{NAME}/Oi/init.xyz"}, open(os.path.join(OUT, "info.json"), "w"))


def save():
    np.savez(state_fn, symbols=np.array(atoms.get_chemical_symbols()), positions=atoms.get_positions(),
             momenta=atoms.get_momenta(), cell=atoms.cell.array, step=step, hist=np.array(hist), thist=np.array(thist))


t0 = time.time()
npt = NPTBerendsen(atoms, DT, temperature_K=T, pressure_au=0.0, taut=100 * units.fs, taup=1000 * units.fs,
                   compressibility_au=1.0 / (B_GPa * units.GPa))
nvt = None
while step < N_NPT + N_NVT:
    if step < N_NPT:
        npt.run(REC)
    else:
        if nvt is None:
            nvt = Langevin(atoms, DT, temperature_K=T, friction=0.001 / units.fs,
                           rng=np.random.default_rng(SEED + 1000 + step))
        nvt.run(REC)
        hist.append(atoms.get_positions().astype(np.float32)); thist.append((step + REC - N_NPT) * 1e-3)
    step += REC
    if step % CKPT == 0:
        save()
        print(f"step {step} ({step/1000:.0f} ps) T {atoms.get_temperature():.0f} K | V/atom "
              f"{atoms.get_volume()/len(atoms):.3f} | {(time.time()-t0)/60:.1f} min", file=log, flush=True)
save()
print("DONE", file=log, flush=True)
