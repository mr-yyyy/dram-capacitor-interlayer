"""Laptop environment check: GPU, M3GNet load, TiN relax, force-call speed.
Run:  python check_env.py
"""
import time, warnings, numpy as np, torch, matgl
warnings.filterwarnings("ignore")
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| device:", dev,
      torch.cuda.get_device_name(0) if dev == "cuda" else "(no NVIDIA GPU -> CPU)")
if dev == "cuda":
    print("compute capability", torch.cuda.get_device_capability(0),
          "| VRAM GB", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
torch.set_default_device(dev)
from matgl.ext.ase import PESCalculator
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
pot = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2").to(dev)
calc = PESCalculator(pot, stress_unit="eV/A3")   # IMPORTANT: default is GPa -> wrong for ASE cell relax
a = bulk("TiN", "rocksalt", a=4.30); a.calc = calc
FIRE(FrechetCellFilter(a), logfile=None).run(fmax=0.005, steps=500)
print("TiN a =", round(a.cell.lengths()[0] * 2**0.5, 4), "A  (expected ~4.249; exp 4.24)")
for n in (2, 3):
    s = bulk("TiN", "rocksalt", a=4.25, cubic=True).repeat((n, n, n)); s.calc = calc; s.get_forces()
    ts = []
    for _ in range(20):
        s.positions += np.random.normal(0, 1e-3, s.positions.shape)
        if dev == "cuda": torch.cuda.synchronize()
        t = time.perf_counter(); s.get_forces()
        if dev == "cuda": torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    print(f"{len(s)} atoms: force call median {np.median(ts):.3f} s  (desktop RTX5060 ref: 0.04-0.05 s idle-ish, 0.15-0.17 s when GPU shared)")
