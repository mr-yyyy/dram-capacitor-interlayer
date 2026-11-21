"""Step 3: oxygen migration barrier by CI-NEB with M3GNet-PES-MatPES-PBE-2025.2 (ASE + matgl).

Host: relaxed bulk from 02_relax_elastic/<NAME>_relaxed.cif, cell fixed (lattice from step 2).
Supercell: repeats chosen so natoms in [150, 320] and the cell is as isotropic as possible.
Mechanisms (the lower barrier is the material's O migration barrier):
  Oi    : lowest-energy O interstitial found among every void type (grid search for maxima of
          the nearest-atom distance) and 9 split configurations around the central anion; the
          end point is the same defect mapped by a host space-group operation (nearest, > 0.5 A).
          (Oi_void = old definition: largest void -> nearest equivalent void.)
  ON_VN : O substitutes an anion site X, the nearest anion neighbour is removed (V_X);
          O hops into V_X.  (for an oxide host this is O-vacancy migration)
Endpoints: FIRE, positions only, fmax 0.02 eV/A.  NEB: 7 intermediate images, IDPP
interpolation, improved tangent, FIRE maxstep 0.1 A; stage 1 plain NEB to fmax 0.2 eV/A,
stage 2 climbing image to fmax 0.05 eV/A.
Neutral cells only (charge states are not representable with an MLIP).
Usage: python neb.py NAME ANION [Oi] [ON_VN]
"""
import os, sys, json, itertools, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, torch, matgl
from ase import Atom
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE
from ase.geometry import get_distances
from matgl.ext.ase import PESCalculator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEV)
N_IMAGES, FMAX_END, FMAX_NEB, MAX_NEB_STEPS = 7, 0.02, 0.05, 2000
POT = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2").to(DEV)


def calc():
    return PESCalculator(POT, stress_unit="eV/A3")


def choose_repeat(atoms, lo=150, hi=320):
    L = atoms.cell.lengths()
    best = None
    for r in itertools.product(range(1, 9), repeat=3):
        n = len(atoms) * np.prod(r)
        if not lo <= n <= hi:
            continue
        lens = L * np.array(r)
        score = (lens.max() / lens.min(), -lens.min())
        if best is None or score < best[0]:
            best = (score, r)
    return best[1]


def void_sites(unit, step=0.15):
    """Local maxima of the nearest-atom distance on a grid in the unit cell (fractional coords, radius)."""
    n = np.maximum((unit.cell.lengths() / step).astype(int), 8)
    g = np.stack(np.meshgrid(*[np.arange(k) / k for k in n], indexing="ij"), -1).reshape(-1, 3)
    cart = g @ unit.cell.array
    _, d = get_distances(cart, unit.positions, cell=unit.cell, pbc=True)
    r = d.min(1).reshape(n)
    mx = np.ones_like(r, bool)
    for sh in itertools.product((-1, 0, 1), repeat=3):
        if sh != (0, 0, 0):
            mx &= r >= np.roll(r, sh, axis=(0, 1, 2))
    idx = np.argwhere(mx)
    sites = [(idx_ / n, r[tuple(idx_)]) for idx_ in idx]
    sites.sort(key=lambda s: -s[1])
    uniq = []
    for f, rad in sites:      # merge grid points of the same void
        c = f @ unit.cell.array
        if all(get_distances(c, u[0] @ unit.cell.array, cell=unit.cell, pbc=True)[1].min() > 0.5 for u in uniq):
            uniq.append((f, rad))
    return uniq


def relax(atoms, fmax=FMAX_END, steps=2000):
    atoms.calc = calc()
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=steps)
    return atoms


def center_index(atoms, mask):
    c = atoms.cell.array.sum(0) / 2
    d = np.linalg.norm(atoms.positions - c, axis=1)
    d[~mask] = np.inf
    return int(np.argmin(d))


def endpoints_Oi(sc, unit, rep):
    voids = void_sites(unit)
    f0, r0 = voids[0]
    same = [v for v in voids if abs(v[1] - r0) < 0.05]
    # all equivalent void positions in the supercell
    pts = []
    for v, _ in same:
        for t in itertools.product(*[range(k) for k in rep]):
            pts.append(((v + np.array(t)) / np.array(rep)) @ sc.cell.array)
    pts = np.array(pts)
    c = sc.cell.array.sum(0) / 2
    i0 = int(np.argmin(np.linalg.norm(pts - c, axis=1)))
    _, d = get_distances(pts[i0], pts, cell=sc.cell, pbc=True)
    d = d[0]; d[d < 0.5] = np.inf
    i1 = int(np.argmin(d))
    A, B = sc.copy(), sc.copy()
    A.append(Atom("O", pts[i0])); B.append(Atom("O", pts[i1]))
    info = {"void_radius_A": float(r0), "hop_distance_A": float(d[i1]), "n_void_types": len(voids)}
    return A, B, info


def lowest_Oi(sc, unit, rep, anion, log):
    """Search the lowest-energy O interstitial configuration: every void type + split
    configurations around the central anion (O at 1.2 A in 9 directions). All relaxed (fmax 0.02)."""
    starts = {}
    shift = np.floor(np.array(rep) / 2)
    for k, (f, r) in enumerate(void_sites(unit)):
        a = sc.copy(); a.append(Atom("O", ((f + shift) / np.array(rep)) @ sc.cell.array)); starts[f"void{k}_r{r:.2f}"] = a
    iX = center_index(sc, np.array([s == anion for s in sc.get_chemical_symbols()]))
    dirs = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (-1, 0, 0), (0, 1, 1), (1, -1, 1)]
    rng = np.random.default_rng(0)
    dirs += [tuple(rng.normal(size=3)) for _ in range(3)]
    for v in dirs:
        v = np.array(v, float); v /= np.linalg.norm(v)
        a = sc.copy(); a.append(Atom("O", sc.positions[iX] + 1.2 * v)); starts[f"split_{anion}_{np.round(v, 2).tolist()}"] = a
    best, out = None, {}
    for k, a in starts.items():
        relax(a)
        e = a.get_potential_energy(); out[k] = e
        if best is None or e < best[1]:
            best = (k, e, a)
    e0 = min(out.values())
    log["Oi_site_search_dE_eV"] = {k: round(v - e0, 3) for k, v in sorted(out.items(), key=lambda kv: kv[1])}
    log["Oi_lowest_start"] = best[0]
    return best[2]


def equivalent_endpoint(A, host):
    for tol in (1.5, 2.0, 2.5):
        try:
            B, dist = _equivalent_endpoint(A, host, tol)
            return B, dist, tol
        except RuntimeError:
            continue
    raise RuntimeError("no symmetry-equivalent O_i image found (mapping tolerance up to 2.5 A)")


def _equivalent_endpoint(A, host, tol):
    """Apply host space-group operations to the relaxed defect cell A; return the image B with the
    same atom ordering whose O moved the least (but > 0.5 A)."""
    import spglib
    cell = (host.cell.array, host.get_scaled_positions(), host.numbers)
    ops = spglib.get_symmetry(cell, symprec=0.05)
    fA = A.get_scaled_positions(wrap=False); o = len(A) - 1
    syms = np.array(A.get_chemical_symbols())
    # defect centre: midpoint of O and a host atom bonded within 1.6 A (split), else O itself.
    # The end point must move this centre by > 1.0 A (a pure reorientation of a split is not migration).
    dh = A.get_distances(o, range(o), mic=True); jb = int(np.argmin(dh))
    def centre(f):
        if dh[jb] < 1.6:
            d = f[jb] - f[o]; d -= np.round(d); return f[o] + d / 2
        return f[o]
    cA = centre(fA)
    best = None
    for R, t in zip(ops["rotations"], ops["translations"]):
        fT = fA @ R.T + t
        dc = centre(fT) if dh[jb] >= 1.6 else None
        if dh[jb] < 1.6:   # centre of transformed split: transformed O and transformed bonded atom
            d = fT[jb] - fT[o]; d -= np.round(d); dc = fT[o] + d / 2
        dcen = dc - cA; dcen -= np.round(dcen); dist = np.linalg.norm(dcen @ A.cell.array)
        if dist < 1.0 or (best is not None and dist >= best[0]):
            continue
        perm = -np.ones(len(A), int); ok = True
        fH = host.get_scaled_positions(wrap=False)   # ideal host sites (A[i] <-> host[i] for i < o)
        for i in range(len(A) - 1):           # transformed atom i -> nearest same-species ideal host site
            d = fH - fT[i]; d -= np.round(d); dd = np.linalg.norm(d @ A.cell.array, axis=1)
            dd[syms[:-1] != syms[i]] = np.inf
            j = int(np.argmin(dd))
            if dd[j] > tol or perm[j] >= 0:
                ok = False; break
            perm[j] = i
        if not ok:
            continue
        best = (dist, perm, fT)
    if best is None:
        raise RuntimeError("no symmetry-equivalent O_i image found")
    dist, perm, fT = best
    B = A.copy()
    pos = A.get_scaled_positions(wrap=False)
    for j in range(len(A) - 1):
        d = fT[perm[j]] - fA[j]; d -= np.round(d); pos[j] = fA[j] + d
    d = fT[o] - fA[o]; d -= np.round(d); pos[o] = fA[o] + d
    B.set_scaled_positions(pos)
    return B, float(dist)


def endpoints_Oi_lowest(sc, unit, rep, anion):
    info = {}
    A = lowest_Oi(sc, unit, rep, anion, info)
    write(os.path.join(HERE, f"_lowest_Oi_{len(A)}_{A.get_chemical_formula()}.xyz"), A)
    B, dist, tol = equivalent_endpoint(A, sc)
    info["hop_distance_A"] = dist
    info["symmetry_mapping_tolerance_A"] = tol
    return A, B, info


def endpoints_ON_VN(sc, anion):
    mask = np.array([s == anion for s in sc.get_chemical_symbols()])
    i0 = center_index(sc, mask)
    _, d = get_distances(sc.positions[i0], sc.positions, cell=sc.cell, pbc=True)
    d = d[0].copy(); d[~mask] = np.inf; d[i0] = np.inf
    i1 = int(np.argmin(d))
    A = sc.copy(); A[i0].symbol = "O"; posB = sc.positions[i1].copy(); del A[i1]
    oi = i0 if i0 < i1 else i0 - 1          # index of the moving O after deleting the vacancy site
    B = A.copy(); B.positions[oi] = posB      # identical atom ordering in both endpoints
    return A, B, {"hop_distance_A": float(d[i1]), "moving_index": oi}


def run_neb(name, mech, A, B, info):
    out_dir = os.path.join(HERE, name, mech); os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    pA0, pB0 = A.positions.copy(), B.positions.copy()
    relax(A); relax(B)
    eA, eB = A.get_potential_energy(), B.get_potential_energy()
    info.update({"natoms": len(A), "E_init": eA, "E_final": eB,
                 "init_relax_max_shift_A": float(np.linalg.norm(A.positions - pA0, axis=1).max()),
                 "final_relax_max_shift_A": float(np.linalg.norm(B.positions - pB0, axis=1).max()),
                 "endpoint_separation_A": float(np.linalg.norm(A.positions - B.positions, axis=1).max())})
    write(os.path.join(out_dir, "init.xyz"), A); write(os.path.join(out_dir, "final.xyz"), B)
    log = os.path.join(out_dir, "neb.log")
    images, converged, nsteps, diverged = band(A, B, log)
    write(os.path.join(out_dir, "band.xyz"), images)
    E = np.array([im.get_potential_energy() for im in images])
    if diverged:
        # fallback (2026-11-21): split the path at the relaxed midpoint M; barrier = max over A->M->B
        M = images[len(images) // 2].copy()
        mid = [A] + [A.copy() for _ in range(N_IMAGES)] + [B]
        NEB(mid).interpolate(method="idpp", mic=True)
        M = mid[len(mid) // 2].copy(); relax(M, steps=3000)
        i1, c1, n1, d1 = band(A, M, log)
        i2, c2, n2, d2 = band(M, B, log)
        write(os.path.join(out_dir, "band_A_M.xyz"), i1); write(os.path.join(out_dir, "band_M_B.xyz"), i2)
        E = np.array([im.get_potential_energy() for im in i1] + [im.get_potential_energy() for im in i2[1:]])
        converged, nsteps = bool(c1 and c2 and not d1 and not d2), nsteps + n1 + n2
        info["fallback"] = f"single band diverged; two segments via relaxed midpoint M (dE_M = {M.get_potential_energy() - eA:.3f} eV)"
    info.update({"converged": bool(converged), "neb_steps": nsteps,
                 "E_profile_rel_eV": (E - E[0]).round(4).tolist(),
                 "barrier_forward_eV": float(E.max() - E[0]), "barrier_reverse_eV": float(E.max() - E[-1]),
                 "minutes": (time.time() - t0) / 60})
    return info


def band(A, B, log):
    """Two-stage NEB (plain to fmax 0.2, then CI to FMAX_NEB), FIRE maxstep 0.1.
    Divergence check: after 300 steps of a stage, fmax > 3 eV/A -> stop and report diverged."""
    images = [A] + [A.copy() for _ in range(N_IMAGES)] + [B]
    for im in images:
        im.calc = calc()
    neb = NEB(images, climb=False, method="improvedtangent")
    neb.interpolate(method="idpp", mic=True)
    nsteps, converged = 0, False
    for climb, fmax, maxsteps in ((False, 0.2, MAX_NEB_STEPS // 2), (True, FMAX_NEB, MAX_NEB_STEPS)):
        neb.climb = climb
        opt = FIRE(neb, maxstep=0.1, logfile=log)
        f = np.inf
        for k, _ in enumerate(opt.irun(fmax=fmax, steps=maxsteps)):
            f = np.sqrt((neb.get_forces() ** 2).sum(axis=1).max())
            if k > 300 and f > 3.0:
                return images, False, nsteps + k, True
        nsteps += opt.nsteps
        converged = f <= fmax
    return images, bool(converged), nsteps, False


if __name__ == "__main__":
    name, anion = sys.argv[1], sys.argv[2]
    mechs = sys.argv[3:] or ["Oi", "ON_VN"]
    unit = read(os.path.join(ROOT, "02_relax_elastic", f"{name}_relaxed.cif"))
    rep = choose_repeat(unit)
    sc = unit.repeat(rep)
    res_fn = os.path.join(HERE, "results.json")
    res = json.load(open(res_fn)) if os.path.exists(res_fn) else {}
    for mech in mechs:
        if mech == "Oi":            # lowest O_i site + symmetry-equivalent neighbour (changed 2026-11-20)
            A, B, info = endpoints_Oi_lowest(sc, unit, rep, anion)
        elif mech == "Oi_void":     # old definition: largest void -> nearest equivalent void
            A, B, info = endpoints_Oi(sc, unit, rep)
        else:
            A, B, info = endpoints_ON_VN(sc, anion)
        info["repeat"] = list(rep)
        r = run_neb(name, mech, A, B, info)
        res = json.load(open(res_fn)) if os.path.exists(res_fn) else {}   # re-read: other runs may have written
        res.setdefault(name, {})[mech] = r
        json.dump(res, open(res_fn, "w"), indent=1)
        print(f"{name} {mech}: Ea fwd {r['barrier_forward_eV']:.3f} rev {r['barrier_reverse_eV']:.3f} eV | "
              f"conv {r['converged']} in {r['neb_steps']} steps | dE(end) {r['E_final']-r['E_init']:.3f} | "
              f"{r['minutes']:.1f} min", flush=True)
