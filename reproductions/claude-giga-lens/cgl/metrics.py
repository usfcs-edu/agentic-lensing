"""Benchmark metrics: rank-normalized R-hat/ESS (arviz), efficiency
(ESS per gradient / per second), mode-recovery metrics, logZ comparison and
the budget ledger.

Sample-array convention THROUGHOUT: samples are (T, C, dim) = (draws, chains,
parameters) in UNCONSTRAINED z-space (the tfp sample_chain layout used by
every reproduction in this lineage). arviz wants (chain, draw), so this
module transposes internally -- callers never do.

All functions are numpy/CPU; no jax.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# R-hat / ESS (rank-normalized, split -- arviz 1.x defaults)
# --------------------------------------------------------------------------- #
def rank_diagnostics(samples: np.ndarray, labels: Sequence[str],
                     mass_labels: Sequence[str]) -> dict:
    """Per-parameter rank-normalized split-R-hat + bulk/tail ESS + summaries.

    samples: (T, C, dim). Returns per-param arrays and the min/median/max
    summaries for the mass-label subset and for all parameters.
    """
    import arviz

    T, C, dim = samples.shape
    assert len(labels) == dim
    # arviz array layout is (chain, draw): transpose (T, C, dim) -> (dim, C, T).
    # NOTE the axis order matters enormously: feeding (draw, chain) silently
    # HIDES stuck-chain pathologies (each pseudo-chain then spans all real
    # chains, so between-"chain" variance collapses and R-hat ~ 1). This was
    # a real P2a bug, caught because the stored gu-2022 sys003 chains (known
    # max R-hat 2.07) came back clean; tests/test_zoo.py pins the regression.
    x = np.transpose(np.asarray(samples, dtype=np.float64), (2, 1, 0))
    rhat = np.empty(dim)
    ess_bulk = np.empty(dim)
    ess_tail = np.empty(dim)
    for k in range(dim):
        rhat[k] = float(arviz.rhat(x[k]))                  # method="rank"
        ess_bulk[k] = float(arviz.ess(x[k], method="bulk"))
        ess_tail[k] = float(arviz.ess(x[k], method="tail",
                                      prob=(0.05, 0.95)))

    mass_idx = [list(labels).index(m) for m in mass_labels]
    def _sub(a, idx):
        v = a[idx] if idx else a
        return dict(min=float(np.min(v)), median=float(np.median(v)),
                    max=float(np.max(v)))

    return dict(
        n_draws=int(T), n_chains=int(C), dim=int(dim),
        labels=list(labels), mass_labels=list(mass_labels),
        rhat=rhat.tolist(), ess_bulk=ess_bulk.tolist(),
        ess_tail=ess_tail.tolist(),
        summary=dict(
            rhat_all=_sub(rhat, list(range(dim))),
            rhat_mass=_sub(rhat, mass_idx),
            ess_bulk_all=_sub(ess_bulk, list(range(dim))),
            ess_bulk_mass=_sub(ess_bulk, mass_idx),
            ess_tail_mass=_sub(ess_tail, mass_idx),
        ),
    )


def efficiency(diag: dict, n_grad: int, n_logp: int, wall_s: float,
               hardware: str) -> dict:
    """ESS-per-gradient and per-wallclock-second, min over the mass subset
    and over all params (gradient counts supplied by the adapters)."""
    ess_mass_min = diag["summary"]["ess_bulk_mass"]["min"]
    ess_all_min = diag["summary"]["ess_bulk_all"]["min"]
    return dict(
        hardware=hardware,
        n_grad=int(n_grad), n_logp=int(n_logp), wall_s=float(wall_s),
        ess_mass_min=ess_mass_min, ess_all_min=ess_all_min,
        ess_per_grad_mass=ess_mass_min / max(n_grad, 1),
        ess_per_grad_all=ess_all_min / max(n_grad, 1),
        ess_per_sec_mass=ess_mass_min / max(wall_s, 1e-9),
        ess_per_sec_all=ess_all_min / max(wall_s, 1e-9),
    )


# --------------------------------------------------------------------------- #
# mode metrics
# --------------------------------------------------------------------------- #
def assign_modes_mahalanobis(Z: np.ndarray, centers: np.ndarray,
                             covs: Optional[np.ndarray] = None) -> np.ndarray:
    """Nearest-reference-mode assignment by Mahalanobis distance.

    Z: (..., dim); centers (K, dim); covs (K, dim, dim) or None (Euclidean).
    Returns int assignments with Z's leading shape.
    """
    lead = Z.shape[:-1]
    dim = Z.shape[-1]
    flat = np.asarray(Z, dtype=np.float64).reshape(-1, dim)
    K = centers.shape[0]
    d2 = np.empty((flat.shape[0], K))
    for k in range(K):
        r = flat - centers[k][None, :]
        if covs is None:
            d2[:, k] = np.sum(r * r, axis=1)
        else:
            sol = np.linalg.solve(covs[k], r.T)             # (dim, N)
            d2[:, k] = np.sum(r.T * sol, axis=0)
    return np.argmin(d2, axis=1).reshape(lead)


def assign_modes_param_threshold(values: np.ndarray,
                                 thresholds: Sequence[float]) -> np.ndarray:
    """Assignment by thresholding one physical parameter (T3 gamma split).
    K = len(thresholds)+1 ordered bins."""
    return np.digitize(np.asarray(values, dtype=np.float64),
                       np.asarray(thresholds, dtype=np.float64))


def assign_modes(reference, Z: Optional[np.ndarray] = None,
                 phys: Optional[dict] = None) -> np.ndarray:
    """Dispatch on reference.mode_assigner.

    mahalanobis      -> needs Z (..., dim) + reference.mode_centers/covs
    param_threshold  -> needs phys dict containing the named parameter,
                        with the SAME leading shape as the samples.
    """
    method = (reference.mode_assigner or {}).get("method", "mahalanobis")
    if method == "mahalanobis":
        assert Z is not None and reference.mode_centers is not None
        return assign_modes_mahalanobis(Z, np.asarray(reference.mode_centers),
                                        None if reference.mode_covs is None
                                        else np.asarray(reference.mode_covs))
    if method == "param_threshold":
        cfg = reference.mode_assigner
        assert phys is not None and cfg["param"] in phys
        return assign_modes_param_threshold(phys[cfg["param"]],
                                            cfg["thresholds"])
    raise ValueError(f"unknown mode_assigner method {method!r}")


def mode_occupancy(assign: np.ndarray, n_modes: int,
                   ref_weights: Optional[np.ndarray] = None) -> dict:
    """Recovery rate + occupancy vs reference weights."""
    flat = np.asarray(assign).ravel()
    occ = np.array([(flat == k).mean() for k in range(n_modes)])
    recovered = occ > 0.0
    out = dict(
        n_modes=int(n_modes),
        occupancy=occ.tolist(),
        recovered=[bool(r) for r in recovered],
        recovery_rate=float(recovered.mean()),
    )
    if ref_weights is not None:
        rw = np.asarray(ref_weights, dtype=np.float64)
        out["ref_weights"] = rw.tolist()
        out["max_abs_weight_error"] = float(np.max(np.abs(occ - rw)))
    return out


def count_mode_round_trips(assign_tc: np.ndarray) -> dict:
    """Inter-mode traffic per chain. assign_tc: (T, C) int assignments.

    A ROUND TRIP for a chain is a completed return to its starting mode after
    visiting at least one other mode (A->...->B->...->A counts 1).
    """
    a = np.asarray(assign_tc)
    T, C = a.shape
    transitions = np.zeros(C, dtype=int)
    round_trips = np.zeros(C, dtype=int)
    for c in range(C):
        col = a[:, c]
        trans = np.count_nonzero(col[1:] != col[:-1])
        transitions[c] = trans
        home = col[0]
        away = False
        rt = 0
        for v in col[1:]:
            if v != home:
                away = True
            elif away:
                rt += 1
                away = False
        round_trips[c] = rt
    return dict(
        transitions_per_chain=transitions.tolist(),
        round_trips_per_chain=round_trips.tolist(),
        total_transitions=int(transitions.sum()),
        total_round_trips=int(round_trips.sum()),
        n_migrating_chains=int((transitions > 0).sum()),
        n_chains=int(C),
    )


# --------------------------------------------------------------------------- #
# parallel-tempering round trips (P2c PT reference)
# --------------------------------------------------------------------------- #
def pt_walker_temps_from_adjacent(accepted_adjacent: np.ndarray) -> np.ndarray:
    """Reconstruct a SINGLE chain's walker temperature-slot trajectory from the
    TFP ReplicaExchangeMC per-step ACCEPTED adjacent swaps.

    accepted_adjacent: (T, R-1) bool; entry [t, j] = the swap between
    temperature slots j and j+1 was accepted (and therefore applied) at step t.
    Using the *accepted* (not proposed) swaps makes the reconstruction
    per-chain-correct: TFP proposes a shared even/odd parity but each batch
    chain accepts/rejects independently. Even/odd parity means the accepted
    transpositions within one step are NON-OVERLAPPING, so applying them in
    slot order is unambiguous. Returns (T, R) int where entry [t, w] = the
    temperature slot (0 = coldest / beta=1, R-1 = hottest / beta_min) occupied
    by walker w after step t.
    """
    acc = np.asarray(accepted_adjacent, dtype=bool)
    assert acc.ndim == 2, f"accepted_adjacent must be (T, R-1), got {acc.shape}"
    T, Rm1 = acc.shape
    R = Rm1 + 1
    occupant = np.arange(R)                    # occupant[slot] = walker id
    temps = np.empty((T, R), dtype=int)
    ident = np.arange(R)
    for t in range(T):
        for j in np.nonzero(acc[t])[0]:
            occupant[j], occupant[j + 1] = occupant[j + 1], occupant[j]
        temps[t, occupant] = ident             # invert: temp slot of each walker
    return temps


def count_pt_round_trips(temp_of_walker: np.ndarray) -> dict:
    """beta=1 round trips from a walker temperature-slot trajectory.

    temp_of_walker: (T, R) int, entry [t, w] = temperature slot of walker w
    (0 = coldest / beta=1, R-1 = hottest / beta_min). A ROUND TRIP for a walker
    is a completed coldest -> hottest -> coldest excursion (the standard PT
    mixing yardstick; total across walkers is the 23_pt_reference stop signal).
    """
    a = np.asarray(temp_of_walker, dtype=int)
    T, R = a.shape
    hot = R - 1
    round_trips = np.zeros(R, dtype=int)
    reached_hot = np.zeros(R, dtype=int)       # per-walker upper-end touches
    for w in range(R):
        last = 0                               # -1 cold end, +1 hot end, 0 none
        for v in a[:, w]:
            if v == 0:
                if last == 1:
                    round_trips[w] += 1
                last = -1
            elif v == hot:
                reached_hot[w] += 1
                last = 1
    return dict(
        round_trips_per_walker=round_trips.tolist(),
        total_round_trips=int(round_trips.sum()),
        n_walkers=int(R),
        n_walkers_reaching_hot=int((reached_hot > 0).sum()),
    )


# --------------------------------------------------------------------------- #
# evidence comparison
# --------------------------------------------------------------------------- #
def compare_logZ(logz_est: float, logz_ref: Optional[float],
                 logz_err: Optional[float] = None) -> dict:
    """logZ comparison helper (convention: logZ = log \\int e^{log_like} d prior,
    the cgl.zoo.api Reference.logZ definition)."""
    out = dict(logz_est=float(logz_est),
               logz_ref=None if logz_ref is None else float(logz_ref),
               logz_err=None if logz_err is None else float(logz_err))
    if logz_ref is not None:
        diff = float(logz_est - logz_ref)
        out["abs_diff"] = abs(diff)
        out["diff"] = diff
        if logz_err:
            out["n_sigma"] = abs(diff) / float(logz_err)
    return out


# --------------------------------------------------------------------------- #
# budget ledger
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class BudgetLedger:
    """Gradient/likelihood-eval counters, per phase.

    Convention (recorded with every cell): n_grad counts LOG-POSTERIOR
    gradient evaluations (1 per leapfrog step per chain for HMC-family;
    1 per sample per step for MAP/SVI); n_logp counts plain density
    evaluations not already implied by a counted gradient.
    """
    phases: dict = dataclasses.field(default_factory=dict)

    def add(self, phase: str, n_grad: int = 0, n_logp: int = 0, note: str = ""):
        p = self.phases.setdefault(phase, dict(n_grad=0, n_logp=0, note=note))
        p["n_grad"] += int(n_grad)
        p["n_logp"] += int(n_logp)
        if note:
            p["note"] = note

    @property
    def n_grad(self) -> int:
        return sum(p["n_grad"] for p in self.phases.values())

    @property
    def n_logp(self) -> int:
        return sum(p["n_logp"] for p in self.phases.values())

    def as_dict(self) -> dict:
        return dict(phases=self.phases, n_grad_total=self.n_grad,
                    n_logp_total=self.n_logp,
                    convention="n_grad = logp gradient evals (leapfrog steps "
                               "x chains for HMC; samples x steps for "
                               "MAP/SVI); n_logp = extra plain evals")
