"""Numerical research checks for MATH.md; not a learned layer or CFD solver.

Run from the repository root: python research/check_math.py
The fixed setups, thresholds and seeds below precede outcome measurement.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
ROT = np.array([[0.0, -1.0], [1.0, 0.0]])
SEED = 20261007


def relative_error(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def spectrum(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular**2
    total = float(energy.sum())
    return {
        "rank_relative_tolerance_1e-10": int(np.sum(singular > singular[0] * 1e-10)) if total else 0,
        "energy_rank_95": int(np.searchsorted(np.cumsum(energy), 0.95 * total) + 1) if total else 0,
        "frobenius_norm": float(np.sqrt(total)),
        "singular_values": singular.tolist(),
    }


def derivative_fd(fun, x, epsilon):
    eye = np.eye(x.size)
    return np.column_stack([(fun(x + epsilon * e) - fun(x - epsilon * e)) / (2 * epsilon) for e in eye])


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def positions(n, rng):
    width = int(np.ceil(np.sqrt(n)))
    q = np.array([(k % width, k // width) for k in range(n)], dtype=float)
    return (1.5 * q + rng.uniform(-0.15, 0.15, (n, 2))).ravel()


def vortex_velocity(q, gamma):
    q = q.reshape(-1, 2)
    r = q[:, None, :] - q[None, :, :]
    distance2 = np.einsum("ijk,ijk->ij", r, r)
    np.fill_diagonal(distance2, np.inf)
    return np.sum((r @ ROT.T) * (gamma[None, :] / (2 * np.pi * distance2))[:, :, None], axis=1).ravel()


def vortex_jacobian(q, gamma):
    q = q.reshape(-1, 2)
    n = len(q)
    r = q[:, None, :] - q[None, :, :]
    distance2 = np.einsum("ijk,ijk->ij", r, r)
    np.fill_diagonal(distance2, np.inf)
    kernel = ROT[None, None, :, :] / distance2[:, :, None, None]
    kernel -= 2 * np.einsum("ija,ijb->ijab", r @ ROT.T, r) / distance2[:, :, None, None] ** 2
    weighted = kernel * (gamma / (2 * np.pi))[None, :, None, None]
    blocks = -weighted.copy()
    blocks[np.arange(n), np.arange(n)] = weighted.sum(axis=1)
    return blocks.transpose(0, 2, 1, 3).reshape(2 * n, 2 * n)


def trajectory(fun, x, times):
    sol = solve_ivp(lambda t, z: fun(z), (0, float(times[-1])), x, t_eval=times,
                    method="DOP853", rtol=2e-12, atol=2e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def tangent_trajectory(fun, jac, x, times):
    n = x.size

    def rhs(t, state):
        point = state[:n]
        fundamental = state[n:].reshape(n, n)
        return np.concatenate([fun(point), (jac(point) @ fundamental).ravel()])

    sol = solve_ivp(rhs, (0, float(times[-1])), np.concatenate([x, np.eye(n).ravel()]),
                    t_eval=times, method="DOP853", rtol=2e-12, atol=2e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    matrices = sol.y[n:].T.reshape(len(times), n, n)
    return sol.y[:n].T, matrices


def hopf_flow_and_tangent(q, time, mu, a=1.0, omega=2.0):
    radius2 = float(q @ q)
    alpha = 2 * a * time if mu == 0 else a * np.expm1(2 * mu * time) / mu
    denominator = 1 + alpha * radius2
    gain = np.exp(mu * time) / np.sqrt(denominator)
    turn = rotation(omega * time)
    next_q = gain * (turn @ q)
    tangent = gain * turn @ (np.eye(2) - alpha / denominator * np.outer(q, q))
    return next_q, tangent


def hopf_jacobian(q, mu, a=1.0, omega=2.0):
    return mu * np.eye(2) + omega * ROT - a * ((q @ q) * np.eye(2) + 2 * np.outer(q, q))


def operator_dictionary(q, m=32, a=1.0, kappa=1.0):
    """Three shared full-size matrices, coefficients constrained to a 2D cone."""
    size = m + 2
    b0, b1, b2 = (np.zeros((size, size)) for _ in range(3))
    b0[:2, :2] = -2 * a * np.eye(2)
    b0[2:, 2:] = -kappa * np.eye(m)
    b1[:2, :2] = -a * np.diag([1.0, -1.0])
    b2[:2, :2] = -a * np.array([[0.0, 1.0], [1.0, 0.0]])
    x, y = q
    coefficients = np.array([x * x + y * y, x * x - y * y, 2 * x * y])
    return coefficients, coefficients[0] * b0 + coefficients[1] * b1 + coefficients[2] * b2


def main():
    rng = np.random.default_rng(SEED)
    checks = []

    def check(name, error, upper_bound):
        checks.append({"name": name, "value": float(error), "upper_bound": float(upper_bound),
                       "passed": bool(np.isfinite(error) and error <= upper_bound)})

    q = positions(8, rng)
    gamma = np.linspace(0.8, 1.2, 8)
    vf = lambda x: vortex_velocity(x, gamma)
    vj = lambda x: vortex_jacobian(x, gamma)
    derivative_errors = {str(e): relative_error(derivative_fd(vf, q, e), vj(q))
                         for e in (1e-3, 1e-4, 1e-5)}
    check("vortex_analytic_jacobian_vs_central_difference", derivative_errors["1e-05"], 1e-8)
    check("vortex_trace_zero", abs(np.trace(vj(q))), 1e-12)
    translations = np.tile(np.eye(2), (8, 1))
    check("vortex_translation_nullspace", np.linalg.norm(vj(q) @ translations), 1e-12)

    instantaneous = []
    for n in (4, 8, 16):
        base = positions(n, rng)
        strengths = np.linspace(0.8, 1.2, n)
        baseline = vortex_jacobian(base, strengths)
        ranks, energy_ranks = [], []
        for event in range(30):
            site = event % n
            angle = 2 * np.pi * event / 30
            altered = base.copy()
            altered[2 * site:2 * site + 2] += 0.07 * np.array([np.cos(angle), np.sin(angle)])
            delta = vortex_jacobian(altered, strengths) - baseline
            spec = spectrum(delta)
            ranks.append(spec["rank_relative_tolerance_1e-10"])
            energy_ranks.append(spec["energy_rank_95"])
        check(f"one_vortex_full_translation_reduced_rank_n{n}", max(abs(r - (2 * n - 2)) for r in ranks), 0)
        instantaneous.append({"vortices": n, "state_dimension": 2 * n, "event_count": 30,
                              "ranks": ranks, "energy_ranks_95": energy_ranks})

    prepared = q.copy()
    prepared[:2] += np.array([0.07, 0.0])
    delay, horizon = 1.0, 1.5
    baseline_at_probe = trajectory(vf, q, np.array([delay]))[-1]
    prepared_at_probe = trajectory(vf, prepared, np.array([delay]))[-1]
    times = np.linspace(0.05, horizon, 30)
    _, phi0 = tangent_trajectory(vf, vj, baseline_at_probe, times)
    _, phia = tangent_trajectory(vf, vj, prepared_at_probe, times)
    response0, responsea = phi0.reshape(-1, 16), phia.reshape(-1, 16)
    finite_delta = responsea - response0
    finite_difference_errors = []
    for _ in range(3):
        probe = rng.normal(size=16)
        probe /= np.linalg.norm(probe)
        epsilon = 1e-5
        for point, response in [(baseline_at_probe, response0), (prepared_at_probe, responsea)]:
            fd = (trajectory(vf, point + epsilon * probe, times) -
                  trajectory(vf, point - epsilon * probe, times)).ravel() / (2 * epsilon)
            finite_difference_errors.append(relative_error(fd, response @ probe))
    check("vortex_finite_time_tangent_vs_probe_finite_differences", max(finite_difference_errors), 2e-7)
    volume_errors = [abs(np.linalg.det(phi[-1]) - 1) for phi in [phi0, phia]]
    check("vortex_tangent_volume_preservation", max(volume_errors), 2e-8)

    # Outcome-independent finite-time scan. A shared basis is fitted only on
    # sites 0..4; sites 5..7 are held out. Projection uses the true operator,
    # so test error is an optimistic reconstruction bound, NOT prediction.
    finite_events, update_vectors = [], []
    cardinal_writes = [(0.07, 0.0), (0.0, 0.07), (-0.07, 0.0), (0.0, -0.07)]
    for site in range(8):
        for write in cardinal_writes:
            altered = q.copy()
            altered[2 * site:2 * site + 2] += write
            point = trajectory(vf, altered, np.array([delay]))[-1]
            _, tangent = tangent_trajectory(vf, vj, point, times)
            delta = tangent.reshape(-1, 16) - response0
            update_vectors.append(delta.ravel())
            finite_events.append({"site": site, "write": list(write), "split": "train" if site < 5 else "held_out",
                                  "update": spectrum(delta),
                                  "relative_update_norm": float(np.linalg.norm(delta) / np.linalg.norm(response0))})
    train_vectors = np.column_stack(update_vectors[:20])
    shared_basis, shared_singular, _ = np.linalg.svd(train_vectors, full_matrices=False)
    shared_energy = shared_singular**2
    shared_rank = int(np.searchsorted(np.cumsum(shared_energy), 0.95 * shared_energy.sum()) + 1)
    shared_basis = shared_basis[:, :shared_rank]
    for event, vector in zip(finite_events, update_vectors):
        approximation = shared_basis @ (shared_basis.T @ vector)
        event["oracle_shared_basis_relative_error"] = relative_error(approximation, vector)
    shared_basis_result = {
        "meaning": "POD basis of flattened finite response updates, not the rank of an individual response matrix",
        "training_sites": [0, 1, 2, 3, 4], "held_out_sites": [5, 6, 7],
        "training_examples": 20, "held_out_examples": 12,
        "basis_components_for_95pct_training_energy": shared_rank,
        "coefficient_source": "orthogonal projection of each TRUE update; optimistic bound, no predictor trained",
        "training_relative_error_median": float(np.median([e["oracle_shared_basis_relative_error"] for e in finite_events[:20]])),
        "held_out_relative_error_median": float(np.median([e["oracle_shared_basis_relative_error"] for e in finite_events[20:]])),
        "held_out_relative_error_min_max": [float(min(e["oracle_shared_basis_relative_error"] for e in finite_events[20:])),
                                             float(max(e["oracle_shared_basis_relative_error"] for e in finite_events[20:]))],
    }

    linear_matrix = np.kron(np.eye(8), -0.2 * np.eye(2) + 0.9 * ROT)
    linear_f = lambda x: linear_matrix @ x
    linear_j = lambda x: linear_matrix
    _, linear0 = tangent_trajectory(linear_f, linear_j, baseline_at_probe, times)
    _, lineara = tangent_trajectory(linear_f, linear_j, prepared_at_probe, times)
    linear_delta_norm = relative_error(lineara, linear0)
    check("linear_dynamics_history_changes_no_response_operator", linear_delta_norm, 1e-9)

    mu_values = [-1.0, -0.3, -0.1, -0.03, -0.01, 0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
    hopf_sweep = []
    for mu in mu_values:
        prepared_q, _ = hopf_flow_and_tangent(np.array([0.3, 0.0]), 5.0, mu)
        _, p_after = hopf_flow_and_tangent(prepared_q, 2.0, mu)
        _, p_before = hopf_flow_and_tangent(np.zeros(2), 2.0, mu)
        radius2 = float(prepared_q @ prepared_q)
        full_delta_j = np.zeros((34, 34))
        full_delta_j[:2, :2] = hopf_jacobian(prepared_q, mu) - hopf_jacobian(np.zeros(2), mu)
        full_delta_j[2:, 2:] = -radius2 * np.eye(32)
        coefficients, decoded = operator_dictionary(prepared_q)
        check(f"hopf_shared_dictionary_exact_mu{mu}", relative_error(decoded, full_delta_j), 1e-10)
        check(f"hopf_dictionary_coefficient_constraint_mu{mu}", abs(coefficients[0] ** 2 - coefficients[1] ** 2 - coefficients[2] ** 2), 1e-12)
        alpha = 4.0 if mu == 0 else np.expm1(4 * mu) / mu
        bulk_tangent_change = np.exp(-0.5 * 2.0) * ((1 + alpha * radius2) ** (-0.5) - 1)
        full_delta_p = np.zeros((34, 34))
        full_delta_p[:2, :2] = p_after - p_before
        full_delta_p[2:, 2:] = bulk_tangent_change * np.eye(32)
        hopf_sweep.append({"mu": mu, "prepared_radius": float(np.sqrt(radius2)),
                           "two_dimensional_finite_response": spectrum(p_after - p_before),
                           "augmented_instantaneous_update": spectrum(full_delta_j),
                           "augmented_finite_response_update": spectrum(full_delta_p),
                           "dictionary_coefficients": coefficients.tolist()})
        check(f"hopf_augmented_full_rank_mu{mu}", abs(spectrum(full_delta_j)["rank_relative_tolerance_1e-10"] - 34), 0)
        check(f"hopf_augmented_energy_rank95_mu{mu}", abs(spectrum(full_delta_j)["energy_rank_95"] - 32), 0)

    # Independent ODE integration checks the closed-form flow and tangent.
    max_hopf_flow_error = max_hopf_tangent_error = 0.0
    for mu in (-0.1, 0.0, 0.1):
        initial = np.array([0.2, -0.3])
        hf = lambda x: mu * x + 2 * (ROT @ x) - (x @ x) * x
        hj = lambda x: hopf_jacobian(x, mu)
        numeric_q, numeric_phi = tangent_trajectory(hf, hj, initial, np.array([3.0]))
        exact_q, exact_phi = hopf_flow_and_tangent(initial, 3.0, mu)
        max_hopf_flow_error = max(max_hopf_flow_error, relative_error(numeric_q[-1], exact_q))
        max_hopf_tangent_error = max(max_hopf_tangent_error, relative_error(numeric_phi[-1], exact_phi))
    check("hopf_closed_form_vs_independent_ode_flow", max_hopf_flow_error, 1e-9)
    check("hopf_closed_form_vs_independent_ode_tangent", max_hopf_tangent_error, 1e-9)

    # Independent full-dimensional check of the stable-block propagator.
    augmented_q, _ = hopf_flow_and_tangent(np.array([0.3, 0.0]), 5.0, 0.0)
    augmented_initial = np.concatenate([augmented_q, np.zeros(32)])

    def augmented_f(x):
        q, w = x[:2], x[2:]
        return np.concatenate([2 * ROT @ q - (q @ q) * q, -(0.5 + q @ q) * w])

    def augmented_j(x):
        q, w = x[:2], x[2:]
        jac = np.zeros((34, 34))
        jac[:2, :2] = hopf_jacobian(q, 0.0)
        jac[2:, :2] = -2 * np.outer(w, q)
        jac[2:, 2:] = -(0.5 + q @ q) * np.eye(32)
        return jac

    _, numeric_augmented = tangent_trajectory(augmented_f, augmented_j, augmented_initial, np.array([2.0]))
    expected_augmented = np.zeros((34, 34))
    expected_augmented[:2, :2] = hopf_flow_and_tangent(augmented_q, 2.0, 0.0)[1]
    expected_augmented[2:, 2:] = np.exp(-1) * (1 + 4 * (augmented_q @ augmented_q)) ** (-0.5) * np.eye(32)
    augmented_tangent_error = relative_error(numeric_augmented[-1], expected_augmented)
    check("hopf_augmented_finite_tangent_vs_independent_ode", augmented_tangent_error, 1e-9)

    # A nonlinear frequency shift links amplitude history to accumulated phase.
    shear = 0.7
    max_shear_flow_error = max_shear_tangent_error = 0.0
    for mu in (-0.1, 0.0, 0.1):
        initial, time = np.array([0.2, -0.3]), 3.0
        alpha = 2 * time if mu == 0 else np.expm1(2 * mu * time) / mu
        denominator = 1 + alpha * (initial @ initial)
        turn = rotation(2 * time - shear * np.log(denominator) / 2)
        gain = np.exp(mu * time) / np.sqrt(denominator)
        exact_q = gain * turn @ initial
        exact_phi = gain * turn @ (np.eye(2) - alpha / denominator * np.outer(initial + shear * ROT @ initial, initial))
        sf = lambda x: mu * x + 2 * ROT @ x - (x @ x) * (x + shear * ROT @ x)
        sj = lambda x: mu * np.eye(2) + 2 * ROT - (x @ x) * (np.eye(2) + shear * ROT) - 2 * np.outer(x + shear * ROT @ x, x)
        numeric_q, numeric_phi = tangent_trajectory(sf, sj, initial, np.array([time]))
        max_shear_flow_error = max(max_shear_flow_error, relative_error(numeric_q[-1], exact_q))
        max_shear_tangent_error = max(max_shear_tangent_error, relative_error(numeric_phi[-1], exact_phi))
    check("hopf_amplitude_phase_shear_flow_vs_independent_ode", max_shear_flow_error, 1e-9)
    check("hopf_amplitude_phase_shear_tangent_vs_independent_ode", max_shear_tangent_error, 1e-9)

    mu, horizon = 0.1, 2.0
    cycle_q = np.array([np.sqrt(mu), 0.0])
    _, phase_phi = hopf_flow_and_tangent(cycle_q, horizon, mu)
    floquet_phi = rotation(2 * horizon) @ np.diag([np.exp(-2 * mu * horizon), 1.0])
    check("hopf_cycle_radial_phase_formula", relative_error(phase_phi, floquet_phi), 1e-12)
    _, opposite_phi = hopf_flow_and_tangent(-cycle_q, horizon, mu)
    opposite_response_error = relative_error(opposite_phi, phase_phi)
    check("opposite_hopf_states_have_identical_tangents", opposite_response_error, 1e-14)
    _, quarter_phase_phi = hopf_flow_and_tangent(ROT @ cycle_q, horizon, mu)

    # Neural gain change: one altered tanh activity changes one Jacobian column.
    neural_weights = rng.normal(size=(16, 16)) / np.sqrt(16)
    v0 = rng.normal(scale=0.3, size=16)
    va = v0.copy()
    va[3] += 0.1
    neural_delta = neural_weights @ np.diag((1 - np.tanh(va) ** 2) - (1 - np.tanh(v0) ** 2))
    neural_rank = spectrum(neural_delta)["rank_relative_tolerance_1e-10"]
    check("one_neural_gain_change_is_rank_one", abs(neural_rank - 1), 0)

    fhn_epsilon, fhn_b = 0.08, 0.8
    fhn_v = np.sqrt(1 - fhn_epsilon * fhn_b)
    fhn_j = np.array([[1 - fhn_v**2, -1], [fhn_epsilon, -fhn_epsilon * fhn_b]])
    check("fhn_hopf_candidate_trace_zero", abs(np.trace(fhn_j)), 1e-12)
    check("fhn_hopf_candidate_det_formula", abs(np.linalg.det(fhn_j) - fhn_epsilon * (1 - fhn_epsilon * fhn_b**2)), 1e-12)

    report = {
        "status": "mathematical research probe; no trained model or cylinder-wake replication",
        "seed": SEED,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": checks,
        "vortex_jacobian_central_difference_errors": derivative_errors,
        "vortex_instantaneous_single_write": instantaneous,
        "vortex_finite_time_two_history": {
            "vortices": 8, "circulations": gamma.tolist(), "initial_positions": q.tolist(),
            "write": [0.07, 0.0], "write_site": 0, "preparation_delay": delay,
            "horizon": 1.5, "observation_times": times.tolist(), "probe_basis": "all 16 orthonormal position directions",
            "response_observation": "all vortex positions, stacked in time, excluding direct time-zero injection",
            "baseline": spectrum(response0), "update": spectrum(finite_delta),
            "update_to_baseline_norm": float(np.linalg.norm(finite_delta) / np.linalg.norm(response0)),
            "finite_difference_relative_errors": finite_difference_errors,
            "final_propagator_volume_errors": volume_errors,
        },
        "linear_control_relative_update_norm": linear_delta_norm,
        "vortex_finite_time_event_scan": finite_events,
        "vortex_shared_update_basis": shared_basis_result,
        "hopf_sweep": hopf_sweep,
        "hopf_closed_form_flow_error": max_hopf_flow_error,
        "hopf_closed_form_tangent_error": max_hopf_tangent_error,
        "hopf_augmented_tangent_error": augmented_tangent_error,
        "hopf_shear_flow_error": max_shear_flow_error,
        "hopf_shear_tangent_error": max_shear_tangent_error,
        "hopf_phase_response_difference_norm": float(np.linalg.norm(phase_phi - quarter_phase_phi)),
        "opposite_hopf_states_response_error": opposite_response_error,
        "opposite_hopf_states_baseline_output_distance": float(2 * np.sqrt(mu)),
        "neuron_one_gain_update_rank": neural_rank,
        "fhn_hopf_candidate": {"v": float(fhn_v), "jacobian": fhn_j.tolist(), "frequency": float(np.sqrt(np.linalg.det(fhn_j)))},
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "math_checks.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    axes[0].plot([v["state_dimension"] for v in instantaneous], [np.median(v["ranks"]) for v in instantaneous], "o-", label="Numerical rank")
    axes[0].plot([v["state_dimension"] for v in instantaneous], [np.median(v["energy_ranks_95"]) for v in instantaneous], "s-", label="95% energy rank")
    axes[0].set(xlabel="Point-vortex state dimension", ylabel="Update modes", title="Move one vortex")
    axes[0].legend(fontsize=8)
    axes[1].plot(mu_values, [v["augmented_instantaneous_update"]["energy_rank_95"] for v in hopf_sweep], "o-", label="95% energy rank")
    axes[1].axvline(0, color="0.5", linewidth=1, linestyle="--")
    axes[1].axhline(3, color="#18816c", linestyle=":", label="Shared matrix coefficients")
    axes[1].set(xlabel="Hopf parameter μ", ylabel="Modes / coefficients", title="34D counterexample", ylim=(0, 36))
    axes[1].legend(fontsize=8)
    qtime = np.linspace(0, 30, 200)
    for mu in (-0.3, -0.03, 0.0):
        radius = [np.linalg.norm(hopf_flow_and_tangent(np.array([0.3, 0.0]), t, mu)[0]) for t in qtime]
        axes[2].plot(qtime, radius, label=f"μ = {mu:g}")
    axes[2].set(xlabel="Time after a pulse", ylabel="State amplitude", title="Slow decay near onset")
    axes[2].legend(fontsize=8)
    fig.suptitle("VMN mathematical checks — ideal vortices and a Hopf normal form", fontsize=11)
    fig.savefig(ROOT / "results" / "math_summary.svg")
    plt.close(fig)

    failed = [item for item in checks if not item["passed"]]
    print(json.dumps({"checks_passed": len(checks) - len(failed), "checks_total": len(checks), "failed": failed,
                      "vortex_fd_errors": derivative_errors,
                      "instantaneous_vortex_rank_summary": [
                          {"N": item["vortices"], "rank_min_max": [min(item["ranks"]), max(item["ranks"])],
                           "energy_rank95_median": float(np.median(item["energy_ranks_95"]))} for item in instantaneous],
                      "vortex_finite_time": {"baseline_rank95": spectrum(response0)["energy_rank_95"],
                                              "update_rank95": spectrum(finite_delta)["energy_rank_95"],
                                              "update_relative_norm": report["vortex_finite_time_two_history"]["update_to_baseline_norm"]},
                      "vortex_finite_scan": {"events": len(finite_events),
                                             "update_rank95_median": float(np.median([e["update"]["energy_rank_95"] for e in finite_events])),
                                             "relative_update_norm_median": float(np.median([e["relative_update_norm"] for e in finite_events]))},
                      "vortex_shared_basis": shared_basis_result,
                      "hopf_at_zero": {key: value for key, value in next(row for row in hopf_sweep if row["mu"] == 0).items() if key not in ["augmented_instantaneous_update", "augmented_finite_response_update"]},
                      "closed_form_tangent_error": max_hopf_tangent_error}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
