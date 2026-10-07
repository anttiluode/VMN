"""Reproduce the fixed geometry/shear protocol, checks, receipts and figure.

Run from the VMN root: python research/check_geometry.py
--reuse-memory/--reuse-response accept local checkpoints only if dependency
hashes match. Research gates can fail while numerical consistency checks pass.
"""
import argparse
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

from check_math import ROOT, SEED, derivative_fd, relative_error, tangent_trajectory, trajectory
from geometry_model import make_model, phase_kick, relaxed_response_norm, rollout
from geometry_memory import (LAGS, LENGTHS, OFFSETS, PENALTIES, WASHOUT,
                             fit_readout, memory_experiment, memory_targets,
                             score_predictions, select_readout)
from geometry_response import (decode_state_difference, encode_state_difference,
                               response_experiment, response_records)


MODEL_FILES = ("GEOMETRY_PROTOCOL.md","research/check_math.py","research/geometry_model.py")
SOURCE_FILES = MODEL_FILES+("research/geometry_memory.py","research/geometry_response.py",
                           "research/check_geometry.py","research/test_geometry.py",
                           "research/test_geometry_readout.py","research/test_geometry_response.py")


def source_hashes(files):
    return {path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in files}


def cached_or_run(checkpoint,key,dependency,run):
    if checkpoint is None:
        return run()
    payload = json.loads(Path(checkpoint).read_text())
    if payload["source_sha256"] != source_hashes(MODEL_FILES+(dependency,)):
        raise ValueError(f"{key} checkpoint source hashes do not match; regenerate it")
    return payload[key]


def numerical_checks():
    checks,phase_rows = [],[]
    def check(name,value,bound):
        checks.append({"name":name,"value":float(value),"upper_bound":float(bound),
                       "passed":bool(np.isfinite(value) and value<=bound)})
    q = np.linspace(-.21,.17,16)
    zero = np.zeros(16)
    for beta in (0.,2.):
        fixed = make_model(SEED,0.,beta)
        moving = make_model(SEED,1.,beta)
        check(f"matched_rest_response_shear{int(beta)}",
              np.linalg.norm(fixed.jacobian(zero)-moving.jacobian(zero)),1e-13)
        for geometry,model in [(0.,fixed),(1.,moving)]:
            fd = derivative_fd(model.vector_field,q,1e-5)
            check(f"hybrid_jacobian_geometry{int(geometry)}_shear{int(beta)}",
                  relative_error(model.jacobian(q),fd),2e-8)
            abscissa = np.max(np.linalg.eigvals(model.jacobian(zero)).real)
            check(f"coupled_stability_offset_geometry{int(geometry)}_shear{int(beta)}",
                  abs(abscissa+.03),2e-12)

    model = make_model(SEED,1.,2.)
    times = np.array([.2,.5])
    _,phi = tangent_trajectory(model.vector_field,model.jacobian,q,times)
    probe = np.random.default_rng(SEED+40000).normal(size=16)
    probe /= np.linalg.norm(probe)
    eps = 1e-5
    fd = (trajectory(model.vector_field,q+eps*probe,times)
          -trajectory(model.vector_field,q-eps*probe,times))/2/eps
    check("hybrid_tangent_vs_physical_finite_pulse",
          relative_error(fd,np.einsum("tij,j->ti",phi,probe)),2e-7)
    linear = make_model(SEED,0.,0.,cubic=0.)
    raw = response_records(linear,q,[{"site":3,"write":[.07,0.]}],1.,times,(1,))
    check("fixed_linear_two_history_negative_control",
          np.linalg.norm(raw["events"][0]["response_update"])/np.linalg.norm(raw["baseline_response"]),2e-8)

    # Compare the actual sample-and-hold RK4 integrator with independently
    # adaptive integration, including the largest positive operating offset.
    inputs = np.random.default_rng(SEED+50000).uniform(-1,1,32)
    for geometry in (0.,1.):
        model = make_model(SEED,geometry,2.,offset=.10)
        current = q.copy()
        reference = []
        for u in inputs:
            sol = solve_ivp(lambda t,x:model.vector_field(x,u),(0,.25),current,
                            method="DOP853",rtol=2e-12,atol=2e-13)
            if not sol.success:
                raise RuntimeError(sol.message)
            current = sol.y[:,-1]
            reference.append(current.copy())
        reference = np.array(reference)
        coarse = rollout(model,inputs,initial=q)
        fine = rollout(model,inputs,substeps=4,initial=q)
        check(f"sample_and_hold_RK4_vs_adaptive_geometry{int(geometry)}",
              relative_error(coarse,reference),2e-4)
        check(f"refined_RK4_vs_adaptive_geometry{int(geometry)}",
              relative_error(fine,reference),2e-5)

    # Exact asymptotic phase and SPECTRAL norm of the relaxed Jacobian change.
    beta,rho = 2.,1e-5
    for mu in (.4,.1,.025,.00625):
        rstar = np.sqrt(mu)
        phase = phase_kick(mu,beta,rho)
        def polar(t,y):
            r = y[0]
            return [mu*r-r**3,-beta*(r*r-mu)]
        sol = solve_ivp(polar,(0,12/mu),[rstar+rho,0.],method="DOP853",rtol=2e-12,atol=2e-13)
        if not sol.success:
            raise RuntimeError(sol.message)
        check(f"exact_phase_vs_independent_ODE_mu{mu}",abs(sol.y[1,-1]-phase),2e-10)
        rotation = np.array([[0.,-1.],[1.,0.]])
        def jac(x):
            return mu*np.eye(2)+rotation-(np.eye(2)+beta*rotation)@((x@x)*np.eye(2)+2*np.outer(x,x))
        q0 = np.array([rstar,0.])
        q1 = rstar*np.array([np.cos(phase),np.sin(phase)])
        measured = np.linalg.norm(jac(q1)-jac(q0),2)
        predicted = relaxed_response_norm(mu,beta,phase)
        check(f"relaxed_phase_to_response_spectral_norm_mu{mu}",abs(measured-predicted),1e-12)
        phase_rows.append({"mu":mu,"kick":rho,"phase_shift":phase,
                           "relaxed_jacobian_spectral_norm":predicted,
                           "phase_for_ten_percent_relative_kick":phase_kick(mu,beta,.1*rstar)})
    return checks,phase_rows


def figure(receipt):
    plt.rcParams.update({"svg.hashsalt":"vmn-geometry-20261007","font.size":10})
    fig,axes = plt.subplots(2,2,figsize=(12,8.2),constrained_layout=True)
    labels = ["linear","geometry0_shear0","geometry0_shear2","geometry1_shear0","geometry1_shear2"]
    display = ["Linear","Fixed\nβ=0","Fixed\nβ=2","Moving\nβ=0","Moving\nβ=2"]
    selected = receipt["memory"]["validation_selected"]
    values = [[r["test"]["mean_nonlinear_r2"] for r in selected if r["variant"]==label] for label in labels]
    ax = axes[0,0]
    ax.bar(range(5),[np.mean(v) for v in values],color=["#aaa","#819cb2","#386f9a","#d9a26e","#ac652f"])
    for i,v in enumerate(values):
        ax.scatter(np.linspace(i-.12,i+.12,len(v)),v,c="#222",s=15,zorder=3)
    ax.axhline(0,color="#555",lw=1)
    ax.set_xticks(range(5),display)
    ax.set_ylabel("Held-out nonlinear R²")
    ax.set_title("A. Memory advantage gates failed")

    row = next(r for r in receipt["response"]["rows"] if r["geometry"]==1 and r["beta"]==2 and r["horizon"]==2)
    errors = [row["fixed_dictionary"]["held_out_relative_error_median"]]+[d["held_out_update_error_median"] for d in row["state_decoder_summary"]]
    ax = axes[0,1]
    bars = ax.bar(range(4),errors,color=["#aaa","#9fc6ae","#64a080","#307553"])
    ax.set_xticks(range(4),[f"POD\n{row['fixed_dictionary']['components']} coeffs","State\n3 numbers","State\n6 numbers","State\n12 numbers"])
    ax.set_ylim(0,1.1)
    for bar,error in zip(bars,errors):
        ax.text(bar.get_x()+bar.get_width()/2,error+.025,f"{100*error:.1f}%",ha="center",fontsize=9)
    ax.set_ylabel("Median relative update error")
    ax.set_title("B. Unseen sites: moving geometry, β=2, T=2")

    phase = receipt["phase_memory"]
    mu = [r["mu"] for r in phase]
    ax = axes[1,0]
    ax.loglog(mu,[abs(r["phase_shift"])/abs(phase[0]["phase_shift"]) for r in phase],"o-",label="Retained phase")
    ax.loglog(mu,[r["relaxed_jacobian_spectral_norm"]/phase[0]["relaxed_jacobian_spectral_norm"] for r in phase],"s-",label="Response change")
    ax.set_xlabel("μ above isolated Hopf onset")
    ax.set_ylabel("Ratio to μ=0.4 value")
    ax.set_title("C. More phase sensitivity, less Jacobian change")
    ax.legend()

    ax = axes[1,1]
    for geometry,color in [(0,"#386f9a"),(1,"#ac652f")]:
        rows = [r for r in receipt["response"]["rows"] if r["geometry"]==geometry and r["beta"]==2]
        x = [r["horizon"] for r in rows]
        ax.plot(x,[r["baseline"]["participation_rank"] for r in rows],"o-",color=color,label=f"{'Fixed' if geometry==0 else 'Moving'} baseline")
        ax.plot(x,[r["median_update_participation_rank"] for r in rows],"s--",color=color,label=f"{'Fixed' if geometry==0 else 'Moving'} update")
    ax.set_xlabel("Future probe horizon")
    ax.set_ylabel("Participation rank")
    ax.set_title("D. Compare baseline and update concentration")
    ax.legend(fontsize=8)
    fig.suptitle("VMN: useful response reconstruction; no demonstrated memory advantage",fontsize=14)
    fig.savefig(ROOT/"results"/"geometry_summary.svg",metadata={"Date":None})
    plt.close(fig)


def protocol_metadata():
    models = []
    for seed in range(SEED,SEED+4):
        model = make_model(seed)
        models.append({"layout_seed":seed,"anchors":model.anchors.tolist(),
                       "gamma":model.gamma.tolist(),"omega":model.omega.tolist(),
                       "decay":model.decay.tolist(),"input_vector":model.input_vector.tolist(),
                       "coupling":model.coupling,"smoothing":model.smoothing,
                       "coupled_boundary":model.stability_boundary()})
    return {"models":models,"geometry_values":[0.,1.],"shear_values":[0.,2.],
            "geometry_and_shear_gate":{"minimum_mean_test_nonlinear_r2_improvement":.02,
                                       "minimum_paired_seed_wins":3,"seed_count":4},
            "memory":{"layout_seeds":list(range(SEED,SEED+4)),
                      "input_seeds":[s+10000 for s in range(SEED,SEED+4)],
                      "integration_dt":.25,"rk4_substeps":2,"washout":WASHOUT,
                      "split_lengths":dict(zip(("train","validation","test"),LENGTHS)),
                      "independent_trajectory_and_rest_initialization_per_split":True,
                      "lags":list(LAGS),"legendre_orders":[1,2,3],
                      "offsets_from_coupled_boundary":list(OFFSETS),
                      "stable_offsets_for_linear_control":list(OFFSETS[:3]),
                      "ridge_penalties":list(PENALTIES),
                      "state_standardization":"training states only",
                      "ridge_selection":"validation MSE normalized by training target variance",
                      "offset_selection":"validation mean signed R2 for Legendre orders 2 and 3"},
            "response":{"layout_seed":SEED,"preparation_input_seed":SEED+20000,
                        "probe_seed":SEED+30000,"training_sites":list(range(5)),
                        "held_out_sites":[5,6,7],"write_magnitude":.07,
                        "write_directions":[[1,0],[0,1],[-1,0],[0,-1]],
                        "delay":1.,"horizons":[.5,2.,6.],"observations_per_horizon":16,
                        "state_blocks_retained":[1,2,4],"POD_training_energy_fraction":.95,
                        "ODE_method":"DOP853","ODE_rtol":2e-12,"ODE_atol":2e-13,
                        "regeneration_state_and_variational_coordinates":16+16*16,
                        "regeneration_work":"one full-model variational ODE solve per decoded state; dense J Phi multiplication"},
            "numerical_checks":{"tangent_probe_seed":SEED+40000,
                                "integration_refinement_input_seed":SEED+50000,
                                "finite_difference_step":1e-5,"isolated_phase_kick":1e-5}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-memory",type=Path)
    parser.add_argument("--reuse-response",type=Path)
    args = parser.parse_args()
    checks,phase = numerical_checks()
    memory = cached_or_run(args.reuse_memory,"memory","research/geometry_memory.py",
                           lambda:memory_experiment(list(range(SEED,SEED+4))))
    response = cached_or_run(args.reuse_response,"response","research/geometry_response.py",response_experiment)
    receipt = {"schema_version":1,"date":"2026-10-07","source_sha256":source_hashes(SOURCE_FILES),
               "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
               "protocol":protocol_metadata(),"numerical_checks":checks,"phase_memory":phase,
               "memory":memory,"response":response}
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"results"/"geometry_checks.json").write_text(json.dumps(receipt,indent=2)+"\n")
    figure(receipt)
    failed = [check for check in checks if not check["passed"]]
    print(json.dumps({"numerical_checks_passed":len(checks)-len(failed),"numerical_checks_total":len(checks),
                      "failed":failed,"geometry_gate":memory["geometry_gate"],
                      "shear_gates":memory["shear_gates"],"phase_growth_factor":abs(phase[-1]["phase_shift"]/phase[0]["phase_shift"]),
                      "response_shrink_factor":phase[0]["relaxed_jacobian_spectral_norm"]/phase[-1]["relaxed_jacobian_spectral_norm"]},indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
