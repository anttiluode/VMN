"""Reproduce the lifted-tangent checks and protocol-frozen reader follow-up."""
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
from scipy.optimize import linear_sum_assignment

from check_math import ROOT, SEED, relative_error, tangent_trajectory
from geometry_lift import (doubled_matrix, model_complex_tangent,
                           real_to_complex, unitary_lift, vortex_complex_tangent)
from geometry_memory import LAGS, LENGTHS, PENALTIES, WASHOUT
from geometry_model import make_model
from geometry_reader import reader_experiment


DEPENDENCIES = ("READER_PROTOCOL.md","research/geometry_reader.py",
                "research/geometry_model.py","research/geometry_memory.py",
                "research/check_math.py","results/geometry_checks.json")
SOURCES = DEPENDENCIES+("research/geometry_lift.py","research/check_reader.py",
                       "research/test_geometry_lift.py","research/test_geometry_reader.py")
VARIANTS = ("geometry0_shear0","geometry0_shear2","geometry1_shear0","geometry1_shear2","linear")
LABELS = ("Fixed\nβ=0","Fixed\nβ=2","Moving\nβ=0","Moving\nβ=2","Linear\ncontrol")


def hashes(files):
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}


def numerical_checks():
    checks,ideal_rows = [],[]
    maxima = {name:0. for name in ("analytic_tangent_vs_real","analytic_tangent_vs_real_block_decomposition",
                                  "unitary_similarity","matched_eigenvalues","matched_singular_values",
                                  "physical_conjugacy","unitary_normalization")}
    for seed in range(SEED,SEED+4):
        for geometry in (0.,1.):
            for beta in (0.,2.):
                model = make_model(seed,geometry,beta)
                transform = unitary_lift(8)
                for scale in (0.,.05,.3):
                    q = scale*np.random.default_rng(seed+73).normal(size=16)
                    j = model.jacobian(q)
                    a,b = model_complex_tangent(model,q)
                    doubled = doubled_matrix(a,b)
                    ar,br = real_to_complex(j)
                    expected = transform@j@transform.conj().T
                    maxima["analytic_tangent_vs_real"] = max(maxima["analytic_tangent_vs_real"],
                                                              relative_error(doubled,expected))
                    maxima["analytic_tangent_vs_real_block_decomposition"] = max(
                        maxima["analytic_tangent_vs_real_block_decomposition"],
                        relative_error(doubled,doubled_matrix(ar,br)))
                    maxima["unitary_similarity"] = max(maxima["unitary_similarity"],
                                                       relative_error(transform.conj().T@doubled@transform,j))
                    ej,ed = np.linalg.eigvals(j),np.linalg.eigvals(doubled)
                    i,k = linear_sum_assignment(np.abs(ej[:,None]-ed[None,:]))
                    maxima["matched_eigenvalues"] = max(maxima["matched_eigenvalues"],
                                                        float(np.max(np.abs(ej[i]-ed[k]))))
                    maxima["matched_singular_values"] = max(maxima["matched_singular_values"],
                        float(np.max(np.abs(np.linalg.svd(j,compute_uv=False)-np.linalg.svd(doubled,compute_uv=False)))))
                    physical = transform@np.linspace(-.3,.2,16)
                    response = doubled@physical
                    maxima["physical_conjugacy"] = max(maxima["physical_conjugacy"],
                                                       relative_error(response[8:],response[:8].conj()))
                    maxima["unitary_normalization"] = max(maxima["unitary_normalization"],
                                                           relative_error(transform@transform.conj().T,np.eye(16)))
    for name,value in maxima.items():
        bound = 1e-11 if name in ("matched_eigenvalues","matched_singular_values") else 1e-13
        checks.append({"name":name,"value":value,"upper_bound":bound,"passed":bool(value<=bound)})

    max_square = max_imaginary = max_negative = max_weak_growth = max_prediction = 0.
    for seed in range(SEED,SEED+4):
        model = make_model(seed)
        av,m = vortex_complex_tangent(model.anchors,model.gamma,0.)
        nu = np.linalg.eigvals(m@m.conj())
        critical = 1/np.sqrt(np.max(nu.real))
        max_imaginary = max(max_imaginary,float(np.max(np.abs(nu.imag))))
        max_negative = max(max_negative,float(max(0.,-np.min(nu.real))))
        for fraction in (.1,.3,1.2):
            kappa = fraction*critical
            h = doubled_matrix(1j*np.eye(8),kappa*m)
            squared = np.block([[kappa*kappa*m@m.conj()-np.eye(8),np.zeros((8,8))],
                                [np.zeros((8,8)),kappa*kappa*m.conj()@m-np.eye(8)]])
            max_square = max(max_square,relative_error(h@h,squared))
            growth = float(np.max(np.linalg.eigvals(h).real))
            predicted_growth = float(np.sqrt(max(fraction*fraction-1.,0.)))
            max_prediction = max(max_prediction,abs(growth-predicted_growth))
            if fraction < 1:
                max_weak_growth = max(max_weak_growth,abs(growth))
            ideal_rows.append({"seed":seed,"smoothing":0.,"uniform_frequency":1.,
                               "common_decay":0.,"kappa_threshold":float(critical),
                               "kappa":float(kappa),"fraction_of_threshold":fraction,
                               "max_growth_rate":growth,"predicted_max_growth_rate":predicted_growth})
    for name,value in [("ideal_squared_matrix_identity",max_square),
                       ("positive_circulation_nu_real",max_imaginary),
                       ("positive_circulation_nu_nonnegative",max_negative),
                       ("ideal_weak_coupling_growth_is_zero",max_weak_growth),
                       ("ideal_growth_threshold_prediction",max_prediction)]:
        checks.append({"name":name,"value":value,"upper_bound":1e-11,"passed":bool(value<=1e-11)})

    finite_time_error = 0.
    for geometry in (0.,1.):
        for beta in (0.,2.):
            model = make_model(SEED,geometry,beta)
            q = np.linspace(-.15,.1,16)
            times = np.array([0.,.75])
            _,real_phi = tangent_trajectory(model.vector_field,model.jacobian,q,times)
            transform = unitary_lift(8)
            def rhs(t,state):
                real_q = state[:16].real
                a,b = model_complex_tangent(model,real_q)
                return np.r_[model.vector_field(real_q),
                             (doubled_matrix(a,b)@state[16:].reshape(16,16)).ravel()]
            initial = np.r_[q.astype(complex),np.eye(16,dtype=complex).ravel()]
            solution = solve_ivp(rhs,(0.,.75),initial,t_eval=times,method="DOP853",rtol=2e-12,atol=2e-13)
            if not solution.success:
                raise RuntimeError(solution.message)
            expected = transform@real_phi[-1]@transform.conj().T
            finite_time_error = max(finite_time_error,relative_error(solution.y[16:,-1].reshape(16,16),expected))
    checks.append({"name":"independently_integrated_lifted_finite_response","value":finite_time_error,
                   "upper_bound":1e-10,"passed":bool(finite_time_error<=1e-10)})
    return checks,ideal_rows


def aggregate(reader):
    summary = []
    for variant in VARIANTS:
        rows = [r for r in reader["rows"] if r["variant"]==variant]
        item = {"variant":variant,"frozen_offsets":[r["offset_from_coupled_boundary"] for r in rows]}
        for mode in ("raw","quadratic"):
            item[mode] = {"mean_order_r2":{str(order):float(np.mean([
                r["readers"][mode]["test"]["mean_order_r2"][str(order)] for r in rows]))
                for order in (1,2,3)},
                "mean_nonlinear_r2":float(np.mean([r["readers"][mode]["test"]["mean_nonlinear_r2"] for r in rows])),
                "shuffled_mean_nonlinear_r2":float(np.mean([r["shuffled_label_control"][mode]["test"]["mean_nonlinear_r2"] for r in rows])),
                "shuffled_mean_order2_r2":float(np.mean([r["shuffled_label_control"][mode]["test"]["mean_order_r2"]["2"] for r in rows]))}
        item["quadratic_order2_r2_by_lag"] = np.mean([
            r["readers"]["quadratic"]["test_per_target"]["r2"][6:12] for r in rows],axis=0).tolist()
        summary.append(item)
    return summary


def figure(summary):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    fig,axes = plt.subplots(2,2,figsize=(12,8.4),constrained_layout=True)
    positions = np.arange(5)
    for ax,key,title in [(axes[0,0],"order2","A. Even delayed targets: primary gate failed"),
                         (axes[0,1],"nonlinear","B. Combined nonlinear memory remains below zero")]:
        for mode,shift,color in [("raw",-.19,"#557c9e"),("quadratic",.19,"#d28b42")]:
            values = [r[mode]["mean_order_r2"]["2"] if key=="order2" else r[mode]["mean_nonlinear_r2"] for r in summary]
            ax.bar(positions+shift,values,width=.36,color=color,label=mode.capitalize())
        if key=="order2":
            ax.plot(positions,[r["quadratic"]["shuffled_mean_order2_r2"] for r in summary],"kx",label="Shuffled labels, quadratic")
        ax.axhline(0,color="#444",linewidth=.7)
        ax.set_xticks(positions,LABELS)
        ax.set_ylabel("Mean held-out signed R²")
        ax.set_title(title,fontsize=11)
        ax.legend(fontsize=8)
    ax = axes[1,0]
    values = np.array([r["quadratic_order2_r2_by_lag"] for r in summary])
    scale = float(np.max(np.abs(values)))
    image = ax.imshow(values,cmap="RdBu",vmin=-scale,vmax=scale,aspect="auto")
    ax.set_xticks(np.arange(6),LAGS)
    ax.set_yticks(np.arange(5),[label.replace("\n"," ") for label in LABELS])
    for i in range(5):
        for k in range(6):
            ax.text(k,i,f"{values[i,k]:.3f}",ha="center",va="center",fontsize=8,
                    color="white" if abs(values[i,k])>.65*scale else "black")
    fig.colorbar(image,ax=ax,label="Order-2 R²",shrink=.8)
    ax.set_xlabel("Lag in input samples")
    ax.set_title("C. Quadratic reader: weak signal at short lags",fontsize=11)
    ax = axes[1,1]
    ratio = np.linspace(0,1.4,300)
    ax.plot(ratio,np.sqrt(np.maximum(1-ratio*ratio,0)),label="Leading mode frequency / ω")
    ax.plot(ratio,np.sqrt(np.maximum(ratio*ratio-1,0)),label="Maximum growth / ω")
    ax.axvline(1,color="#666",linestyle="--",linewidth=.8)
    ax.set_xlabel("κ / κ* (ideal uniform-frequency model)")
    ax.set_ylabel("Normalized frequency or growth")
    ax.set_title("D. Positive circulation: zero growth below threshold",fontsize=11)
    ax.legend(fontsize=8)
    fig.suptitle("VMN: exact lifted response; quadratic reader misses the memory gate",fontsize=14)
    fig.savefig(ROOT/"results"/"reader_summary.svg",metadata={"Date":None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-memory",type=Path,help="Reuse a local checkpoint only if every experiment dependency hash matches")
    args = parser.parse_args()
    checks,ideal = numerical_checks()
    if args.reuse_memory:
        checkpoint = json.loads(args.reuse_memory.read_text())
        if checkpoint["source_sha256"] != hashes(DEPENDENCIES):
            raise ValueError("reader checkpoint hashes differ; regenerate it")
        reader = checkpoint["reader"]
    else:
        reader = reader_experiment(json.loads((ROOT/"results/geometry_checks.json").read_text()))
    summary = aggregate(reader)
    protocol = {"baseline_commit":"72f94291f266cf793a7b780ef09ef38f4759bc23",
                "layout_seeds":list(range(SEED,SEED+4)),
                "input_seeds":[s+60000 for s in range(SEED,SEED+4)],
                "shuffle_seeds":[s+70000 for s in range(SEED,SEED+4)],
                "split_lengths":dict(zip(("train","validation","test"),LENGTHS)),
                "washout":WASHOUT,"integration_dt":.25,"rk4_substeps":2,
                "independent_rest_initialization_per_split":True,
                "lags":list(LAGS),"legendre_orders":[1,2,3],"ridge_penalties":list(PENALTIES),
                "ridge_selection":"validation normalized MSE over all 18 targets; training-only feature statistics",
                "operating_point_selection":"frozen from baseline validation; no new offset selection",
                "primary_gate":{"variant":"geometry0_shear0","metric":"mean_order2_test_r2",
                                "minimum_mean_gain":.02,"minimum_wins":3,"out_of":4,
                                "require_positive_quadratic_mean_score":True},
                "secondary_model_gates":{"metric":"mean_nonlinear_test_r2","minimum_mean_gain":.02,
                                         "minimum_wins":3,"out_of":4},
                "jacobian_check_count":48,"jacobian_state_rng_offset":73,
                "jacobian_state_scales":[0.,.05,.3]}
    receipt = {"schema_version":1,"date":"2026-10-07","source_sha256":hashes(SOURCES),
               "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
                           "matplotlib":matplotlib.__version__},
               "protocol":protocol,"numerical_checks":checks,"ideal_limit":ideal,
               "reader":reader,"aggregate":summary,
               "timing":{"integration_seconds":sum(r["integration_seconds"] for r in reader["rows"]),
                         "fit_and_score_seconds":sum(v["fit_and_score_seconds"] for r in reader["rows"]
                             for key in ("readers","shuffled_label_control") for v in r[key].values())}}
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"results/reader_checks.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    figure(summary)
    failed = [r for r in checks if not r["passed"]]
    print(json.dumps({"numerical_checks_passed":len(checks)-len(failed),"numerical_checks_total":len(checks),
                      "failed":failed,"primary_reader_gate":reader["primary_reader_gate"],
                      "secondary_gates_passed":sum(r["passed_predeclared_gate"] for r in reader["secondary_model_gates"]),
                      "secondary_gates_total":len(reader["secondary_model_gates"]),
                      "summary":[{"variant":r["variant"],
                                  "raw_order2_r2":r["raw"]["mean_order_r2"]["2"],
                                  "quadratic_order2_r2":r["quadratic"]["mean_order_r2"]["2"],
                                  "quadratic_nonlinear_r2":r["quadratic"]["mean_nonlinear_r2"]} for r in summary]},indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
