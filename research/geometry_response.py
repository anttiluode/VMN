"""Sparse state codes and two-history tangent response reconstruction."""
import numpy as np

from check_math import (relative_error, spectrum, tangent_trajectory, trajectory)
from geometry_model import make_model, rollout


def encode_state_difference(baseline,actual,blocks):
    baseline,actual = np.asarray(baseline,dtype=float),np.asarray(actual,dtype=float)
    if (baseline.ndim != 1 or baseline.shape != actual.shape or not baseline.size or baseline.size%2
        or not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(actual))
        or not isinstance(blocks,(int,np.integer)) or not 1<=blocks<=baseline.size//2):
        raise ValueError("code requires matching finite planar states and a valid integer block count")
    difference = (actual-baseline).reshape(-1,2)
    indices = np.argsort(-np.sum(difference**2,axis=1),kind="stable")[:blocks]
    return {"indices":indices.tolist(),"offsets":difference[indices].tolist(),
            "per_event_numbers_including_indices":3*int(blocks)}


def decode_state_difference(baseline,code):
    baseline = np.asarray(baseline,dtype=float)
    indices = np.asarray(code["indices"])
    offsets = np.asarray(code["offsets"],dtype=float)
    if (baseline.ndim != 1 or baseline.size%2 or indices.ndim != 1
        or not np.issubdtype(indices.dtype,np.integer) or offsets.shape != (len(indices),2)
        or len(set(indices.tolist())) != len(indices) or np.any(indices<0)
        or np.any(indices>=baseline.size//2) or not np.all(np.isfinite(baseline))
        or not np.all(np.isfinite(offsets))):
        raise ValueError("invalid sparse planar state code")
    result = baseline.copy().reshape(-1,2)
    result[indices] += offsets
    return result.ravel()


def response_records(model,before_write,events,delay,times,stored_blocks=(1,2,4)):
    baseline_at_probe = trajectory(model.vector_field,before_write,np.array([delay]))[-1]
    baseline_states,baseline_response = tangent_trajectory(
        model.vector_field,model.jacobian,baseline_at_probe,times)
    records = []
    for event in events:
        altered = before_write.copy()
        site = event["site"]
        altered[2*site:2*site+2] += event["write"]
        at_probe = trajectory(model.vector_field,altered,np.array([delay]))[-1]
        states,matrices = tangent_trajectory(model.vector_field,model.jacobian,at_probe,times)
        decoders = []
        for blocks in stored_blocks:
            code = encode_state_difference(baseline_at_probe,at_probe,blocks)
            decoded = decode_state_difference(baseline_at_probe,code)
            ys,phis = tangent_trajectory(model.vector_field,model.jacobian,decoded,times)
            decoders.append({"blocks":blocks,"code":code,
                             "response_update":phis-baseline_response,
                             "state_change":ys-baseline_states})
        records.append({"site":int(site),"write":list(event["write"]),
                        "split":"train" if site<5 else "held_out",
                        "response_update":matrices-baseline_response,
                        "state_change":states-baseline_states,"decoders":decoders,
                        "state_difference_at_probe":(at_probe-baseline_at_probe).tolist()})
    return {"baseline_at_probe":baseline_at_probe,"baseline_response":baseline_response,
            "events":records}


def concentration(matrix):
    spec = spectrum(matrix)
    sv = np.array(spec.pop("singular_values"))
    energy = sv**2
    spec["participation_rank"] = float(energy.sum()**2/np.sum(energy**2)) if energy.sum() else 0.
    return spec


def response_experiment(seed=20261007,progress=print):
    horizons = (.5,2.,6.)
    observations = {h:np.round(np.linspace(h/16,h,16),12) for h in horizons}
    times = np.unique(np.concatenate(list(observations.values())))
    selectors = {h:np.searchsorted(times,t) for h,t in observations.items()}
    writes = ((.07,0.),(0.,.07),(-.07,0.),(0.,-.07))
    events = [{"site":site,"write":write} for site in range(8) for write in writes]
    history = np.random.default_rng(seed+20000).uniform(-1,1,256)
    probes = np.random.default_rng(seed+30000).normal(size=(3,16))
    probes /= np.linalg.norm(probes,axis=1)[:,None]
    results = []
    for geometry in (0.,1.):
        for beta in (0.,2.):
            model = make_model(seed,geometry,beta,offset=-.03)
            before_write = rollout(model,history)[-1]
            raw = response_records(model,before_write,events,1.,times)
            for horizon in horizons:
                selected_times = selectors[horizon]
                baseline = raw["baseline_response"][selected_times].reshape(-1,16)
                updates = [event["response_update"][selected_times].reshape(-1,16) for event in raw["events"]]
                training = np.column_stack([u.ravel() for u in updates[:20]])
                basis,singular,_ = np.linalg.svd(training,full_matrices=False)
                rank = int(np.searchsorted(np.cumsum(singular**2),.95*np.sum(singular**2))+1)
                basis = basis[:,:rank]
                rows = []
                for event,update in zip(raw["events"],updates):
                    oracle = (basis @ (basis.T @ update.ravel())).reshape(update.shape)
                    decoders = []
                    actual_change = event["state_change"][selected_times]
                    for decoded in event["decoders"]:
                        predicted = decoded["response_update"][selected_times].reshape(-1,16)
                        probe_errors = [relative_error(predicted @ p,update @ p) for p in probes]
                        decoders.append({"blocks":decoded["blocks"],"code":decoded["code"],
                                         "response_update_relative_error":relative_error(predicted,update),
                                         "held_out_probe_relative_errors":probe_errors,
                                         "future_state_change_relative_error":relative_error(
                                             decoded["state_change"][selected_times],actual_change)})
                    rows.append({"site":event["site"],"write":event["write"],"split":event["split"],
                                 "update":concentration(update),
                                 "relative_update_norm":float(np.linalg.norm(update)/np.linalg.norm(baseline)),
                                 "oracle_fixed_dictionary_relative_error":relative_error(oracle,update),
                                 "state_decoders":decoders})
                test_rows = rows[20:]
                decoder_summary = []
                for blocks in (1,2,4):
                    entries = [next(d for d in row["state_decoders"] if d["blocks"]==blocks) for row in test_rows]
                    decoder_summary.append({"blocks":blocks,"per_event_numbers_including_indices":3*blocks,
                        "held_out_update_error_median":float(np.median([d["response_update_relative_error"] for d in entries])),
                        "held_out_probe_error_median":float(np.median([e for d in entries for e in d["held_out_probe_relative_errors"]])),
                        "held_out_future_state_change_error_median":float(np.median([d["future_state_change_relative_error"] for d in entries]))})
                results.append({"geometry":geometry,"beta":beta,"horizon":horizon,
                    "baseline":concentration(baseline),
                    "median_update_participation_rank":float(np.median([row["update"]["participation_rank"] for row in rows])),
                    "median_update_energy_rank95":float(np.median([row["update"]["energy_rank_95"] for row in rows])),
                    "median_relative_update_norm":float(np.median([row["relative_update_norm"] for row in rows])),
                    "fixed_dictionary":{"components":rank,"shared_matrix_numbers":int(basis.size),
                        "per_event_coefficients":rank,"training_sites":list(range(5)),"held_out_sites":[5,6,7],
                        "training_relative_error_median":float(np.median([row["oracle_fixed_dictionary_relative_error"] for row in rows[:20]])),
                        "held_out_relative_error_median":float(np.median([row["oracle_fixed_dictionary_relative_error"] for row in test_rows])),
                        "coefficient_source":"true update projection; optimistic representation bound"},
                    "state_decoder_summary":decoder_summary,"events":rows})
            progress(f"response geometry {int(geometry)}, shear {int(beta)} complete",flush=True)
    return {"seed":seed,"offset_from_coupled_boundary":-.03,"preparation_samples":256,
            "delay":1.,"observations_per_horizon":16,"horizons":list(horizons),
            "rows":results,"probe_count":3,"probe_selection":"independent fixed random unit vectors",
            "decoder_cost":{"full_state_coordinates":16,"shared_baseline_coordinates":16,
                            "shared_model_parameters":62,"optional_actual_model_cache_numbers":272,
                            "sparse_code_numbers":"3 per retained planar block, including its index",
                            "regeneration":"full-model ODE and variational integration, not a speed saving"}}
