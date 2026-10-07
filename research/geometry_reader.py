"""Matched readers of identical reservoir states; no new recurrent state."""
import hashlib
import time

import numpy as np

from geometry_memory import (matrix, select_readout, score_predictions,
                             memory_targets, LAGS, LENGTHS, WASHOUT)
from geometry_model import make_model, rollout


def reader_features(states,mode):
    if np.iscomplexobj(states):
        raise ValueError("states must be real interleaved planar coordinates")
    x = matrix(states,"states")
    if x.shape[1] < 2 or x.shape[1] % 2:
        raise ValueError("states require an even positive coordinate count")
    if mode == "raw":
        return x.copy()
    if mode != "quadratic":
        raise ValueError("reader must be raw or quadratic")
    re,im = x[:,0::2],x[:,1::2]
    with np.errstate(over="ignore",invalid="ignore"):
        result = np.column_stack([x,re*re+im*im,re*re-im*im,2*re*im])
    if not np.all(np.isfinite(result)):
        raise ValueError("quadratic coordinates overflowed")
    return result


def compare_readers(datasets,descriptions,shuffle_seed=None):
    if len(datasets) != 3:
        raise ValueError("provide training, validation and test datasets")
    data = [(matrix(x,"states"),matrix(y,"targets")) for x,y in datasets]
    if (any(len(x) != len(y) for x,y in data)
        or any(y.shape[1] != len(descriptions) for _,y in data)
        or {d["order"] for d in descriptions} != {1,2,3}):
        raise ValueError("rows and target descriptions must match all three polynomial orders")
    training_targets,validation_targets = data[0][1],data[1][1]
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        training_targets = training_targets[rng.permutation(len(training_targets))]
        validation_targets = validation_targets[rng.permutation(len(validation_targets))]
    results = {}
    for mode in ("raw","quadratic"):
        start = time.perf_counter()
        features = [reader_features(x,mode) for x,_ in data]
        fitted,penalty = select_readout(features[0],training_targets,
                                       features[1],validation_targets)
        # Validation is scored against its actual fitting labels in the control.
        scores = {"validation":score_predictions(validation_targets,fitted.predict(features[1])),
                  "test":score_predictions(data[2][1],fitted.predict(features[2]))}
        summaries = {}
        for split,score in scores.items():
            values = np.asarray(score["r2"])
            correlations = np.asarray(score["correlation_squared"])
            nonlinear = np.array([d["order"]>1 for d in descriptions])
            long_lag = np.array([d["lag"]>=16 for d in descriptions])
            summary = {"mean_nonlinear_r2":float(np.mean(values[nonlinear])),
                       "mean_linear_r2":float(np.mean(values[~nonlinear])),
                       "mean_long_lag_r2":float(np.mean(values[long_lag])) if long_lag.any() else None,
                       "memory_capacity_sum_correlation_squared":float(correlations.sum()),
                       "capacity_by_order":{str(order):float(correlations[
                           [d["order"]==order for d in descriptions]].sum()) for order in (1,2,3)}}
            summary["mean_order_r2"] = {str(order):float(np.mean(values[
                [d["order"]==order for d in descriptions]])) for order in (1,2,3)}
            summaries[split] = summary
        digest = hashlib.sha256()
        for parameter in (fitted.mean,fitted.scale,fitted.weights):
            digest.update(parameter.astype("<f8").tobytes())
        results[mode] = {"ridge_penalty":penalty,"feature_count":features[0].shape[1],
                         "readout_coefficients":int(fitted.weights.size),
                         "parameters_sha256":digest.hexdigest(),
                         "training_feature_mean":fitted.mean.tolist(),
                         "training_feature_scale":fitted.scale.tolist(),
                         **summaries,"test_per_target":scores["test"],
                         "fit_and_score_seconds":time.perf_counter()-start}
    return results


def fresh_datasets(model,input_seed,lengths=LENGTHS,washout=WASHOUT):
    if (len(lengths) != 3 or any(not isinstance(n,(int,np.integer)) or n < 1 for n in lengths)
        or not isinstance(washout,(int,np.integer)) or washout < max(LAGS)):
        raise ValueError("provide three positive split sizes and washout covering the target lags")
    rng = np.random.default_rng(input_seed)
    datasets,hashes = [],[]
    for length in lengths:
        inputs = rng.uniform(-1,1,washout+length)
        states = rollout(model,inputs)
        indices,targets,descriptions = memory_targets(inputs)
        keep = indices >= washout
        datasets.append((states[indices[keep]],targets[keep]))
        hashes.append(hashlib.sha256(inputs.astype("<f8").tobytes()).hexdigest())
    return datasets,descriptions,hashes


def reader_gate(rows,variant):
    selected = [row for row in rows if row["variant"]==variant]
    differences = [r["readers"]["quadratic"]["test"]["mean_order_r2"]["2"]
                   -r["readers"]["raw"]["test"]["mean_order_r2"]["2"] for r in selected]
    if len(selected) != 4 or len({r["seed"] for r in selected}) != 4:
        raise ValueError("reader gates require four unique paired layout seeds")
    average = float(np.mean(differences))
    wins = int(np.sum(np.asarray(differences)>0))
    score = float(np.mean([r["readers"]["quadratic"]["test"]["mean_order_r2"]["2"]
                          for r in selected]))
    return {"variant":variant,"paired_seeds":[r["seed"] for r in selected],
            "paired_seed_differences":differences,"mean_difference":average,
            "wins":wins,"out_of":4,"mean_quadratic_order2_r2":score,
            "passed_predeclared_gate":bool(average>=.02 and wins>=3 and score>0)}


def model_comparison(rows,a,b):
    differences = []
    seeds = sorted({r["seed"] for r in rows})
    for seed in seeds:
        ra = next(r for r in rows if r["seed"]==seed and r["variant"]==a)
        rb = next(r for r in rows if r["seed"]==seed and r["variant"]==b)
        differences.append(ra["readers"]["quadratic"]["test"]["mean_nonlinear_r2"]
                           -rb["readers"]["quadratic"]["test"]["mean_nonlinear_r2"])
    average,wins = float(np.mean(differences)),int(np.sum(np.asarray(differences)>0))
    return {"comparison":f"{a} minus {b}","paired_seeds":seeds,
            "paired_seed_differences":differences,"mean_difference":average,
            "wins":wins,"out_of":len(seeds),
            "passed_predeclared_gate":bool(average>=.02 and wins>=3)}


def reader_experiment(baseline_receipt,progress=print):
    frozen = baseline_receipt["memory"]["validation_selected"]
    variants = ["geometry0_shear0","geometry0_shear2","geometry1_shear0","geometry1_shear2","linear"]
    seeds = list(range(20261007,20261011))
    if (len(frozen) != 20 or {(r["seed"],r["variant"]) for r in frozen}
        != {(s,v) for s in seeds for v in variants}):
        raise ValueError("baseline receipt must provide the twenty protocol-frozen models")
    rows = []
    for configuration in frozen:
        seed = configuration["seed"]
        model = make_model(seed,configuration["geometry"],configuration["beta"],
                           configuration["offset_from_coupled_boundary"],configuration["cubic"])
        started = time.perf_counter()
        data,descriptions,input_hashes = fresh_datasets(model,seed+60000)
        integration_seconds = time.perf_counter()-started
        row = {k:configuration[k] for k in ("seed","variant","geometry","beta","cubic",
                                            "offset_from_coupled_boundary")}
        row.update({"mu":model.mu,"input_seed":seed+60000,"shuffle_seed":seed+70000,
                    "input_sha256_by_split":input_hashes,
                    "state_sha256_by_split":[hashlib.sha256(x.astype("<f8").tobytes()).hexdigest()
                                             for x,_ in data],
                    "integration_seconds":integration_seconds,
                    "readers":compare_readers(data,descriptions),
                    "shuffled_label_control":compare_readers(data,descriptions,seed+70000)})
        rows.append(row)
        progress(f"reader seed {seed}: {row['variant']} complete",flush=True)
    comparisons = [("geometry1_shear2","geometry0_shear2"),
                   ("geometry0_shear2","geometry0_shear0"),
                   ("geometry1_shear2","geometry1_shear0")]
    comparisons.extend((v,"linear") for v in variants if v != "linear")
    return {"rows":rows,"target_descriptions":descriptions,
            "primary_reader_gate":reader_gate(rows,"geometry0_shear0"),
            "reader_differences_by_variant":[reader_gate(rows,v) for v in variants],
            "secondary_model_gates":[model_comparison(rows,a,b) for a,b in comparisons],
            "selection":"operating offsets frozen from earlier validation; ridge chosen on fresh validation only",
            "budget":{"recurrent_state_coordinates":16,
                      "raw":{"features":16,"coefficients":306},
                      "quadratic":{"features":40,"coefficients":738},
                      "targets":18,"extra_observables_are_computed_from_current_state":True}}
