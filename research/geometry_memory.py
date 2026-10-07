"""Training-only affine readouts for independent held-out input histories."""
from dataclasses import dataclass

import numpy as np

from geometry_model import make_model, rollout


LAGS = (1,2,4,8,16,32)
OFFSETS = (-.30,-.10,-.03,.03,.10)
PENALTIES = (1e-6,1e-4,1e-2,1.)
WASHOUT = 256
LENGTHS = (1200,400,800)


def memory_targets(inputs,lags=LAGS):
    inputs = np.asarray(inputs,dtype=float)
    if (inputs.ndim != 1 or not np.all(np.isfinite(inputs)) or not len(lags)
        or any(not isinstance(lag,(int,np.integer)) or lag < 1 for lag in lags)
        or len(inputs) <= max(lags)):
        raise ValueError("targets require a finite sequence and positive integer past lags")
    indices = np.arange(max(lags),len(inputs))
    lagged = np.column_stack([inputs[indices-lag] for lag in lags])
    targets = np.column_stack([lagged,(3*lagged**2-1)/2,(5*lagged**3-3*lagged)/2])
    descriptions = [{"order":order,"lag":int(lag)} for order in (1,2,3) for lag in lags]
    return indices,targets,descriptions


def matrix(value,name):
    value = np.asarray(value,dtype=float)
    if value.ndim != 2 or not len(value) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return value


@dataclass
class Readout:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def predict(self,x):
        x = matrix(x,"states")
        if x.shape[1] != len(self.mean):
            raise ValueError("readout state dimension differs from training")
        z = np.column_stack([(x-self.mean)/self.scale,np.ones(len(x))])
        return z @ self.weights


def fit_readout(x,y,penalty):
    x,y = matrix(x,"states"),matrix(y,"targets")
    if len(x) != len(y) or not np.isfinite(penalty) or penalty < 0:
        raise ValueError("readout rows must match and penalty must be finite and nonnegative")
    mean,scale = np.mean(x,axis=0),np.std(x,axis=0)
    scale = np.where(scale>1e-12,scale,1.)
    z = np.column_stack([(x-mean)/scale,np.ones(len(x))])
    if penalty == 0:
        weights = np.linalg.lstsq(z,y,rcond=None)[0]
    else:
        regularizer = np.diag(np.r_[np.full(x.shape[1],penalty*len(x)),0.])
        weights = np.linalg.solve(z.T @ z+regularizer,z.T @ y)
    return Readout(mean,scale,weights)


def select_readout(x,y,x_validation,y_validation,penalties=PENALTIES):
    variance = np.maximum(np.var(y,axis=0),1e-30)
    selected = None
    for penalty in penalties:
        fitted = fit_readout(x,y,penalty)
        error = float(np.mean((fitted.predict(x_validation)-y_validation)**2/variance))
        if selected is None or error < selected[0]:
            selected = (error,fitted,penalty)
    if selected is None:
        raise ValueError("provide at least one ridge penalty")
    return selected[1],float(selected[2])


def score_predictions(target,prediction):
    target,prediction = matrix(target,"targets"),matrix(prediction,"predictions")
    if target.shape != prediction.shape:
        raise ValueError("target and prediction shapes must match")
    centered = target-np.mean(target,axis=0)
    pred_centered = prediction-np.mean(prediction,axis=0)
    variance = np.mean(centered**2,axis=0)
    r2 = np.where(variance>1e-30,1-np.mean((prediction-target)**2,axis=0)/np.maximum(variance,1e-30),0.)
    denom = np.sum(centered**2,axis=0)*np.sum(pred_centered**2,axis=0)
    correlation = np.sum(centered*pred_centered,axis=0)**2/np.maximum(denom,1e-30)
    return {"r2":r2.tolist(),"correlation_squared":correlation.tolist()}


def summarize_scores(scores,descriptions):
    r2 = np.array(scores["r2"])
    corr = np.array(scores["correlation_squared"])
    nonlinear = np.array([d["order"]>1 for d in descriptions])
    long_lag = np.array([d["lag"]>=16 for d in descriptions])
    return {"mean_nonlinear_r2":float(np.mean(r2[nonlinear])),
            "mean_linear_r2":float(np.mean(r2[~nonlinear])),
            "mean_long_lag_r2":float(np.mean(r2[long_lag])),
            "memory_capacity_sum_correlation_squared":float(np.sum(corr)),
            "capacity_by_order":{str(order):float(np.sum(corr[[d["order"]==order for d in descriptions]]))
                                  for order in (1,2,3)}}


def memory_experiment(seeds,progress=print):
    rows = []
    descriptions = None
    for seed in seeds:
        rng = np.random.default_rng(seed+10000)
        inputs = [rng.uniform(-1,1,WASHOUT+length) for length in LENGTHS]
        target_sets = [memory_targets(sequence) for sequence in inputs]
        variants = [(geometry,beta,1.) for geometry in (0.,1.) for beta in (0.,2.)]+[(0.,0.,0.)]
        for geometry,beta,cubic in variants:
            label = "linear" if cubic == 0 else f"geometry{int(geometry)}_shear{int(beta)}"
            offsets = OFFSETS if cubic else OFFSETS[:3]
            for offset in offsets:
                model = make_model(seed,geometry,beta,offset,cubic)
                states = [rollout(model,sequence) for sequence in inputs]
                datasets = []
                for state,(indices,targets,descriptions) in zip(states,target_sets):
                    keep = indices>=WASHOUT
                    datasets.append((state[indices[keep]],targets[keep]))
                readout,penalty = select_readout(*datasets[0],*datasets[1])
                validation_scores = score_predictions(datasets[1][1],readout.predict(datasets[1][0]))
                test_scores = score_predictions(datasets[2][1],readout.predict(datasets[2][0]))
                rows.append({"seed":int(seed),"variant":label,"geometry":geometry,"beta":beta,
                             "cubic":cubic,"offset_from_coupled_boundary":offset,
                             "mu":model.mu,"coupled_boundary":model.stability_boundary(),
                             "ridge_penalty":penalty,"validation":summarize_scores(validation_scores,descriptions),
                             "test":summarize_scores(test_scores,descriptions),"test_per_target":test_scores,
                             "state_rms":float(np.sqrt(np.mean(states[2][WASHOUT:]**2)))})
            progress(f"memory seed {seed}: {label} complete",flush=True)
    selected = []
    for seed in seeds:
        labels = sorted({row["variant"] for row in rows if row["seed"]==seed})
        for label in labels:
            options = [row for row in rows if row["seed"]==seed and row["variant"]==label]
            best = max(options,key=lambda row:row["validation"]["mean_nonlinear_r2"])
            selected.append(best)
    def paired(a,b):
        differences = []
        for seed in seeds:
            ra = next(row for row in selected if row["seed"]==seed and row["variant"]==a)
            rb = next(row for row in selected if row["seed"]==seed and row["variant"]==b)
            differences.append(ra["test"]["mean_nonlinear_r2"]-rb["test"]["mean_nonlinear_r2"])
        average,wins = float(np.mean(differences)),int(np.sum(np.array(differences)>0))
        return {"comparison":f"{a} minus {b}","paired_seed_differences":differences,
                "mean_difference":average,"wins":wins,"out_of":len(seeds),
                "passed_predeclared_gate":bool(average>=.02 and wins>=3)}
    return {"target_descriptions":descriptions,"configurations":rows,"validation_selected":selected,
            "geometry_gate":paired("geometry1_shear2","geometry0_shear2"),
            "shear_gates":[paired(f"geometry{i}_shear2",f"geometry{i}_shear0") for i in (0,1)],
            "selection":"ridge and offset selected on validation; test trajectories are independent",
            "budget":{"state_coordinates":16,"readout_coefficients":17*18,
                      "train_samples":1200,"validation_samples":400,"test_samples":800,
                      "washout_samples_per_split":WASHOUT}}
