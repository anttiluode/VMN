"""Measured-port features and readers with explicit train/validation separation."""
from dataclasses import dataclass

import numpy as np


def local_features(record):
    phase = np.angle(record.local_samples*np.conj(record.local_pre[:,None,:]))
    return phase.reshape(len(phase),-1)


def sum_features(record,goals):
    change = record.sum_samples-record.sum_pre[:,None,:]
    return np.c_[record.sum_pre.real,record.sum_pre.imag,
                  change.real.reshape(len(change),-1),change.imag.reshape(len(change),-1),goals]


def phase_features(z):
    unit = z/np.maximum(np.abs(z),1e-12)
    return np.c_[unit.real,unit.imag]


def direct_features(bank,z,goals):
    relative = goals@bank.wavevectors.T-np.angle(z)
    return np.c_[np.sin(relative),np.cos(relative)]


@dataclass
class FittedReader:
    kind: str
    parameter: float
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray | None
    training_features: np.ndarray | None
    training_targets: np.ndarray | None
    validation_error: float
    candidate_validation: list

    @property
    def storage_scalars(self):
        learned = self.weights.size if self.weights is not None else self.training_features.size+self.training_targets.size
        return int(self.mean.size+self.scale.size+learned)

    def predict(self,features):
        features = np.asarray(features,dtype=float)
        if features.ndim!=2 or features.shape[1]!=len(self.mean) or not np.all(np.isfinite(features)):
            raise ValueError('prediction features must match the fitted reader and be finite')
        x = (features-self.mean)/self.scale
        if self.kind=='ridge':
            return np.c_[x,np.ones(len(x))]@self.weights
        return _neighbors(x,self.training_features,self.training_targets,int(self.parameter))


def _neighbors(x,training,targets,k):
    # Bounded temporary storage; O(query rows * training rows * features).
    predictions = []
    for first in range(0,len(x),256):
        batch = x[first:first+256]
        distance = np.maximum(0.,np.sum(batch**2,axis=1)[:,None]
                              +np.sum(training**2,axis=1)[None,:]-2*batch@training.T)
        index = np.argpartition(distance,k-1,axis=1)[:,:k]
        predictions.append(targets[index].mean(axis=1))
    return np.concatenate(predictions) if predictions else np.empty((0,targets.shape[1]))


def fit_reader(x_train,y_train,x_val,y_val):
    arrays = [np.asarray(a,dtype=float) for a in (x_train,y_train,x_val,y_val)]
    x_train,y_train,x_val,y_val = arrays
    if (any(a.ndim!=2 or not len(a) or not np.all(np.isfinite(a)) for a in arrays)
        or len(x_train)!=len(y_train) or len(x_val)!=len(y_val)
        or x_train.shape[1]!=x_val.shape[1] or y_train.shape[1]!=y_val.shape[1]):
        raise ValueError('reader splits must be finite nonempty matrices with matching rows/features/targets')
    mean,scale = x_train.mean(axis=0),x_train.std(axis=0)
    scale = np.maximum(scale,1e-9)
    training,validation = (x_train-mean)/scale,(x_val-mean)/scale
    augmented = np.c_[training,np.ones(len(training))]
    validation_augmented = np.c_[validation,np.ones(len(validation))]
    gram,rhs = augmented.T@augmented,augmented.T@y_train
    candidates,choices = [],[]
    for penalty in (.01,.1,1.,10.,100.):
        weights = np.linalg.solve(gram+penalty*np.eye(len(gram)),rhs)
        error = float(np.linalg.norm(validation_augmented@weights-y_val,axis=1).mean())
        candidates.append({'kind':'ridge','parameter':penalty,'validation_error':error})
        choices.append(('ridge',penalty,weights,error))
    for count in (5,15,40):
        if count>len(training):
            continue
        error = float(np.linalg.norm(_neighbors(validation,training,y_train,count)-y_val,axis=1).mean())
        candidates.append({'kind':'knn','parameter':count,'validation_error':error})
        choices.append(('knn',count,None,error))
    kind,parameter,weights,error = min(choices,key=lambda item:item[3])
    return FittedReader(kind,parameter,mean,scale,weights,
                        training.copy() if kind=='knn' else None,
                        y_train.copy() if kind=='knn' else None,error,candidates)
