"""Matched fixed/moving geometry oscillators; a designed research model.

This is not a Navier–Stokes solver. The moving coupling is the nonlinear
increment of a regularized vortex velocity; the fixed coupling is its tangent.
"""
from dataclasses import dataclass, field

import numpy as np

from check_math import ROT, SEED, positions, vortex_jacobian, vortex_velocity


@dataclass
class GeometryModel:
    anchors: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray
    decay: np.ndarray
    input_vector: np.ndarray
    coupling: float
    mu: float
    beta: float
    geometry: float
    cubic: float = 1.0
    smoothing: float = 0.15
    fixed_jacobian: np.ndarray = field(init=False, repr=False)
    anchor_velocity: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.anchors = np.asarray(self.anchors, dtype=float).copy()
        if self.anchors.ndim != 2 or self.anchors.shape[1] != 2 or not len(self.anchors):
            raise ValueError("anchors must contain at least one planar position")
        n = len(self.anchors)
        for name, size in [("gamma",n),("omega",n),("decay",n),("input_vector",2*n)]:
            value = np.asarray(getattr(self,name), dtype=float).copy()
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} has an invalid shape or non-finite values")
            setattr(self,name,value)
        scalars = [self.coupling,self.mu,self.beta,self.geometry,self.cubic,self.smoothing]
        if not np.all(np.isfinite(self.anchors)) or not np.all(np.isfinite(scalars)):
            raise ValueError("model parameters must be finite")
        if min(self.geometry,self.cubic,self.smoothing) < 0:
            raise ValueError("geometry, cubic and smoothing must be nonnegative")
        self.fixed_jacobian = vortex_jacobian(self.anchors.ravel(),self.gamma,self.smoothing)
        self.anchor_velocity = vortex_velocity(self.anchors.ravel(),self.gamma,self.smoothing)
        if not np.all(np.isfinite(self.fixed_jacobian)):
            raise ValueError("singular vortex anchors; separate them or enable smoothing")

    @property
    def dimension(self):
        return 2*len(self.gamma)

    def vector_field(self, q, u=0.0):
        q = np.asarray(q).reshape(-1,2)
        turn = q @ ROT.T
        radius2 = np.einsum("ij,ij->i",q,q)
        local = ((self.mu-self.decay)[:,None]*q + self.omega[:,None]*turn
                 - self.cubic*radius2[:,None]*(q+self.beta*turn))
        if self.geometry == 0:
            coupling = self.coupling*(self.fixed_jacobian @ q.ravel())
        else:
            p = self.anchors.ravel()+self.geometry*q.ravel()
            coupling = self.coupling/self.geometry*(
                vortex_velocity(p,self.gamma,self.smoothing)-self.anchor_velocity)
        return local.ravel()+coupling+self.input_vector*u

    def jacobian(self, q):
        q = np.asarray(q).reshape(-1,2)
        r2 = np.einsum("ij,ij->i",q,q)
        outer = np.einsum("ia,ib->iab",q,q)
        blocks = ((self.mu-self.decay)[:,None,None]*np.eye(2)
                  +self.omega[:,None,None]*ROT
                  -self.cubic*np.einsum("ab,ibc->iac",np.eye(2)+self.beta*ROT,
                                       r2[:,None,None]*np.eye(2)+2*outer))
        result = np.zeros((self.dimension,self.dimension))
        for i,b in enumerate(blocks):
            result[2*i:2*i+2,2*i:2*i+2] = b
        kernel = self.fixed_jacobian if self.geometry == 0 else vortex_jacobian(
            self.anchors.ravel()+self.geometry*q.ravel(),self.gamma,self.smoothing)
        return result+self.coupling*kernel

    def stability_boundary(self):
        at_zero_mu = self.jacobian(np.zeros(self.dimension))-self.mu*np.eye(self.dimension)
        eigenvalues = np.linalg.eigvals(at_zero_mu)
        abscissa = float(np.max(eigenvalues.real))
        leading = eigenvalues[eigenvalues.real >= abscissa-1e-8]
        return {"mu_critical":-abscissa,
                "critical_modes":[{"real":float(e.real-abscissa),"imag":float(e.imag)}
                                  for e in leading],
                "oscillatory_crossing_candidate":bool(np.all(abs(leading.imag)>1e-8)),
                "note":"linear stability boundary; nonlinear Hopf conditions are not established"}


def make_model(seed=SEED, geometry=0.0, beta=0.0, offset=-.03, cubic=1.0):
    rng = np.random.default_rng(seed)
    n = 8
    anchors = positions(n,rng).reshape(n,2)
    gamma = rng.uniform(.8,1.2,n)
    omega = rng.permutation(np.linspace(1.,1.7,n))
    decay = rng.permutation(np.linspace(.04,.20,n))
    inputs = rng.normal(size=2*n)
    inputs *= .15/np.linalg.norm(inputs)
    kernel = vortex_jacobian(anchors.ravel(),gamma,.15)
    coupling = .35/np.linalg.norm(kernel,2)
    model = GeometryModel(anchors,gamma,omega,decay,inputs,coupling,0.,beta,geometry,cubic)
    model.mu = model.stability_boundary()["mu_critical"]+offset
    return model


def rollout(model, inputs, dt=.25, substeps=2, initial=None):
    inputs = np.asarray(inputs,dtype=float)
    if inputs.ndim != 1 or not np.all(np.isfinite(inputs)):
        raise ValueError("inputs must be a finite scalar sequence")
    if not np.isfinite(dt) or dt <= 0 or not isinstance(substeps,int) or substeps <= 0:
        raise ValueError("dt and the integer substep count must be positive")
    q = np.zeros(model.dimension) if initial is None else np.asarray(initial,dtype=float).copy()
    if q.shape != (model.dimension,) or not np.all(np.isfinite(q)):
        raise ValueError("initial state has an invalid shape or non-finite values")
    out = np.empty((len(inputs),model.dimension))
    step = dt/substeps
    rhs = model.vector_field
    for t,u in enumerate(inputs):
        for _ in range(substeps):
            k1 = rhs(q,u)
            k2 = rhs(q+.5*step*k1,u)
            k3 = rhs(q+.5*step*k2,u)
            k4 = rhs(q+step*k3,u)
            q += step/6*(k1+2*k2+2*k3+k4)
        if not np.all(np.isfinite(q)) or np.linalg.norm(q)>1e3:
            raise FloatingPointError("reservoir trajectory diverged")
        out[t] = q
    return out


def phase_kick(mu,beta,rho):
    if not np.all(np.isfinite([mu,beta,rho])) or mu <= 0 or np.sqrt(mu)+rho <= 0:
        raise ValueError("phase kick requires mu > 0 and a positive post-kick radius")
    return float(-beta*np.log1p(rho/np.sqrt(mu)))


def relaxed_response_norm(mu,beta,phase_shift):
    if not np.all(np.isfinite([mu,beta,phase_shift])) or mu < 0:
        raise ValueError("relaxed-cycle parameters must be finite with mu >= 0")
    return float(2*mu*np.sqrt(1+beta*beta)*abs(np.sin(phase_shift)))
