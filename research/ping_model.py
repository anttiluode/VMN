"""Noisy phase memory with explicit output ports and counterfactual diagnostics.

The bank is a designed oscillator model, not a simulation of neural tissue or
fluid. Only PortRecord is an observation available to a physical listener.
"""
from dataclasses import dataclass

import numpy as np

from check_math import positions, vortex_jacobian
from geometry_lift import real_to_complex


MU = .5


@dataclass(frozen=True)
class PhaseBank:
    wavevectors: np.ndarray
    sum_weights: np.ndarray
    vortex_a: np.ndarray
    vortex_b: np.ndarray

    def __post_init__(self):
        wavevectors = np.asarray(self.wavevectors,dtype=float)
        if wavevectors.ndim != 2 or wavevectors.shape[1] != 2 or len(wavevectors)<2:
            raise ValueError('wavevectors must be an N by 2 array, with N >= 2')
        n = len(wavevectors)
        for name,value,shape,dtype in (
            ('wavevectors',wavevectors,(n,2),float),
            ('sum_weights',self.sum_weights,(n,2),complex),
            ('vortex_a',self.vortex_a,(n,n),complex),
            ('vortex_b',self.vortex_b,(n,n),complex)):
            value = np.array(value,dtype=dtype,copy=True)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f'{name} must be finite with shape {shape}')
            value.setflags(write=False)
            object.__setattr__(self,name,value)

    @property
    def units(self):
        return len(self.wavevectors)


@dataclass(frozen=True)
class PortRecord:
    local_pre: np.ndarray
    local_samples: np.ndarray
    sum_pre: np.ndarray
    sum_samples: np.ndarray


@dataclass(frozen=True)
class QueryDiagnostic:
    post: np.ndarray
    unpinged_post: np.ndarray
    twin_phase: np.ndarray


def make_bank(seed,units=32):
    if not isinstance(units,int) or units<2:
        raise ValueError('unit count must be an integer >= 2')
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.,2*np.pi,units)
    frequency = 2*np.pi*np.resize([1.,2.,4.],units)
    wavevectors = frequency[:,None]*np.c_[np.cos(angle),np.sin(angle)]
    weights,_ = np.linalg.qr(rng.normal(size=(units,2))+1j*rng.normal(size=(units,2)))
    anchors = positions(units,rng)
    gamma = rng.uniform(.8,1.2,units)
    kernel = vortex_jacobian(anchors,gamma,.15)
    kernel /= np.linalg.norm(kernel,2)
    a,b = real_to_complex(kernel)
    return PhaseBank(wavevectors,weights,a,b)


def uncoupled_step(z,phase_advance,dt,mu=MU):
    z = np.asarray(z,dtype=complex)
    if (not np.all(np.isfinite(z)) or not np.all(np.isfinite(phase_advance))
        or not np.all(np.isfinite([dt,mu])) or dt<=0 or mu<=0):
        raise ValueError('flow requires finite state/phase, dt > 0 and mu > 0')
    growth = np.expm1(2*mu*dt)
    return z*np.exp(mu*dt+1j*np.asarray(phase_advance))/np.sqrt(1+np.abs(z)**2/mu*growth)


def complex_noise(rng,shape,sigma,dt):
    return sigma*np.sqrt(dt/2)*(rng.normal(size=shape)+1j*rng.normal(size=shape))


def velocity_trajectory(seed,episodes,steps=200):
    if not isinstance(episodes,int) or episodes<=0 or not isinstance(steps,int) or steps<=0:
        raise ValueError('episode and step counts must be positive integers')
    rng = np.random.default_rng(seed)
    start = rng.uniform(.1,.9,(episodes,2))
    x = start.copy()
    velocity = rng.normal(size=x.shape)*.02
    increments = np.empty((steps,episodes,2))
    for t in range(steps):
        velocity = .9*velocity+.1*rng.normal(size=x.shape)*.03
        next_x = x+velocity
        for d in range(2):
            lo,hi = next_x[:,d]<0,next_x[:,d]>1
            next_x[lo,d] *= -1
            next_x[hi,d] = 2-next_x[hi,d]
            velocity[lo|hi,d] *= -1
        increments[t] = next_x-x
        x = next_x
    return start,increments,x


def store_path(bank,start,increments,rng,sigma=.05,velocity_bias=None):
    start = np.asarray(start,dtype=float)
    increments = np.asarray(increments,dtype=float)
    if (start.ndim!=2 or start.shape[1]!=2 or increments.ndim!=3
        or increments.shape[1:]!=start.shape or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(increments)) or not np.isfinite(sigma) or sigma<0):
        raise ValueError('path arrays must be finite and have matching episode/coordinate dimensions')
    bias = np.zeros_like(start) if velocity_bias is None else np.asarray(velocity_bias,dtype=float)
    if bias.shape!=start.shape or not np.all(np.isfinite(bias)):
        raise ValueError('velocity bias must match the starting positions')
    z = np.sqrt(MU)*np.exp(1j*(start@bank.wavevectors.T))
    for displacement in increments:
        advance = (displacement+bias)@bank.wavevectors.T/4
        for _ in range(4):
            z = uncoupled_step(z,advance,.25)+complex_noise(rng,z.shape,sigma,.25)
    return z


def _query_step(bank,z,dt,kappa):
    if kappa==0:
        return uncoupled_step(z,0.,dt)
    def rhs(q):
        return (MU-np.abs(q)**2)*q+kappa*(q@bank.vortex_a.T+q.conj()@bank.vortex_b.T)
    k1 = rhs(z)
    k2 = rhs(z+.5*dt*k1)
    k3 = rhs(z+.5*dt*k2)
    k4 = rhs(z+dt*k3)
    return z+dt/6*(k1+2*k2+2*k3+k4)


def query(bank,z,goals,amplitude,rng,sigma=.05,kappa=0.,dt=.25,reference=None):
    """Observe one bank; return separate privileged twins for diagnostics only.

    reference optionally carries the un-kicked history through repeated queries.
    It never contributes to PortRecord. All samples are outputs of the evolving
    kicked state, not exponentials of a frozen tangent.
    """
    z,goals = np.asarray(z,dtype=complex),np.asarray(goals,dtype=float)
    if (z.ndim!=2 or z.shape[1]!=bank.units or goals.shape!=(len(z),2)
        or not np.all(np.isfinite(z)) or not np.all(np.isfinite(goals))
        or not np.all(np.isfinite([amplitude,sigma,kappa,dt]))
        or min(amplitude,sigma,kappa)<0 or dt<=0 or dt>1):
        raise ValueError('query requires matching finite states/goals and nonnegative pulse/noise/coupling')
    unit_steps = int(round(1/dt))
    if abs(unit_steps*dt-1)>1e-12:
        raise ValueError('query dt must divide one time unit exactly')
    unpinged = z.copy() if reference is None else np.array(reference,dtype=complex,copy=True)
    if unpinged.shape!=z.shape or not np.all(np.isfinite(unpinged)):
        raise ValueError('reference must match the query state')
    pinged = z+amplitude*np.sqrt(MU)*np.exp(1j*(goals@bank.wavevectors.T))
    samples,twin = [],[]
    for step in range(1,4*unit_steps+1):
        n = complex_noise(rng,z.shape,sigma,dt)
        pinged = _query_step(bank,pinged,dt,kappa)+n
        unpinged = _query_step(bank,unpinged,dt,kappa)+n
        if step in (unit_steps,2*unit_steps,4*unit_steps):
            samples.append(pinged.copy())
            twin.append(np.angle(pinged*np.conj(unpinged)))
    samples = np.stack(samples,axis=1)
    record = PortRecord(z.copy(),samples,z@bank.sum_weights,samples@bank.sum_weights)
    return record,QueryDiagnostic(pinged,unpinged,np.stack(twin,axis=1))
