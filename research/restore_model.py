"""Two-pulse controllers based on supplied goals and measured local outputs."""
from dataclasses import dataclass

import numpy as np

from ping_model import MU,PortRecord,complex_noise,uncoupled_step


@dataclass(frozen=True)
class PairedDiagnostic:
    post: np.ndarray
    unpinged_post: np.ndarray
    pulse_energy: np.ndarray


def pulse_pair(bank,local_pre,goals,amplitude,family='fixed',pattern='opposite'):
    """Return waveforms; radial balance uses observed magnitudes, not a twin.

    All active two-pulse patterns have energy proxy 2*N*mu*amplitude**2 per
    episode. The single-pulse diagnostic has the corresponding missing second
    pulse. Electrical sign inversion differs from spatial reflection of a goal.
    """
    z,g = np.asarray(local_pre,dtype=complex),np.asarray(goals,dtype=float)
    if (z.ndim!=2 or not len(z) or z.shape[1]!=bank.units or g.shape!=(len(z),2)
        or not np.all(np.isfinite(z)) or not np.all(np.isfinite(g))
        or not np.isfinite(amplitude) or amplitude<0
        or family not in ('fixed','radial_balance')
        or pattern not in ('repeat','mirror','opposite','passive','single')):
        raise ValueError('pulse controller requires finite matching outputs/goals and a known family/pattern')
    ratio = np.ones_like(z.real)
    if family=='radial_balance':
        radius = np.abs(z)
        if np.any(radius<=1e-12):
            raise ValueError('radial balance requires nonzero measured magnitudes')
        ratio = np.abs(uncoupled_step(z,0.,4.))/radius
    normalization = np.sqrt((bank.units+np.sum(ratio**2,axis=1))/(2*bank.units))
    first = amplitude*np.sqrt(MU)*np.exp(1j*(g@bank.wavevectors.T))/normalization[:,None]
    if pattern=='mirror':
        reflected = g.copy()
        reflected[:,1] = 1-reflected[:,1]
        second = amplitude*np.sqrt(MU)*ratio*np.exp(1j*(reflected@bank.wavevectors.T))/normalization[:,None]
    else:
        second = ratio*first*(-1 if pattern=='opposite' else 1)
    if pattern in ('single','passive'):
        second = np.zeros_like(second)
    if pattern=='passive':
        first = np.zeros_like(first)
    return first,second


def paired_query(bank,z,goals,amplitude,rng,sigma=.05,family='fixed',
                 pattern='opposite',reference=None,dt=.25):
    """Listen at 1,2,4, compensate at 4, then retain the bank at time 8.

    The PortRecord contains only the queried bank's outputs collected BEFORE
    the second pulse. The unqueried copy is for causal damage measurement only.
    A supplied reference advances accumulated controls through repeated pairs.
    """
    first,second = pulse_pair(bank,z,goals,amplitude,family,pattern)
    z = np.asarray(z,dtype=complex)
    if not np.all(np.isfinite([sigma,dt])) or sigma<0 or dt<=0 or dt>1:
        raise ValueError('query requires finite nonnegative noise and a positive dt <= 1')
    unit_steps = int(round(1/dt))
    if abs(unit_steps*dt-1)>1e-12:
        raise ValueError('query dt must divide one time unit exactly')
    unpinged = z.copy() if reference is None else np.array(reference,dtype=complex,copy=True)
    if unpinged.shape!=z.shape or not np.all(np.isfinite(unpinged)):
        raise ValueError('reference must match the queried state')
    pinged = z+first
    samples = []
    for step in range(1,8*unit_steps+1):
        noise = complex_noise(rng,z.shape,sigma,dt)
        pinged = uncoupled_step(pinged,0.,dt)+noise
        unpinged = uncoupled_step(unpinged,0.,dt)+noise
        if step in (unit_steps,2*unit_steps,4*unit_steps):
            samples.append(pinged.copy())
        if step==4*unit_steps:
            pinged = pinged+second
    samples = np.stack(samples,axis=1)
    record = PortRecord(z.copy(),samples,z@bank.sum_weights,samples@bank.sum_weights)
    energy = np.sum(abs(first)**2+abs(second)**2,axis=1)
    return record,PairedDiagnostic(pinged,unpinged,energy)
