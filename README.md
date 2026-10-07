# VMN — Vortex, Matrix, Neuron

**A mathematical investigation of how a small dynamical state changes the response of a larger machine.**

The motivating question is: can an earlier input change the *operator through which the next input is processed*, and can that change be retained cheaply?

The bridge is the tangent matrix of nonlinear dynamics:

$$
\dot x=f(x)+Bu,\qquad
J(x)=D_x f(x),\qquad
\delta\dot x=J(x(t))\delta x+B\delta u.
$$

A past event changes the current state $x$. If that changes $J$, the same later pulse has a different response. Vortices, nonlinear neural activity and oscillators give different concrete realizations of this structure.

This repository currently contains **derivations and a reproducible numerical research probe**. A learned vortex layer, a cylinder-wake solver and competitive ML benchmarks remain future work.

- [Mathematical derivations, proofs and falsifiers](MATH.md)
- [Runnable numerical checks](research/check_math.py)
- [Machine-readable receipt](results/math_checks.json)

![VMN mathematical checks](results/math_summary.svg)

## What the mathematics established

1. **Nonlinearity enables history to change the next response operator.** A fixed linear system with additive inputs and a linear observation retains history in its state, but its probe-response operator is independent of that history.
2. **Oscillation supplies amplitude and phase.** In a Hopf normal form, radial perturbations decay while phase perturbations can persist. The orientation of the current state changes which directions are suppressed by the nonlinear response.
3. **A small controlling state does not imply a low-rank matrix change.** Moving one of $N$ ordinary point vortices generically changes the instantaneous Jacobian at rank $2N-2$. Its description can still be short: a shared baseline, a vortex index and two displacement coordinates.
4. **Hopf onset does not guarantee low-rank updates.** An explicit 34-dimensional counterexample has one critical Hopf pair, but an operator update requiring 32 modes for 95% of its energy. Three shared matrix coefficients nevertheless describe that update exactly. Low rank and a compact operator codebook are different kinds of compression.

The elementary proofs are included in [MATH.md](MATH.md). Their research novelty has not been assessed.

## Fresh numerical evidence — 7 October 2026

| Check or experiment | Result |
|---|---:|
| Numerical consistency checks | **63 / 63 pass** |
| Analytic point-vortex Jacobian vs central differences | relative error **7.19 × 10⁻¹¹** |
| One-vortex instantaneous write, $N=16$, 30 predefined events | rank **30 / 32** in every case |
| Finite response baseline, eight vortices, all 16 position probes | **15 modes** for 95% energy |
| Finite response update, 32 predefined events | median **5 modes**, range **3–8** |
| Update norm / baseline norm in that scan | median **0.736%** |
| Shared update dictionary: 20 training events at five sites | **8 basis components** for 95% training energy |
| Same dictionary, 12 events at three unseen sites | median relative reconstruction error **93.9%** |
| Hopf counterexample at onset | update rank **34**, 95%-energy rank **32** |
| Same Hopf update, shared analytic matrix dictionary | **3 coefficients**, exact reconstruction |

The vortex dictionary test is an **optimistic reconstruction bound**: coefficients are obtained by projecting the true held-out operator. No coefficient predictor was trained. Its poor transfer shows that individually compact updates need not share a transferable fixed basis. A geometry-dependent decoder remains an open alternative.

These are ideal point-vortex and normal-form experiments, with fixed circulations, noiseless observations and finite horizons. The median vortex effect is small but well above the derivative-check error. This does not establish task utility, persistent learning or reduced total storage.

## The cylinder paper and the Hopf hypothesis

Goto, Nakajima and Notsu's [2020 preprint](https://arxiv.org/abs/2001.08502), later published as [*Twin vortex computer in fluid flow*](https://doi.org/10.1088/1367-2630/ac024d), reports the best performance in its setup **just below vortex-shedding onset**, around $Re=40$ with onset around $45$. Synchronization deteriorates after shedding begins. It does not measure the rank or compressibility of response updates.

Ordinary free point vortices are inviscid Hamiltonian dynamics. They have no Reynolds-number control and cannot reproduce that dissipative transition. VMN therefore keeps the two models separate: point vortices test response geometry; a dissipative Hopf normal form tests the onset argument.

The useful research target is **a compact state or codebook that predicts changes in future responses**, with generalization and decoder cost measured explicitly. A claim that compactness peaks near onset still needs a driven, higher-dimensional experiment.

## Reproduce

Checked with Python 3.12.14 and the versions in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python research/check_math.py
```

The script regenerates `results/math_checks.json` and `results/math_summary.svg`, prints a concise receipt, and exits nonzero if a numerical consistency check fails. A failed research hypothesis is recorded as a finding, rather than treated as a software failure.

Related work: [Kompressori](https://github.com/anttiluode/Kompressori) measures compact finite-response changes in a different nonlinear field. Its five-row local bound does not transfer to globally coupled point vortices.

License: [MIT](LICENSE).
