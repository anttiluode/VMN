# Geometry, shear and memory: what the next VMN gate found

7 October 2026. The protocol was recorded before outcomes in
[GEOMETRY_PROTOCOL.md](GEOMETRY_PROTOCOL.md). The equations, tests and complete
receipt are linked below.

**A sparse code of the current state reconstructs responses at unseen event
locations much better than a fixed operator dictionary. The moving vortex
geometry and shear tested here did not demonstrate a nonlinear-memory advantage.**

These are separate results. The reconstruction works in both fixed and moving
geometry, so its success cannot be credited specifically to moving vortices.

![Geometry and shear gate](results/geometry_summary.svg)

## 1. A fair vortex–matrix–neuron comparison

Let $q_k\in\mathbb R^2$ be an oscillator state, $p_k$ a shared planar anchor and
$R=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$. Local dynamics are

$$
f_k(q_k)=(\mu-d_k)q_k+\omega_kRq_k
-a\|q_k\|^2(I+\beta R)q_k.
$$

The nonlinear variants have $a=1$; the linear control has $a=0$. Shear is
$\beta=0$ or $2$. An additive scalar input uses the same fixed vector $B$ in
every matched model.

For regularized point-vortex velocity, define

$$
V_k(p)=\sum_{j\ne k}\frac{\Gamma_j}{2\pi}
\frac{R(p_k-p_j)}{\|p_k-p_j\|^2+\delta^2},
\qquad \delta=0.15.
$$

The coupled dynamics are $\dot q=f(q)+C_\eta(q)+Bu$, with

$$
C_\eta(q)=
\begin{cases}
\kappa DV(p)q,&\eta=0,\\[2mm]
\dfrac{\kappa}{\eta}\big[V(p+\eta q)-V(p)\big],&\eta>0.
\end{cases}
$$

Fixed geometry uses $\eta=0$; moving geometry uses $\eta=1$. Both have exactly
the same state dimension, input and rest linearization. Taylor expansion gives

$$
C_\eta(q)=\kappa DV(p)q+\frac{\kappa\eta}{2}D^2V(p)[q,q]+O(\|q\|^3).
$$

This isolates geometry-dependent higher-order coupling instead of changing the
linear memory system at the same time. It is a **designed hybrid**, not a
Navier–Stokes reduction or a biological-neuron equivalence.

Its exact Jacobian is

$$
J(q)=\operatorname{blockdiag}_k\left[
(\mu-d_k)I+\omega_kR
-a(I+\beta R)\left(\|q_k\|^2I+2q_kq_k^\mathsf T\right)
\right]+\kappa DV(p+\eta q).
$$

The chain-rule factor $\eta$ cancels the coupling's division by $\eta$.
At $\eta=0$, the last term is the fixed matrix $\kappa DV(p)$.
Central differences and physical finite pulses verify this Jacobian and its
finite-time tangent.

The regularized kernel contains both strain and rotation. The purely
antilinear, strain-only identity applies to the unsmoothed point-vortex kernel;
the experiment keeps the complete real derivative instead of discarding the
regularization's rotation term.

## 2. Measure the coupled onset

At rest, write $J(0)=\mu I+A$. The linear stability boundary is

$$
\mu_c=-\max_{\lambda\in\operatorname{spec}(A)}\operatorname{Re}\lambda.
$$

The four layouts give $\mu_c$ between 0.03984 and 0.04039. The leading modes
are complex pairs, but this calculation alone does not prove a nonlinear Hopf
bifurcation. The experiment sweeps distance from this coupled boundary, not
distance from the isolated unit's $\mu=0$.

The five offsets are $-0.30,-0.10,-0.03,+0.03,+0.10$. Readout regularization and
the operating offset are selected using validation trajectories; test inputs
and targets select neither. Seeds, parameters and split sizes are recorded
explicitly in the receipt.

## 3. The delayed-input memory gate failed

Each reservoir has eight planar units, or 16 real state coordinates. A trained
affine readout predicts delayed Legendre orders 1, 2 and 3 of independent inputs,
at sample lags 1, 2, 4, 8, 16 and 32. Train, validation and test are separate
trajectories. Standardization uses training states only. All readouts have
306 coefficients; the full fixed protocol uses four paired seeds.

The primary threshold requires a mean nonlinear test R² improvement of at
least **0.02**, with wins on at least three of four paired seeds.

| Comparison | Mean nonlinear R² improvement | Paired wins | Gate |
|---|---:|---:|---|
| Moving minus fixed geometry, shear 2 | 0.00050 | 2 / 4 | Failed |
| Shear 2 minus shear 0, fixed geometry | 0.00390 | 4 / 4 | Failed |
| Shear 2 minus shear 0, moving geometry | 0.00276 | 3 / 4 | Failed |

The average selected nonlinear test R² is negative for every variant.
The small improvements therefore do not establish useful nonlinear history
retrieval in this setup.

There is also a reader constraint: the fixed coupling and cubic unit dynamics
are odd in state. Starting at rest and reversing an entire input history
reverses the state, while the order-2 Legendre target stays unchanged. Under
the symmetric input distribution, an affine readout of that odd state has
zero population covariance with this even target. Moving geometry can break
that symmetry through its quadratic terms, but its measured benefit here is
too small. This explains a limitation without changing the failed gate.

| Validation-selected model | Mean linear test R² | Mean nonlinear test R² |
|---|---:|---:|
| Linear oscillator control | 0.2041 | -0.00987 |
| Fixed geometry, shear 0 | 0.1010 | -0.01117 |
| Fixed geometry, shear 2 | 0.0921 | -0.00728 |
| Moving geometry, shear 0 | 0.1482 | -0.00954 |
| Moving geometry, shear 2 | 0.1513 | -0.00678 |

Moving geometry raises some linear-memory scores relative to its nonlinear
fixed-geometry counterpart, but the linear control has the highest mean linear
score. This single recipe does not establish a useful vortex memory device.
It also does not eliminate other geometries, input strengths, readers or tasks.
No post-outcome parameter search is included.

## 4. A state decoder succeeds where a fixed matrix dictionary fails

After a shared driven history, write a small impulse at one oscillator, allow
one unit of unforced evolution, then apply identical later probes to both
histories. There are 32 events: eight sites and four cardinal directions.

The response operator stacks all 16 state-probe directions at 16 observation
times. A fixed POD dictionary is fitted only at sites 0–4 and tested at sites
5–7. Its coefficients are obtained from the true operator, giving an optimistic
reconstruction bound rather than a learned predictor.

The alternative encoder observes the current state difference from the shared
baseline. It retains the largest one, two or four planar difference blocks.
Each block stores its unit index and two coordinate offsets. The decoder
restores that sparse state and regenerates the response using the known
equations. **It receives no past-event label.**

At future horizon $T=2$, held-out median relative response-update errors are:

| Model | Fixed POD oracle | State code: 3 numbers | State code: 6 numbers | State code: 12 numbers |
|---|---:|---:|---:|---:|
| Fixed geometry, shear 0 | 99.7% | 14.9% | 5.1% | 2.0% |
| Fixed geometry, shear 2 | 99.7% | 14.6% | 5.1% | 2.1% |
| Moving geometry, shear 0 | 96.4% | 14.4% | 7.8% | 3.8% |
| Moving geometry, shear 2 | 96.5% | 15.4% | 5.3% | 2.7% |

For moving geometry with shear 2, the six-number code also gives **9.6%**
median relative probe-response update error on three independently chosen directions and
**8.2%** median error in the event-induced future state change. Directional
relative errors can exceed the whole-matrix error.

This supports **state-conditioned response regeneration across locations**.
The fixed-geometry variants also benefit, so the gain comes from preserving
state and using a structural decoder; moving geometry is not required.
Only one layout and local writes are tested for this reconstruction result.

### What the costs actually are

The codes are incremental differences relative to a shared 16-coordinate
baseline. The shared model has 62 scalar parameters; the implementation also
retains 272 numbers of derivative/velocity caches. The full state itself needs
16 coordinates.

The POD oracle needs 10–15 coefficients per event at $T=2$, plus 40,960–61,440
shared matrix numbers. Its decoder is a cheap matrix combination. The state
decoder instead performs a full 272-coordinate state-and-variational ODE solve.
Both can use shared preparation/model information.

Thus six numbers can be an accurate incremental event representation here,
but this establishes **neither an overall storage saving over the full state
nor a speed advantage**. Encoding also reads the full state. Continuous
operation while retaining only the sparse code is not tested.

## 5. Shear gives phase memory, with an exact qualification

For an isolated Stuart–Landau unit,

$$
\dot z=(\mu+i\omega)z-(1+i\beta)|z|^2z,\qquad \mu>0,
$$

the stable oscillation has radius $r_*=\sqrt{\mu}$. Polar dynamics give
$\dot r=\mu r-r^3$ and $\dot\varphi=\omega-\beta r^2$.
The isochron phase $\psi=\varphi-\beta\log r$ advances at the constant rate
$\omega-\beta\mu$. A radial kick $\rho_0$ at unchanged angle consequently leaves
the exact asymptotic phase difference

$$
\boxed{\Delta\varphi=-\beta\log\left(1+\frac{\rho_0}{\sqrt{\mu}}\right).}
$$

This requires a positive post-kick radius. The familiar
$-\beta\rho_0/\sqrt{\mu}$ is its first-order approximation for
$|\rho_0|\ll\sqrt{\mu}$. A fixed fractional kick
$\rho_0=\epsilon\sqrt{\mu}$ gives a phase shift independent of $\mu$.
Increasing sensitivity is therefore not a memory-capacity theorem.

For two fully relaxed states $q_i=\sqrt{\mu}\,v_i$ whose phases differ by
$\Delta\varphi$, their instantaneous Jacobian difference is

$$
\Delta J=-2(I+\beta R)(q_1q_1^\mathsf T-q_0q_0^\mathsf T).
$$

The bracket has two singular values $\mu|\sin\Delta\varphi|$, while
$I+\beta R$ is a rotation times $\sqrt{1+\beta^2}$. Therefore

$$
\boxed{\|\Delta J\|_2=2\mu\sqrt{1+\beta^2}\,|\sin\Delta\varphi|.}
$$

The norm here is the **spectral norm**, not the Frobenius norm. With the same
small absolute kick, moving from $\mu=0.4$ to 0.00625 grows the retained phase
approximately eightfold while shrinking this response-matrix change eightfold.
Both identities are checked against independent integration and direct
Jacobian evaluation.

At a phase difference of $\pi$, opposite states have the same instantaneous
Jacobian, despite different baseline outputs. Probe-response memory and
remembered content must therefore be scored separately.

At $\mu=0$, the cubic model still has $\dot r=-r^3$ and algebraic decay:
$r(t)=r_0/\sqrt{1+2r_0^2t}$. It is not the ideal Hamiltonian-vortex limit.
Below onset, the attracting origin does not supply the persistent limit-cycle
phase mechanism above.

## 6. Where the neuron threshold connection fits

For a point-vortex pair of total circulation $\Gamma$ in external strain $e$,
the relative coordinate satisfies

$$
\dot z=\frac{i\Gamma}{2\pi\bar z}+e\bar z,\qquad
\dot\varphi=\frac{\Gamma}{2\pi r^2}-e\sin2\varphi,\qquad
\dot r=er\cos2\varphi.
$$

If added control constrains $r$, this is an Adler threshold oscillator.
The period of a $\pi$ phase advance is
$\pi/\sqrt{a^2-e^2}$ for $a=\Gamma/(2\pi r^2)>e$.
Without that constraint the radial degree of freedom remains, and the
Hamiltonian separatrix has a logarithmic period divergence. Neither free
Hamiltonian motion nor a pinned radius by itself derives a biological neuron.
The present numerical gate uses cubic oscillators rather than implementing
this pinned pair as its unit.

## 7. Interpretation and reproducibility

The compact response updates are not caused by a collapse of the whole baseline
operator within these tested horizons. For moving geometry and shear 2, the
baseline participation rank is 15.8, 15.4 and 14.7 at horizons 0.5, 2 and 6;
the median update participation rank is 2.2, 1.5 and 1.3. These are effective
spectral concentrations, not assertions that mathematical matrix rank vanished.

The cylinder-wake [paper](https://arxiv.org/abs/2001.08502) remains motivation,
not evidence for this hybrid model. Its best tested computation lies just below
shedding onset; it does not establish compact response storage.

The most concrete next question is whether a state code can remain accurate
through repeated writes and reads without reconstructing and retaining the
whole state. The present local-write reconstruction result makes that a
reasonable gate; the failed task-utility result makes broader AI claims premature.

Reproduce from the repository root:

    python -m pip install -r requirements.txt
    python -m unittest discover -s research -p 'test_*.py'
    python research/check_math.py
    python research/check_geometry.py

The new runner produces [the complete receipt](results/geometry_checks.json)
and the figure above. It exits nonzero for failed numerical checks.
Research hypotheses remain recorded as passed or failed results.
The source hashes include the protocol, every numerical source and all new tests.

The mathematical checks and implementation are independent work in VMN.
[VMNClaude](https://github.com/anttiluode/VMNClaude) supplied useful inspiration
for shear, the vortex-pair threshold connection, locality and the horizon control.
The identities here use standard algebra; research novelty has not been assessed.
