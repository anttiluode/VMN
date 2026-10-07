# VMN: nonlinear state as a controller of response matrices

**Research note — 7 October 2026**

This note develops the vortex–matrix–neuron connection requested for VMN. The objective is a mathematical mechanism and falsifiable experiments. The calculations use standard dynamical-systems and matrix algebra; no new-theorem priority, biological equivalence or ML advantage is claimed.

The strongest finding is a distinction: **a response update can have high matrix rank while belonging to a small, nonlinear family of operators**. Conversely, individually low effective-rank updates can fail to share a useful codebook across locations.

## 1. The shared object: a history-conditioned response

Take a smooth controlled system on a collision-free region, with fixed additive input and linear observation maps:

$$
\dot x=f(x)+Bu(t),\qquad y=Cx.
$$

Two preparation histories lead to states $x_A(0)$ and $x_0(0)$. After preparation, give both the same background input. At time zero, add a small state pulse $Bp$ to each. The history-specific probe response is the difference between its probed and unprobed future; it is not the difference between the two raw futures.

The tangent propagator satisfies

$$
\partial_t\Phi_H(t,s)=J_H(t)\Phi_H(t,s),\qquad
J_H(t)=D f(x_H(t)),\qquad \Phi_H(s,s)=I.
$$

To first order in the pulse,

$$
\delta y_H(t)=C\Phi_H(t,0)Bp+O(\|p\|^2).
$$

The relevant response operator is $R_H(t)=C\Phi_H(t,0)B$. Stack these matrices over observation times to obtain the finite-trajectory response operator used in the numerical probe. Define $\Delta R=R_A-R_0$.

An exact variation identity is

$$
\Phi_A(T,0)-\Phi_0(T,0)
=\int_0^T\Phi_A(T,s)[J_A(s)-J_0(s)]\Phi_0(s,0)\,ds.
$$

It follows by differentiating $\Phi_A(T,s)\Phi_0(s,0)$ with respect to $s$. Each contribution transports a probe to a changed local response law and transports the resulting difference onward. Low rank of individual contributions does not bound the rank of their integral by the same number.

This connects directly to [Kompressori's response calculation](https://github.com/anttiluode/Kompressori/blob/main/Kompressori_exact_response.md), but neither its local stencil nor its empirical spectrum is imported into VMN.

## 2. What nonlinearity contributes

If $f(x)=Lx$, then $J_H(t)=L$ and $R_H(t)=Ce^{Lt}B$ for every preparation history. An earlier pulse can leave a different baseline state, but it cannot change this operator. This statement assumes fixed coefficients, additive inputs and linear observations. A nonlinear observation or state-dependent input map introduces another mechanism.

In a quadratic fluid-mode model,

$$
\dot x=Lx+Q(x,x)+Bu,
$$

with bilinear $Q$, differentiation gives

$$
J(x)=L+Q(x,\cdot)+Q(\cdot,x),
$$

and the instantaneous history-induced change is exactly

$$
J(x+d)-J(x)=Q(d,\cdot)+Q(\cdot,d).
$$

The state now selects the matrix through which the next small input acts. A one-dimensional change $d$ need not produce a rank-one matrix: bilinear coupling can distribute that change across many input directions.

One finite nonlinear interaction statistic is

$$
I(a,p)=y(a,p)-y(a,0)-y(0,p)+y(0,0).
$$

Here $a$ labels an earlier event and $p$ a later probe. A fixed linear system gives $I=0$; nonlinear systems can give $I\ne0$. The leading interaction depends on the expansion point. Quadratic coupling can produce an $ap$ term, while the cubic Hopf model below, expanded about rest, first changes the probe response at order $a^2p$.

## 3. Point vortices: an explicit matrix from geometry

Let $q_i\in\mathbb R^2$ be vortex positions, $\Gamma_i\ne0$ fixed circulations, and

$$
\mathcal R=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\qquad r_{ij}=q_i-q_j.
$$

The unbounded-plane point-vortex equations are

$$
\dot q_i=\sum_{j\ne i}\frac{\Gamma_j}{2\pi}
\frac{\mathcal Rr_{ij}}{\|r_{ij}\|^2}.
$$

These are the standard inviscid equations reviewed by [Aref (2007)](https://backend.orbit.dtu.dk/ws/portalfiles/portal/4793688/Aref.pdf). They approximate a different physical setting from a viscous cylinder wake. The present experiment uses no core regularization, damping, boundaries or vortex creation.

For $r\ne0$, differentiate the interaction kernel:

$$
M(r)=D_r\left(\frac{\mathcal Rr}{\|r\|^2}\right)
=\frac{\mathcal R}{\|r\|^2}
-\frac{2(\mathcal Rr)r^\top}{\|r\|^4}.
$$

Then the Jacobian's $2\times2$ blocks are

$$
J_{ii}=\sum_{j\ne i}\frac{\Gamma_j}{2\pi}M(r_{ij}),\qquad
J_{ij}=-\frac{\Gamma_j}{2\pi}M(r_{ij}).
$$

Thus a vortex configuration literally determines a matrix of instantaneous perturbation interactions. The numerical tangent propagation uses this analytic expression; finite differences only verify it independently.

### Proposition: one moved vortex generically gives rank $2N-2$

Move vortex $k$ by a nonzero displacement $d$, with every other position unchanged. Assume neither configuration has collisions, and no other vortex is exactly the midpoint between the old and new positions of $k$.

For each unchanged vortex $j$, only its interaction with $k$ changes. Applied to a perturbation vector $v$, its block row becomes

$$
(\Delta Jv)_j=\frac{\Gamma_k}{2\pi}\Delta M_j(v_j-v_k),
$$

where $\Delta M_j=M(q_j-q_k-d)-M(q_j-q_k)$.

Writing $r=(x,y)$ gives

$$
M(r)=\frac{1}{(x^2+y^2)^2}
\begin{pmatrix}2xy&y^2-x^2\\y^2-x^2&-2xy\end{pmatrix}.
$$

Every nonzero difference of two such matrices is symmetric and traceless, hence has determinant $-(a^2+b^2)<0$ and is invertible. Moreover, $M(r')=M(r)$ holds exactly when $r'=r$ or $r'=-r$: its norm fixes $\|r\|$, and its angle fixes the direction modulo $\pi$. The displacement and midpoint assumptions exclude these possibilities.

Consequently $\Delta Jv=0$ forces $v_j=v_k$ for every unchanged vortex. Uniform translations are already in the kernel of both Jacobians, so the kernel is exactly two-dimensional. Therefore

$$
\boxed{\operatorname{rank}(\Delta J)=2N-2.}
$$

This is an elementary consequence of the stated kernel, rather than a literature-novelty claim. It establishes why Kompressori's five-row local-write bound cannot be reused for globally coupled point vortices.

### Exact rank, effective rank and short descriptions

For a small displacement and a distant vortex, $\|\Delta M\|=O(\|d\|/\|r\|^3)$. Thus nearby interactions can dominate the update's energy even when every non-translation direction changes. High exact rank and a modest energy rank are compatible.

The update is also generated by a baseline configuration, an index $k$ and a two-coordinate displacement. That is a compact *recipe*, provided the shared baseline and decoder are available. Replaying or evaluating the dynamics still costs computation; a short recipe is not automatically cheap retrieval.

### Why this model has no cylinder-shedding knob

The Jacobian has zero trace on collision-free trajectories. Liouville's formula gives $\det\Phi(t,0)=1$. The unforced model preserves phase volume rather than attracting an open set of states to a stable limit cycle. Hamiltonian instabilities can exist, but they are not the dissipative shedding transition under discussion. Adding common additive forcing does not itself create contraction of this full-state tangent flow.

Point vortices therefore test response geometry and nonlinear interaction. A Reynolds-number sweep or a fading-memory claim needs a different model, or an explicitly introduced dissipation/reset mechanism.

## 4. Neurons: activity changes the effective gain matrix

For a continuous-time rate network,

$$
\tau\dot v=-v+W\sigma(v)+Bu,
$$

where $\sigma$ acts elementwise,

$$
J(v)=\tau^{-1}[-I+W\operatorname{diag}(\sigma'(v))].
$$

The weights stay fixed while activity changes the gains on their columns. If only $k$ activation slopes differ, then

$$
\operatorname{rank}(\Delta J)\le\min(k,\operatorname{rank}W).
$$

A single altered gain is at most rank one for this instantaneous model. Subsequent dynamics can spread the activity change, and finite-time updates can accumulate more directions.

A representative excitable model, FitzHugh–Nagumo, has

$$
\dot v=v-v^3/3-w+I,\qquad
\dot w=\epsilon(v+a-bw),
$$

and

$$
J(v)=\begin{pmatrix}1-v^2&-1\\\epsilon&-\epsilon b\end{pmatrix}.
$$

Its instantaneous update is rank at most one when voltage changes. At an equilibrium, a candidate Hopf crossing satisfies $v_*^2=1-\epsilon b$ and $\det J=\epsilon(1-\epsilon b^2)>0$, with the additional nondegeneracy and crossing conditions required for an actual Hopf bifurcation. The type of the bifurcation is a further question; it is not assumed supercritical here. [Izhikevich's neural bifurcation analysis](https://izhikevich.org/publications/sirev.pdf) provides the biological-model context.

The connection is shared state-dependent sensitivity and, for suitable systems, oscillatory bifurcations. It does not identify neurons with physical vortices, nor require every neuron to oscillate.

## 5. Oscillation: amplitude, phase and a nonlinear response

Use a controlled supercritical Hopf normal form, without claiming it has been fitted to a cylinder wake or a neuron:

$$
\dot q=(\mu I+\omega\mathcal R)q-a\|q\|^2q+Bu,\qquad a>0.
$$

Without input, polar coordinates give

$$
\dot r=\mu r-ar^3,\qquad \dot\theta=\omega.
$$

For $\mu<0$, rest is a stable focus. For $\mu>0$, the stable cycle has radius $r_*=\sqrt{\mu/a}$. At $\mu=0$, a finite pulse decays algebraically:

$$
r(t)=\frac{r(0)}{\sqrt{1+2ar(0)^2t}}.
$$

The instantaneous matrix is

$$
J(q)=\mu I+\omega\mathcal R-a[\|q\|^2I+2qq^\top].
$$

The rotational term by itself is linear and history-independent. The nonlinear terms make the current amplitude and orientation change the next response. Locally, the nonlinear radial damping contribution is $3ar^2$, while the tangential contribution is $ar^2$.

### Amplitude can write a phase history

Including a nonlinear frequency coefficient $b$ gives the complex normal form

$$
\dot z=(\mu+i\omega)z-(a+ib)|z|^2z,
\qquad \dot\theta=\omega-br^2.
$$

A past amplitude excursion now changes the accumulated phase by $-b\int r^2dt$. For the definitions $\alpha_T,D_T$ in section 6, put

$$
g_T=e^{\mu T}/\sqrt{D_T},\qquad
\vartheta_T=\omega T-\frac{b}{2a}\log D_T.
$$

The exact flow and tangent are

$$
q(T)=g_T\operatorname{Rot}(\vartheta_T)q,
$$

$$
\Phi_q(T,0)=g_T\operatorname{Rot}(\vartheta_T)
\left[I-\frac{\alpha_T}{D_T}
\left(q+\frac{b}{a}\mathcal Rq\right)q^\top\right].
$$

This is a concrete mechanism linking an earlier pulse, oscillatory phase and the next response matrix. It includes radial-to-phase shear as well as decay. The probe verifies these formulas at $b=0.7$ for three Hopf parameters. The rank counterexample and vortex scan use the simpler $b=0$ model; no result from that scan is attributed to this shear.

On the stable cycle, for an initial radial unit vector $e_r$,

$$
\Phi(T,0)=\operatorname{Rot}(\omega T)
\left[I+(e^{-2\mu T}-1)e_re_r^\top\right].
$$

The radial Floquet exponent is $-2\mu$ and the phase exponent is zero. Phase can retain a history while radial errors decay. An unforced phase memory also prevents full synchronization across arbitrary initial phases; a driven reservoir must measure conditional stability rather than assume it.

The vortex's "extra step" is therefore a dynamical coordinate and another interval of computation. Its computational contribution depends on how the state selects subsequent operators. For frozen matrices over two short intervals,

$$
e^{J_2h}e^{J_1h}-e^{J_1h}e^{J_2h}
=h^2(J_2J_1-J_1J_2)+O(h^3).
$$

Order can matter. Matrix noncommutativity alone is not the new mechanism; the important mechanism is nonlinear history selecting the matrices. Actual nonlinear trajectories must evolve their states rather than use the frozen approximation indefinitely.

## 6. Counterexample: Hopf onset does not force low-rank updates

Append $m$ stable probe directions $w$:

$$
\dot q=(\mu I+\omega\mathcal R)q-a\|q\|^2q,\qquad
\dot w=-(\gamma+\kappa\|q\|^2)w,
$$

with $\gamma,\kappa>0$. At rest, only the two $q$ eigenvalues approach the imaginary axis as $\mu\to0$; the other $m$ remain at $-\gamma$.

Compare states $(q,0)$ and $(0,0)$. Their instantaneous matrix difference is

$$
\Delta J=
\begin{pmatrix}
-a[r^2I+2qq^\top]&0\\
0&-\kappa r^2I_m
\end{pmatrix}.
$$

For any nonzero $q$, this has rank $m+2$, however close $\mu$ is to zero. With $a=\kappa=1$, its singular values are $3r^2$ and $r^2$ repeated $m+1$ times. Their normalized spectrum is independent of $\mu$ and amplitude.

For $m=32$, 95% of the squared singular-value sum requires **32 of 34 directions**. Thus one two-dimensional Hopf mode does not guarantee a low-rank response update of the whole system.

The finite-time version also retains all stable probe directions. Define

$$
\alpha_T=\begin{cases}a(e^{2\mu T}-1)/\mu,&\mu\ne0,\\2aT,&\mu=0,\end{cases}
\qquad D_T=1+\alpha_T r^2.
$$

The exact Hopf flow and tangent from the state at probe time are

$$
q(T)=\frac{e^{\mu T}}{\sqrt{D_T}}\operatorname{Rot}(\omega T)q,
$$

$$
\Phi_q(T,0)=\frac{e^{\mu T}}{\sqrt{D_T}}\operatorname{Rot}(\omega T)
\left[I-\frac{\alpha_T}{D_T}qq^\top\right].
$$

Since $\int_0^T r(t)^2dt=(2a)^{-1}\log D_T$, the stable-block tangent is $e^{-\gamma T}D_T^{-\kappa/(2a)}I_m$. Its difference from the rest response has rank $m$. In the fixed experiment at $\mu=0$, the whole finite-response update needs **30 modes** for 95% energy.

This counterexample refutes a universal implication. It does not refute an empirical near-onset compactness hypothesis for a particular driven cylinder wake. Observable projection, input directions, spectral gaps and horizon can all change the answer.

## 7. A small nonlinear codebook despite full rank

Let $q=r(\cos\theta,\sin\theta)$ and define

$$
Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix},\qquad
X=\begin{pmatrix}0&1\\1&0\end{pmatrix}.
$$

Then the Hopf response update is

$$
J(q)-J(0)=-2ar^2I-ar^2\cos(2\theta)Z-ar^2\sin(2\theta)X.
$$

For the augmented counterexample, define three shared matrices

$$
G_0=\operatorname{diag}(-2aI_2,-\kappa I_m),\quad
G_1=\operatorname{diag}(-aZ,0_m),\quad
G_2=\operatorname{diag}(-aX,0_m).
$$

The full-rank update is reconstructed exactly as

$$
\boxed{\Delta J=c_0G_0+c_1G_1+c_2G_2,}
$$

where $(c_0,c_1,c_2)=(r^2,r^2\cos2\theta,r^2\sin2\theta)$. The coefficients obey $c_0^2=c_1^2+c_2^2$, $c_0\ge0$: a nonlinear two-dimensional cone of operator coordinates.

The three basis components remain fixed across states and Hopf parameters; the checked decoder reconstructs unseen coefficient values analytically. This is a compact codebook of potentially full-rank operators, distinct from truncating each operator's singular spectrum. The underlying $q$ state is already only two numbers, so this construction establishes structure rather than a storage saving over that state.

It also loses information: $q$ and $-q$ yield the same response matrices, although their baseline outputs are opposite. A sensitivity codebook alone is not a complete memory of content or phase.

## 8. What approaching onset predicts, conditionally

On the stable side, put $\mu=-\varepsilon$. Small amplitudes decay on the state timescale $1/\varepsilon$. Near rest, the cubic response update scales as $r^2$ and decays approximately on timescale $1/(2\varepsilon)$. At onset, a finite pulse instead gives algebraic decay.

The linearized oscillator's gain to a complex harmonic input of frequency $\nu$ is

$$
|\chi(\nu)|=\frac{|B|}{\sqrt{\varepsilon^2+(\nu-\omega)^2}}.
$$

Resonant sensitivity grows near onset, while a DC input at nonzero $\omega$ has finite limiting gain. Thus the input spectrum matters. Noise is also amplified: for two independent noise channels of amplitude $\sqrt{2D}$, the linear stationary mean squared radius is $2D/\varepsilon$. The nonlinear saturation eventually limits this divergence.

These calculations support a memory–sensitivity–noise tradeoff. They do not establish a universal optimum, better task accuracy, or minimal update rank at onset. A few slow coordinates can support a compact response codebook under suitable coupling and observation conditions; those conditions need measurement.

## 9. The published fluid evidence

[Goto, Nakajima and Notsu's preprint](https://arxiv.org/abs/2001.08502) evaluates a driven viscous cylinder wake with linear/nonlinear memory and NARMA tasks. Its computational peak is around $Re=40$, just before a shedding transition near $45$. Synchronization deteriorates above the transition. The final paper is [*Twin vortex computer in fluid flow*, NJP 23, 063051 (2021)](https://doi.org/10.1088/1367-2630/ac024d); [the authors' institution](https://www.kanazawa-u.ac.jp/latest-research/93394) confirms the just-before-shedding interpretation.

The study motivates measuring conditional stability and proximity to onset. It does not test response-update rank, compact operator codebooks or continual learning. Applying its performance result to those quantities is a new hypothesis.

## 10. Reproducible results and the transfer failure

Run `python research/check_math.py`. All setups use seed 20261007; the receipt records the dependency versions, source hash, all spectra and consistency-check thresholds.

The instantaneous scan uses jittered grids, nonzero circulations from 0.8 to 1.2, and 30 deterministic single-vortex displacements for each $N$. All 90 cases attain rank $2N-2$. For $N=4,8,16$, median 95%-energy ranks are respectively 4, 5 and 7.

The finite-time scan fixes eight vortices and all 16 orthonormal position probes. It prepares each of eight sites with each of four cardinal displacements of amplitude 0.07, waits 1.0 time unit, and observes every position at 30 times from 0.05 to 1.5 after the probe. Time-zero injection is excluded. These give complete position-input spectra for this finite-dimensional model, not spectra over arbitrary fluid perturbations or circulation changes.

The baseline needs 15 modes for 95% energy. Across 32 outcome-independent events, updates need a median of 5 modes, range 3–8. Their median norm is 0.736% of the baseline norm, range 0.459–1.006%. The small signal is independently checked: finite-time derivatives match central probe differences to relative error at most $8.28\times10^{-11}$ in the six tested comparisons.

For a separate transfer test, vectorize each entire response update. Fit a proper orthogonal decomposition (POD) dictionary using only the 20 events at sites 0–4. Eight dictionary components capture 95% of their aggregate squared norm. Test on the 12 events at sites 5–7, using optimal projection coefficients obtained from each **true** held-out update.

Median relative reconstruction error rises from **18.0%** on training events to **93.9%** on held-out sites, with held-out range 66.9–98.4%. This is an optimistic representation bound, not a trained prediction result. The dictionary component count is a rank across flattened *examples*, distinct from the rank of an individual response matrix. No cross-site predictor, moving dictionary or geometry-aware decoder is evaluated here.

Consequently this fixed dictionary generalizes poorly even though individual updates have modest effective rank. This fails one simple route to a transferable compact memory representation. It does not eliminate nonlinear or geometry-dependent dictionaries.

The 63 consistency checks also verify the point-vortex Jacobian, translation nullspace, phase-volume preservation, Hopf closed-form solutions against independent ODE integration (including amplitude-to-phase shear), the full augmented propagator, the 34-dimensional counterexample and its exact dictionary, and a rank-one neural gain change. The fixed linear negative control produces a relative response-operator change of only $1.80\times10^{-14}$.

Capturing 95% of spectral energy means a best rank-truncated Frobenius error of at most $\sqrt{0.05}\approx22.4\%$, rather than 95% response accuracy on every probe. Weak directions and relative errors on individual queries can still be poorly preserved.

## 11. What a VMN layer would need to earn

The next useful claim is: **a compact history state predicts or restores changes in response to new inputs, at a competitive total cost**.

A future driven experiment should hold out whole histories and event locations, measure both baseline outputs and probe contributions, and compare a fixed dictionary, a geometry-dependent dictionary and a state representation under the same storage budget. Swap or ablate the retained state to show that it causally changes the continuation. Include linear and oscillatory-linear controls so phase rotation alone cannot receive credit for nonlinear memory.

An onset sweep needs a dissipative, higher-dimensional driven model. Measure conditional stability, update magnitude, compression error, task accuracy and noise sensitivity together. A stable memory-bearing operating region may be more useful than exactly the bifurcation point. LinOSS/coRNN comparisons belong to a later trained-model benchmark, with matched state, parameter, compute and readout budgets.

Count the decoder: for an $n\times n$ update, separate rank-$r$ factors cost $2nr$ numbers, while a known dynamical state may cost only $n$. Shared matrices can amortize that cost, but reconstruction and integration still consume time. A small response derivative also omits baseline content and may not remain sufficient as the system evolves.

The present recommendation is therefore to pursue **low-dimensional control of response geometry**, with low-rank updates as one possible representation. The maths supports that common mechanism; a practical learned memory remains to be demonstrated.
