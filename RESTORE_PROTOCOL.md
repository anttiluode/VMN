# Query, listen, then restore: protocol fixed before outcomes

7 October 2026. Follow-up to [PING.md](PING.md), authorized by Antti.
Only VMN is writable. VMNClaude is not part of the implementation.

## Question and scope

Can an observable, preprogrammed compensating pulse reduce the disturbance
of repeated goal queries while preserving useful answers from one noisy bank?
Left/right spatial reflection and electrical sign reversal are different
operations. No conclusion about biological theta sweeps follows from this gate.

Reuse the uncoupled, isochronous Stuart–Landau bank and measured-output readers
from PING. Preserve its sources and receipts. No fluid solver, learned recurrent
layer, neural-tissue model or hardware performance claim is added.

## Fixed comparisons

- 32 complex units; four new bank seeds 20263007–20263010.
- 1024 training / 256 validation / 512 test episodes; split offsets
  100000 / 200000 / 300000; 200 velocity steps; storage noise 0.05.
- Same unit-box trajectories and goals within radius 0.35 of the endpoint as
  PING. The endpoint is used for generation and scoring, never by the controller
  or listener. The two-dimensional goal and fixed mirror origin are supplied.
- Growth 0.5, cubic coefficient 1, shear 0, resting carrier 0; exact isolated
  deterministic flow with additive Gaussian noise at steps of 0.25.
- Query noise 0 and 0.05. The noisy condition is primary.
- Pulse amplitudes 0.05, 0.15, 0.3; 0.3 is primary and test results select nothing.
- Each pair: goal pulse at time 0; output samples at 1, 2, 4; second pulse at 4;
  evolve to time 8. Query features are collected before the second pulse.
- Repeat: both pulse phase patterns equal exp(i K g).
- Spatial mirror: second target (g_x, 1-g_y), reflected across the externally
  fixed line y=0.5. This control is a spatial reflection, not a claim to recreate
  the rat's ±30-degree sweeps around an unknown current location.
- Opposite: second pulse is the negative of the first electrical waveform.
- Passive: no pulse, same elapsed time and noise. Also retain a single-pulse
  diagnostic, labelled as having half the two-pulse energy.

## Two controller families and matched budgets

Fixed family: both pulse magnitudes are a sqrt(mu).

Radial-balance family: use each measured pre-query magnitude r_0 to predict
its no-input, noiseless magnitude r_T after T=4 with the known radial flow.
Set the second-to-first magnitude ratio to c=r_T/r_0. Normalize both pulse
patterns by sqrt((N+sum(c^2))/(2N)), so every active pair has the same total
sum of squared electrical pulse magnitudes, 2 N mu a^2. This is a pulse-energy
proxy, not joules. Repeat/mirror/opposite all receive this same magnitude
budget at a given starting state. No future noise or counterfactual is used
by the controller. Radial balance requires all local magnitudes and a model;
it cannot be credited as a four-channel controller.

All three patterns share the same first pulse within a controller family.
Consequently their first answer is identical by causality, not an independently
won accuracy result. Later answers must be measured after repeated pairs.

## Measured access, fitting and repetition

Fit the original ridge/kNN reader using training only for normalization and
validation only for reader/hyperparameter selection. Freeze it before testing.
The main listener receives the one bank's local pre-pulse outputs and samples
at 1, 2, 4. No noise-cancelled twin is a feature. Train a separate frozen
position reader on pre-query phases. A shared-noise unqueried copy is used
only to measure causal phase and position disturbance at the same time.

Run eight repeated pairs on each held-out endpoint and fixed goal. Reuse the
first-query reader without refitting. Score query error, frozen-reader position
error, added position error, and causal circular phase RMS after pairs 1, 2, 4, 8.
All conditions receive the same future noise draws. Energy includes both pulses;
time includes restoration. Record actual pulse-energy proxy per pair.

For the fixed primary family, retain goal-only, direct-phase and four-channel
active/passive controls with the original fitting contract. Compensation occurs
after collection, so it cannot improve the first answer from four channels.
Report channel counts, output samples, decoder storage, controller data and
latency. Include shuffled-target local control for the primary query.

## Predeclared mathematical and practical gates

1. Ideal opposite-pulse residual has log-log amplitude slope 1.8–2.2, using
   amplitudes 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 and relative phases 0.3, 0.7, 1.1.
   Use on-cycle states, no noise, T=4. Check the derived second-order coefficient.
2. Spatial ±30-degree counterexample: forward-tuned unit, on-cycle state,
   target distance 1 and wavevector (1,0). Mirrored goals give identical pulses
   and a first-order slope 0.95–1.05, not quadratic cancellation.
3. Fixed opposite pulses, noisy primary setting: after eight pairs, mean
   causal phase RMS <= half that of fixed repeated pulses, with wins on >=3/4 banks.
4. Radial-balanced opposite pulses: the same reduction relative to balanced
   repeated pulses, with wins on >=3/4 banks.
5. Useful repeated reads, separately for each opposite family: eighth-query
   error <= half the zero-answer error and <=1.25 times its first-query error;
   added mean position error <=0.005 after eight pairs; causal phase reduction
   as above; all per-bank comparisons jointly hold on >=3/4 banks.
6. Four-channel first query beats passive and goal-only by >=10% in mean,
   beats both jointly on >=3/4 banks, and error <=half the zero-answer error.

Failure is a result. Do not choose a different amplitude, horizon, controller
or reader after seeing test outcomes to rescue a gate. Report all fixed cells.

## Execution plan and verification

1. Commit this protocol before outcome collection. Baseline: existing 44 tests.
2. Test first: pulse energy, spatial reflection, analytic phase response,
   opposite-pair asymptotics, off-cycle amplitude correction, physical feature
   access, same-time causal control, observation timing and validation failures.
3. Implement `research/restore_model.py`, `research/check_restore.py` and focused
   model/experiment tests; reuse original PING helpers without editing them.
4. Run the frozen sweep. Publish `results/restore_checks.json`, a standalone SVG,
   and `RESTORE.md`, with source hashes, setup, gates, costs and limitations.
5. Run all unit tests, inspect the figure and receipts, independently review,
   then publish the verified tree to VMN after checking the current remote head.

This file is the design and run specification. The live execution ledger and
progress updates record completion; no additional approval stage is needed for
the comparisons Antti has already authorized.
