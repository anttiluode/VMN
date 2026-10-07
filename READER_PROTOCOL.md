# Lifted response and reader follow-up: protocol

Recorded 7 October 2026 before collecting the follow-up outcomes. Antti approved
the doubled-response derivation and the quadratic-reader comparison in VMN.
This is a bounded extension of the existing geometry experiment.

## Question and scope

Does a reader of `[q, |z|², Re(z²), Im(z²)]` retrieve nonlinear delayed-input
targets that an affine reader of `q` misses? The extra coordinates are computed
from the same current state. They do not add recurrent state or past-input
access. Separately, verify the exact complex tangent of the full smoothed,
nonlinear model; distinguish that identity from special-case onset formulas.

The proposal was motivated by the earlier geometry results. Those results stay
unchanged. Fresh input trajectories provide the follow-up evaluation; the four
physical layouts are reused, so this is not a new-layout generalization test.

## Frozen models and fresh trajectories

- Layout seeds: 20261007, 20261008, 20261009, 20261010.
- Five models per seed: fixed/moving geometry × shear 0/2, plus the fixed linear
  oscillator control. All retain 16 real recurrent state coordinates.
- Each operating offset is frozen to its earlier validation-selected value in
  `results/geometry_checks.json` at VMN commit
  `72f94291f266cf793a7b780ef09ef38f4759bc23`. New targets choose no model offset,
  geometry, shear, input weights or feature map.
- Input RNG seed = layout seed + 60000. Draw train, validation and test inputs
  sequentially from one RNG, independently IID uniform on [-1, 1]. These seeds
  differ from the previous memory, preparation and numerical-check seeds.
- Three independent rest-initialized trajectories per model; washout 256
  samples each; retain 1200 training, 400 validation and 800 test samples.
- RK4, input held for dt = 0.25, two substeps, unchanged from the earlier model.
- Targets: delayed Legendre orders 1/2/3, lags 1/2/4/8/16/32, unchanged alignment.

## Readers and costs

Use the exact same trajectory arrays for both readers of a model.

- Raw reader: 16 features plus an intercept, 306 coefficients for 18 targets.
- Quadratic reader: those 16 features plus 8 each of `|z|²`, `Re(z²)` and
  `Im(z²)`: 40 features plus an intercept, 738 coefficients for 18 targets.
- Every model, including the linear control, receives the same quadratic map.
  The expanded versus raw reader comparison intentionally changes readout
  capacity; it is not an equal-parameter comparison between those two readers.
- This feature map has no cross-unit products. It is not a full degree-two
  polynomial reader and need not decode every distinction present in the state.
- Feature mean/scale come only from training data. Reuse the existing ridge
  implementation and penalties 1e-6/1e-4/1e-2/1. Select each reader's penalty
  using validation MSE divided by training target variance, over all 18 targets.
- Fit weights on training data only; no train/validation refit. Test scores
  select nothing. Report signed test R², per target and averaged by order,
  including negative values.

## Primary gate and secondary comparisons

The primary reader gate is fixed geometry, shear 0: quadratic minus raw mean
order-2 test R² must improve by at least 0.02, win in at least 3 of 4 seeds, and
the quadratic reader's mean order-2 test R² must be positive. This measures the
specific even-target limitation identified before the follow-up.

Report the same differences for the other four models, and all models' order-1,
order-2, order-3 and combined order-2/3 scores. Do not substitute one of those
results for the primary gate.

Secondary geometry, shear and nonlinear-versus-linear-control comparisons use
the quadratic reader at each model's frozen operating point. A claimed advantage
requires a mean nonlinear R² difference >= 0.02 and wins on >= 3/4 seeds. These
comparisons are conditional on the frozen operating points, not optimized
architecture rankings. None is a long-memory or continual-learning benchmark.

## Negative controls and numerical checks

- For every model and reader, permute training and validation target rows
  independently using RNG seed = layout seed + 70000. Use the same permutations
  across variants/readers of a seed. Fit/select as above, score against original
  test targets. Report all scores; any claimed signal must be distinguished from
  this destroyed state-target relationship.
- Verify that quadratic features preserve the raw state, have even parity in
  their added coordinates, and can express a known quadratic target on an
  independent synthetic train/test fixture where the raw affine reader cannot.
- Verify analytic complex tangent against the independent real Jacobian,
  a normalized unitary lift, matching eigenvalues and singular values, and
  preservation of conjugacy on physical perturbations, across 48 Jacobians.
- Check the equal-frequency pure-conjugate squared-matrix identity, and the
  zero weak-coupling growth shift for positive-circulation unsmoothed kernels.
  Do not transfer that special formula to the smoothed, heterogeneous model.
- Check that changing only test data changes neither reader parameters nor
  validation-selected penalties. Check split/lag alignment independently.

Record source hashes, the frozen-model receipt hash, exact selected offsets,
derived RNG seeds, integration/fit time, budgets and package versions. A failed
research gate is reported as an outcome, not a numerical-check failure.
