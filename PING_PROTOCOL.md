# Ping-queryable phase memory: protocol before outcomes

7 October 2026. Antti authorized this follow-up in **VMN**. VMNClaude is a
read-only reference. This extends the existing numerical research flow; it
does not implement a fluid solver or establish a hardware advantage.

## Question and model

Can a single noisy oscillator bank answer a goal-vector query from its measured
ports, without the noiseless counterfactual used in VMNClaude Gate 4? How much
does it disturb memory, and what survives when output access is reduced?

The task is velocity integration after a starting landmark. Use isochronous
Stuart–Landau units with growth 0.5, cubic coefficient 1 and zero carrier
frequency in the resting frame. Preferred directions are seeded random;
spatial frequencies cycle through 1, 2 and 4 cycles per box width. Test 8 and
32 complex units (16 and 64 real state coordinates). This is a new input mode
of the oscillator primitive, not a rerun of the earlier additive-input gate.

Each of 200 input steps lasts one time unit, split into four 0.25 steps.
Use the exact deterministic uncoupled radial/rotation flow, then additive
isotropic complex Gaussian noise of magnitude 0.05 per square-root time;
each real component has variance 0.05² dt / 2. The exact flow removes numerical
phase-integration error; the stochastic splitting is still a discretization.
Smooth reflected random walks stay in the unit box. Positions begin uniformly
in [0.1,0.9]². The simulator knows true position; readers receive only their
specified measurements and the query input.

A goal is true endpoint plus an independent displacement uniform in the disk
of radius 0.35. The pulse is `a sqrt(0.5) exp(i K goal)`. Listen while stopped
for four time units, observing at 1, 2 and 4. Pulse amplitudes are 0.05 and 0.3;
the predeclared primary amplitude is 0.3. Evolve the actual kicked state.

## Observation contracts and comparisons

- **Local output ports:** one complex voltage output per unit, sampled before
  the pulse and at the three listening times. Features are measured phase
  changes relative to that same bank's pre-pulse output. There is no unpinged
  copy in this feature computation. These ports expose every unit; this is
  not a demonstration of one-port access to hidden units.
- **Four real output channels:** two fixed complex weighted sums of unit
  outputs, with seeded unit-norm weights. Features are the pre-pulse sums,
  subsequent changes and the known 2D goal. Compare active and passive
  observations with identical ports, sample times, feature counts and readers.
- **Goal-only control:** the 2D goal is supplied to the same reader without
  observations. This controls information due to goals being near endpoints.
- **Privileged twin diagnostic:** unit phase differences between kicked and
  un-kicked copies sharing future noise, as in Claude Gate 4. Label it as a
  counterfactual; never substitute it for the physical listener.
- **Direct phase reference:** sin/cos of goal phase minus current noisy phase.
  Also fit an absolute-position reader from pre-pulse phases for damage checks.
- **Digital integrator:** two real coordinates accumulate the same measured
  displacement. Independent oscillator noise is absent in this arithmetic
  reference; shared sensor error is present. Record this asymmetry explicitly.
- **Vortex during the query:** amplitude 0.3 with strength 0.1 times a unit-
  spectral-norm, fixed smoothed vortex tangent from VMN's kernel. Circulations
  are positive; smoothing is 0.15. Storage is uncoupled. This differs from
  Claude's mixed-sign, unsmoothed conjugate matrix. No retuning after outcomes.
- **Shuffled targets:** permute training and validation targets independently;
  retain true test targets. This measures accidental readout fitting.

Run two noise cases: independent internal noise, and the same internal noise
plus an episode-specific constant velocity bias with independent normal
components of standard deviation 0.0003. The latter coherently changes all
phases. No landmark correction is supplied. Also evaluate four successive
queries at amplitude 0.3, using the frozen position reader, against an un-kicked
copy at the same final time. Twins here are causal damage diagnostics only.

## Splits, readers and success thresholds

Four bank seeds: 20262007–20262010. Per bank/case: 1024 training, 256 validation,
512 test episodes. Generate splits independently with fixed offsets recorded
in the runner. No test outcomes select amplitude, coupling, model or reader.
Training data alone determine feature means and scales. Validation selects
ridge penalty from 0.01, 0.1, 1, 10, 100; kNN neighbour count from 5, 15, 40;
and ridge versus kNN. Store model sizes and validation choices. Readers are
trained separately for each bank, observation contract and condition.

Predeclared primary case: 32 units, independent noise, amplitude 0.3, no
coupling. Average Euclidean goal-vector error over independent test episodes.
The following thresholds describe engineering gates, not significance tests:

1. **Single-bank query:** mean error no greater than half the zero-displacement
   answer's error, with improvement on at least 3/4 banks.
2. **Near direct access:** mean error no greater than 1.25 times the direct-phase
   reference, with that ratio achieved on at least 3/4 banks.
3. **Limited-port query:** active four-channel error at least 10% lower than
   passive and goal-only errors, with both wins on at least 3/4 banks, and mean
   active error no greater than half the zero-answer error.
4. **Read disturbance:** one query adds no more than 0.005 mean position error
   compared with the un-kicked bank after the same noisy listening interval.
5. **Vortex value:** query-time vortex coupling improves local-port error by
   at least 10%, wins on 3/4 banks, and adds no more than 0.005 extra position
   error relative to the matched uncoupled pulse.
6. **Redundancy:** direct-phase reference error with 32 units is at least 25%
   below that with 8 under independent noise. Report sensor-bias results
   separately; do not infer that population redundancy corrects coherent error.

Report errors and paired differences for every bank, all gates, repeated-query
damage, waveform sample counts, state and decoder storage, query latency and
simulation wall time. A successful interface gate alone does not establish
novel neuroscience, novelty over grid-code navigation, faster computation,
lower energy use or a reason to replace a digital register.

## Implementation and verification

1. `research/ping_model.py`: exact isolated flow, noisy path integration,
   measured local/aggregate ports, actual query evolution and labelled twins.
   Check exact phase integration, analytic kick phase, noise cancellation
   boundaries, coherent-error indistinguishability and kernel integration.
2. `research/ping_readout.py`: train-only normalization, validation-only reader
   selection and measured-port feature extraction. Check that zero-pulse
   single-bank observations retain future noise while twin differences cancel.
3. `research/check_ping.py`: independent episode splits, paired conditions,
   frozen-reader damage checks, six gates, hashes, receipt and scientific plot.
   Add a small deterministic integration test; run the full existing suite.
4. Write `PING.md` from the receipt, link it in README, independently review
   equations/access/splits/claims, then publish verified changes to VMN.

This protocol is committed before collecting task outcomes. Earlier receipts
and their source files are preserved.
