# Derivatives Pricing Engine

A C++17 options pricing library that values European, American and barrier options
through four independent numerical methods, and validates every one of them against
the others.

The point of the project is not that it implements Black-Scholes. It is that each
method is checked against an independent method, the discretisation error of each is
measured rather than assumed, and the cases where the numerics genuinely break are
identified and reported rather than hidden.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
./build/dpe_tests            # 126 checks
./build/convergence_study    # the full numerical validation report
```

Requires only a C++17 compiler and CMake. No external dependencies.

---

## Architecture

Payoff, exercise style, stochastic process and pricing method are four **orthogonal**
axes that get composed, never inherited into one another:

```
Payoff              Exercise         PricingEngine
  Vanilla             European         AnalyticBlackScholes
  Digital             American         Binomial (CRR)
  Barrier                              MonteCarlo  -- GBM path generator
                                       FiniteDifference -- explicit/implicit/CN
        \               /                      |
         \             /                       |
          Instrument  (payoff + exercise + maturity)  ---> price()
```

An `Instrument` holds a payoff and an exercise style. Any engine can price any
instrument it declares support for. This is what makes the cross-method comparison a
fair test: all four engines receive the *same* `Instrument` object, not four
near-duplicate re-specifications of the trade.

It also means adding a payoff costs one class and touches no solver. `BarrierPayoff`
implements `from_path()` and every Monte Carlo engine prices it without modification.

Engines advertise what they can handle via `supports()`, so an unsupported
combination throws instead of silently returning a wrong number — the analytic engine
refuses American puts, and the recombining lattice refuses path-dependent payoffs.

---

## Results

All numbers below are produced by `./build/convergence_study` and are reproducible.

### Cross-method agreement

European call, S=100, K=100, r=5%, q=2%, vol=20%, T=1. Benchmark is the closed form.

| Method | Price | Diff (bps) | Time |
|---|---|---|---|
| Closed-form Black-Scholes | 9.22700551 | — | — |
| Binomial CRR, 5000 steps | 9.22661672 | 0.421 | 5 ms |
| Crank-Nicolson 800×800 | 9.22640439 | 0.651 | 12 ms |
| Monte Carlo, 2M paths, antithetic+control | 9.22698274 | 0.025 | 77 ms |

**Four independent methods agree to within 0.65 bps.**

### Monte Carlo: convergence and variance reduction

Standard error falls as O(1/√N) — quadrupling the paths halves it:

| Paths | Std error | SE ratio |
|---|---|---|
| 10,000 | 0.15012827 | — |
| 50,000 | 0.06588117 | 2.28 |
| 500,000 | 0.02079085 | 2.24 |
| 5,000,000 | 0.00658227 | 2.24 |

Variance reduction at a **fixed budget of 1,000,000 paths** (so equal cost — the
antithetic estimator runs 500k pairs, not 1M pairs):

| Technique | Std error | Reduction |
|---|---|---|
| plain | 0.01472484 | — |
| antithetic | 0.01041618 | **29.3%** |
| control variate | 0.00561600 | **61.9%** |
| antithetic + control | 0.00275985 | **81.3%** |

The control variate is the discounted terminal spot, whose expectation
`E[e^{-rT} S_T] = S_0 e^{-qT}` is known exactly; the regression coefficient is
estimated from the same sample.

### Finite difference: measured convergence order

Halving both `dS` and `dt` each row. Error ratio of 2 means first order, 4 means
second order:

| Grid | Implicit error | ratio | Crank-Nicolson error | ratio |
|---|---|---|---|---|
| 100×100 | 5.046e-02 | 3.42 | 3.990e-02 | 3.78 |
| 200×200 | 1.516e-02 | 3.33 | 9.918e-03 | **4.02** |
| 400×400 | 5.096e-03 | 2.98 | 2.476e-03 | **4.01** |
| 800×800 | 1.929e-03 | 2.64 | 6.188e-04 | **4.00** |

Crank-Nicolson holds a ratio of 4.00 — clean second-order accuracy. The implicit
scheme decays toward 2 as the first-order time error comes to dominate.

### Why Rannacher start-up is not optional

Crank-Nicolson is only *weakly* damping: its amplification factor tends to −1 for
high-frequency modes rather than to 0. A discontinuous payoff (here a digital)
excites exactly those modes, and refining the **spatial** grid at a fixed coarse time
grid injects more of them. The result is not a small error — it is a wrong answer:

| Spot pts | Time pts | CN price | +Rannacher | CN gamma | +Rannacher |
|---|---|---|---|---|---|
| 200 | 10 | 0.49041765 | 0.51345323 | 0.020117 | −0.000279 |
| 400 | 10 | 0.36295400 | 0.52286227 | 0.487960 | −0.000304 |
| 800 | 10 | 0.21764483 | 0.52756314 | 3.267574 | −0.000317 |
| 1600 | 10 | 0.12722918 | 0.52991373 | **15.142839** | −0.000323 |
| 1600 | 25 | 0.79657766 | 0.52996954 | **−11.768493** | −0.000323 |

True price 0.53232, true gamma ≈ −0.00033. Raw CN gamma has the **wrong sign** and is
~46,000× too large; the price is off by a factor of four. Replacing the first two CN
steps with four half-size fully-implicit steps damps the oscillation completely.

Note the counter-intuitive part: refining the grid makes raw CN *worse*.

### American early exercise

American put, S=100, K=110, r=5%, vol=30%, T=1:

| | Price |
|---|---|
| European put, closed form | 14.65531432 |
| European put, binomial tree | 14.65550693 |
| American put, binomial tree (5000 steps) | 15.61795032 |
| American put, Crank-Nicolson + PSOR | 15.61721083 |
| **Early-exercise premium** | **0.96263601** |

Tree and PSOR disagree by **0.47 bps** — two structurally unrelated methods agreeing
on the premium.

The PDE side solves a linear complementarity problem via projected SOR. Solving the
linear system and then clipping to the payoff afterwards does *not* work: it violates
the complementarity condition at the free boundary. The constraint has to be enforced
inside the iteration.

### Barrier options and discrete monitoring

Down-and-out call, S=100, K=100, B=90, T=1. The continuous closed form comes from the
method of images; Crank-Nicolson on a domain truncated at the barrier agrees to
**0.22 bps**.

Monte Carlo monitors discretely, which biases the price upward — fewer observation
dates mean fewer chances to breach. The bias is validated against the
**Broadie–Glasserman–Kou** continuity correction, which predicts a discretely
monitored barrier behaves like a continuous one at a barrier shifted by
`exp(−0.5826 · vol · √(T/m))`:

| Monitor dates | MC price | MC std err | BGK predicted | Difference | Within 2 s.e.? |
|---|---|---|---|---|---|
| 12 | 9.59125866 | 0.01124959 | 9.58023149 | +0.01102717 | yes |
| 50 | 9.18699224 | 0.01150727 | 9.18224086 | +0.00475139 | yes |
| 250 | 8.91053641 | 0.01165235 | 8.91485072 | −0.00431431 | yes |
| 1000 | 8.78847775 | 0.01172683 | 8.79406176 | −0.00558401 | yes |
| 4000 | 8.74692663 | 0.01174278 | 8.73076446 | +0.01616218 | yes |

Matching theory at every monitoring frequency shows the gap is the *predicted*
discretisation bias, not an implementation error.

### Greeks, three ways

| Greek | Analytic | FD grid | MC (bump + common random numbers) |
|---|---|---|---|
| delta | 0.50476274 | 0.50475270 | 0.50502378 |
| gamma | 0.01828446 | 0.01828455 | 0.01835461 |
| vega | 34.28336465 | n/a | 34.32748350 |

Delta and gamma come free off the finite-difference grid. Theta then follows exactly
from the PDE itself rather than needing another solve:

```
dV/dt = rV − (r−q)·S·V_S − ½·vol²·S²·V_SS
```

Monte Carlo Greeks use **common random numbers** — the same seed, and therefore the
same normal draws, for the base and bumped runs. Without that, gamma (a second
difference of a noisy estimator) is pure noise.

### Implied volatility and the surface

Newton-Raphson on vega, with Brent's method as a bracketing fallback.

The fallback is not defensive padding. **Vega collapses toward zero in the deep wings
and at short expiry**, which is precisely where a real option chain has most of its
strikes. There the Newton step `f/f'` explodes. Out of a 28-quote synthetic chain, 7
quotes required the fallback.

More importantly, some of those quotes are not invertible at all. At S=100, K=60,
T=0.8, vol=8%, vega is `2.0e-11` — moving volatility by a **full percentage point**
changes the price by 1.3e-11, far below any sane solver tolerance. A root exists and
reprices correctly, but it carries no information about volatility.

So the solver reports its own conditioning:

```cpp
struct ImpliedVolResult {
    double vol;
    double vega;              // conditioning of the inversion
    double vol_uncertainty;   // ~ tol / vega
    bool   well_conditioned;  // vol pinned to better than 1e-4
    bool   used_fallback;
};
```

Reconstructing a known smile from synthetic quotes:

```
max error over WELL-CONDITIONED points : 1.130e-06 vol points
max error over ALL converged points    : 2.345e-01 vol points
```

The second number is not a bug — it is the deep-wing points that carry no
information. `VolatilitySurface` excludes them from interpolation, because feeding
them to a downstream smile fit would corrupt it.

---

## Test suite

126 checks, no external framework. The ones that actually constrain the code:

- **Put-call parity** to 1e-12, and delta parity to `e^{-qT}`.
- **Analytic Greeks vs finite-difference bumps** — all five.
- **American call = European call when q=0.** The sharpest test of the early-exercise
  logic: it verifies the code does *not* fire early exercise when it shouldn't.
- **Explicit scheme rejects an unstable timestep** rather than returning garbage.
- **Unreachable barrier reproduces the vanilla** within Monte Carlo error.
- **Discrete monitoring overprices a knock-out**, and denser monitoring reduces the bias.
- **Implied vol round-trips the price** always, and round-trips the *volatility* only
  where the solver reports the inversion is well conditioned.

---

## What is deliberately not here

- **Stochastic volatility (Heston, SABR).** Calibrating one properly is a project in
  itself; a shallow version would add a keyword and nothing else.
- **Knock-in barriers** on the PDE side — they need the in-out parity relation.
- **Longstaff-Schwartz** for American Monte Carlo. Tree and PSOR already give two
  independent American methods.
- **A production vol interpolator.** The surface uses linear-in-strike interpolation;
  a real one fits a parametric form (SVI) to enforce no-arbitrage in both directions.

## Layout

```
include/dpe/
  types.hpp payoff.hpp instrument.hpp engine.hpp implied_vol.hpp
  engines/  analytic_bs.hpp binomial.hpp monte_carlo.hpp finite_difference.hpp
  math/     normal.hpp rootfind.hpp tridiagonal.hpp
src/        one .cpp per engine
tests/      test_main.cpp
apps/       convergence_study.cpp
```
