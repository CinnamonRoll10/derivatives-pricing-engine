#pragma once

#include <string>

#include "dpe/engine.hpp"

namespace dpe {

/// Theta-scheme selector for the Black-Scholes PDE.
///
///   Explicit       theta = 0    1st order in time, CONDITIONALLY stable
///   Implicit       theta = 1    1st order in time, unconditionally stable
///   CrankNicolson  theta = 1/2  2nd order in time, unconditionally stable
///
/// The explicit scheme is included precisely because it blows up when the
/// stability limit is violated -- the convergence study demonstrates that.
enum class FDScheme { Explicit, Implicit, CrankNicolson };

std::string to_string(FDScheme s);

/// Finite-difference solver for
///
///   dV/dt + (1/2) vol^2 S^2 V_SS + (r - q) S V_S - r V = 0
///
/// marched backwards from the payoff on a uniform grid in S. S-space (rather
/// than log-space) is used so a barrier can be placed exactly on a node.
class FiniteDifferenceEngine : public PricingEngine {
public:
    FiniteDifferenceEngine(int spot_steps = 400, int time_steps = 400,
                           FDScheme scheme = FDScheme::CrankNicolson,
                           bool rannacher = true)
        : spot_steps_(spot_steps),
          time_steps_(time_steps),
          scheme_(scheme),
          rannacher_(rannacher) {}

    PricingResult price(const Instrument& instrument, const MarketData& market) const override;
    std::string name() const override;
    bool supports(const Instrument& instrument) const override;

    void set_spot_steps(int n) { spot_steps_ = n; }
    void set_time_steps(int n) { time_steps_ = n; }
    void set_scheme(FDScheme s) { scheme_ = s; }

    /// Rannacher start-up: replace the first two Crank-Nicolson steps with four
    /// half-size fully-implicit ones. CN is only weakly damping, so the kink in
    /// the payoff at the strike (and the jump in a digital) excites oscillations
    /// that CN does not kill; the implicit steps damp them before CN takes over.
    /// Without this, CN gamma near the strike is visibly wrong.
    void set_rannacher(bool on) { rannacher_ = on; }
    bool rannacher() const { return rannacher_; }

    /// Grid upper bound as a multiple of max(spot, strike). The far boundary
    /// condition is only asymptotically correct, so it must sit far enough out.
    void set_spot_max_multiple(double m) { spot_max_multiple_ = m; }

private:
    int spot_steps_;
    int time_steps_;
    FDScheme scheme_;
    bool rannacher_;
    double spot_max_multiple_{4.0};
};

}  // namespace dpe
