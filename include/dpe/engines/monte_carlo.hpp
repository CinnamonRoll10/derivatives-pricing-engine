#pragma once

#include <cstdint>

#include "dpe/engine.hpp"

namespace dpe {

/// Variance-reduction techniques the Monte Carlo engine can apply.
enum class VarianceReduction {
    None,
    Antithetic,       ///< pair each draw Z with -Z
    ControlVariate,   ///< use discounted terminal spot, whose mean is known exactly
    Both,
};

std::string to_string(VarianceReduction vr);

/// Risk-neutral Monte Carlo over geometric Brownian motion.
///
/// Path-independent payoffs are evaluated with the exact GBM solution in one
/// jump to maturity (no discretisation bias at all). Path-dependent payoffs are
/// stepped on a uniform grid, which introduces the usual discrete-monitoring
/// bias for barriers -- the convergence study quantifies it against the
/// continuously-monitored closed form.
class MonteCarloEngine : public PricingEngine {
public:
    MonteCarloEngine(std::size_t paths = 100000, VarianceReduction vr = VarianceReduction::None,
                     std::uint64_t seed = 42, std::size_t steps = 252)
        : paths_(paths), vr_(vr), seed_(seed), steps_(steps) {}

    PricingResult price(const Instrument& instrument, const MarketData& market) const override;
    std::string name() const override;
    bool supports(const Instrument&) const override { return true; }

    std::size_t paths() const { return paths_; }
    void set_paths(std::size_t n) { paths_ = n; }

    VarianceReduction variance_reduction() const { return vr_; }
    void set_variance_reduction(VarianceReduction vr) { vr_ = vr; }

    void set_seed(std::uint64_t s) { seed_ = s; }
    void set_steps(std::size_t n) { steps_ = n; }

    /// Bump-and-revalue Greeks using common random numbers (the same seed, and
    /// therefore the same normal draws, for the base and bumped runs). Without
    /// CRN the bump difference is swamped by Monte Carlo noise; with it, the
    /// noise largely cancels and the estimate is usable.
    void set_compute_greeks(bool on) { greeks_ = on; }
    bool compute_greeks() const { return greeks_; }

private:
    PricingResult price_terminal(const Instrument&, const MarketData&) const;
    PricingResult price_path(const Instrument&, const MarketData&) const;
    PricingResult price_raw(const Instrument&, const MarketData&) const;

    std::size_t paths_;
    VarianceReduction vr_;
    std::uint64_t seed_;
    std::size_t steps_;
    bool greeks_{false};
};

}  // namespace dpe
