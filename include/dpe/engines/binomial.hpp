#pragma once

#include "dpe/engine.hpp"

namespace dpe {

/// Cox-Ross-Rubinstein binomial tree.
///
/// Chosen over Jarrow-Rudd because the CRR lattice recombines on a fixed grid
/// of spots (S0 * u^j * d^(n-j)), which makes the American early-exercise test
/// a simple comparison at each node. Converges to Black-Scholes as O(1/steps),
/// but oscillates: the price alternates above and below the true value as the
/// strike moves between adjacent terminal nodes. The convergence study measures
/// exactly that.
class BinomialEngine : public PricingEngine {
public:
    explicit BinomialEngine(int steps = 1000) : steps_(steps) {}

    PricingResult price(const Instrument& instrument, const MarketData& market) const override;
    std::string name() const override;
    bool supports(const Instrument& instrument) const override;

    int steps() const { return steps_; }
    void set_steps(int n) { steps_ = n; }

private:
    int steps_;
};

}  // namespace dpe
