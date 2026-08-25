#include "dpe/engines/binomial.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace dpe {

std::string BinomialEngine::name() const {
    return "Binomial CRR (" + std::to_string(steps_) + " steps)";
}

bool BinomialEngine::supports(const Instrument& instrument) const {
    // A recombining lattice cannot carry path history, so path-dependent
    // payoffs are out of scope. Everything else -- including American -- is fine.
    return !instrument.payoff().is_path_dependent();
}

PricingResult BinomialEngine::price(const Instrument& instrument, const MarketData& market) const {
    market.validate();
    if (steps_ < 1) throw std::invalid_argument("BinomialEngine: steps must be >= 1");
    if (!supports(instrument))
        throw std::invalid_argument("BinomialEngine: path-dependent payoffs unsupported");

    const int n = steps_;
    const double T = instrument.maturity();
    const double dt = T / n;
    const double vol = market.vol;

    const double u = std::exp(vol * std::sqrt(dt));
    const double d = 1.0 / u;
    const double disc = std::exp(-market.rate * dt);

    // Risk-neutral probability: growth at (r - q), NOT any real-world drift.
    const double growth = std::exp(market.drift() * dt);
    const double p = (growth - d) / (u - d);

    if (p < 0.0 || p > 1.0)
        throw std::runtime_error(
            "BinomialEngine: risk-neutral probability outside [0,1]; increase steps "
            "(dt too large for this vol/rate combination)");

    const Payoff& payoff = instrument.payoff();
    const bool american = instrument.is_american();

    // Terminal layer: spot at node j is S0 * u^j * d^(n-j).
    std::vector<double> values(n + 1);
    std::vector<double> spots(n + 1);
    for (int j = 0; j <= n; ++j) {
        spots[j] = market.spot * std::pow(u, 2 * j - n);
        values[j] = payoff(spots[j]);
    }

    // Roll backwards. At each node the continuation value is the discounted
    // risk-neutral expectation; an American holder takes the max against
    // intrinsic, which is where the early-exercise premium comes from.
    for (int i = n - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            const double continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
            if (american) {
                const double spot = market.spot * std::pow(u, 2 * j - i);
                values[j] = std::max(continuation, payoff(spot));
            } else {
                values[j] = continuation;
            }
        }
    }

    PricingResult out;
    out.price = values[0];
    return out;
}

}  // namespace dpe
