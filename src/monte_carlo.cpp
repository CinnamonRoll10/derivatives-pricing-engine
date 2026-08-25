#include "dpe/engines/monte_carlo.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "dpe/math/normal.hpp"

namespace dpe {

std::string to_string(VarianceReduction vr) {
    switch (vr) {
        case VarianceReduction::None: return "plain";
        case VarianceReduction::Antithetic: return "antithetic";
        case VarianceReduction::ControlVariate: return "control-variate";
        case VarianceReduction::Both: return "antithetic+control";
    }
    return "unknown";
}

std::string MonteCarloEngine::name() const {
    return "Monte Carlo (" + std::to_string(paths_) + " paths, " + to_string(vr_) + ")";
}

namespace {

/// Deterministic normal draws via inverse-CDF of a uniform stream.
///
/// Deliberately not std::normal_distribution: its output is not specified by the
/// standard, so results would not reproduce across compilers -- unacceptable
/// when the whole point is cross-method agreement to a stated tolerance.
class NormalStream {
public:
    explicit NormalStream(std::uint64_t seed) : rng_(seed) {}

    double next() {
        // Open interval: inverse CDF is infinite at 0 and 1.
        const double u = uniform_(rng_);
        return math::norm_inv_cdf(u);
    }

private:
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_{
        std::nextafter(0.0, 1.0), std::nextafter(1.0, 0.0)};
};

struct Accumulator {
    double sum{0.0};
    double sum_sq{0.0};
    std::size_t n{0};

    void add(double x) {
        sum += x;
        sum_sq += x * x;
        ++n;
    }

    double mean() const { return n ? sum / static_cast<double>(n) : 0.0; }

    /// Sample variance with Bessel's correction.
    double variance() const {
        if (n < 2) return 0.0;
        const double m = mean();
        const double v = (sum_sq - static_cast<double>(n) * m * m) / static_cast<double>(n - 1);
        return v > 0.0 ? v : 0.0;
    }

    double std_error() const {
        return n ? std::sqrt(variance() / static_cast<double>(n)) : 0.0;
    }
};

}  // namespace

/// Path-independent case: jump straight to maturity with the exact GBM solution
///   S_T = S_0 exp((r - q - vol^2/2) T + vol sqrt(T) Z)
/// so there is zero time-discretisation error -- only sampling error.
PricingResult MonteCarloEngine::price_terminal(const Instrument& instrument,
                                               const MarketData& market) const {
    const double T = instrument.maturity();
    const double disc = std::exp(-market.rate * T);
    const double drift = (market.drift() - 0.5 * market.vol * market.vol) * T;
    const double diffusion = market.vol * std::sqrt(T);
    const Payoff& payoff = instrument.payoff();

    const bool antithetic =
        vr_ == VarianceReduction::Antithetic || vr_ == VarianceReduction::Both;
    const bool control =
        vr_ == VarianceReduction::ControlVariate || vr_ == VarianceReduction::Both;

    // Cost-matched comparison: `paths_` is always the number of simulated paths,
    // so antithetic runs paths_/2 pairs rather than paths_ pairs.
    const std::size_t draws = antithetic ? paths_ / 2 : paths_;
    if (draws == 0) throw std::invalid_argument("MonteCarloEngine: too few paths");

    NormalStream z(seed_);
    std::vector<double> ys, xs;
    ys.reserve(draws);
    if (control) xs.reserve(draws);

    for (std::size_t i = 0; i < draws; ++i) {
        const double zi = z.next();
        const double s_up = market.spot * std::exp(drift + diffusion * zi);

        double y = disc * payoff(s_up);
        double x = disc * s_up;

        if (antithetic) {
            const double s_dn = market.spot * std::exp(drift - diffusion * zi);
            y = 0.5 * (y + disc * payoff(s_dn));
            x = 0.5 * (x + disc * s_dn);
        }

        ys.push_back(y);
        if (control) xs.push_back(x);
    }

    Accumulator acc;

    if (control) {
        // Control variate: E[e^{-rT} S_T] = S_0 e^{-qT} is known in closed form.
        // Regress the payoff on it and subtract the (zero-mean) residual.
        const double expected_x = market.spot * std::exp(-market.dividend * T);

        double mean_y = 0.0, mean_x = 0.0;
        for (std::size_t i = 0; i < draws; ++i) {
            mean_y += ys[i];
            mean_x += xs[i];
        }
        mean_y /= static_cast<double>(draws);
        mean_x /= static_cast<double>(draws);

        double cov = 0.0, var_x = 0.0;
        for (std::size_t i = 0; i < draws; ++i) {
            const double dx = xs[i] - mean_x;
            cov += dx * (ys[i] - mean_y);
            var_x += dx * dx;
        }

        // beta is estimated from the same sample, which introduces a small bias
        // that vanishes as draws grows. Standard practice for this estimator.
        const double beta = var_x > 0.0 ? cov / var_x : 0.0;

        for (std::size_t i = 0; i < draws; ++i)
            acc.add(ys[i] - beta * (xs[i] - expected_x));
    } else {
        for (double y : ys) acc.add(y);
    }

    PricingResult out;
    out.price = acc.mean();
    out.std_error = acc.std_error();
    out.has_std_error = true;
    return out;
}

/// Path-dependent case: step the path so the payoff can inspect it.
PricingResult MonteCarloEngine::price_path(const Instrument& instrument,
                                           const MarketData& market) const {
    const double T = instrument.maturity();
    const std::size_t m = steps_;
    if (m == 0) throw std::invalid_argument("MonteCarloEngine: steps must be >= 1");

    const double dt = T / static_cast<double>(m);
    const double disc = std::exp(-market.rate * T);
    const double drift = (market.drift() - 0.5 * market.vol * market.vol) * dt;
    const double diffusion = market.vol * std::sqrt(dt);
    const Payoff& payoff = instrument.payoff();

    const bool antithetic =
        vr_ == VarianceReduction::Antithetic || vr_ == VarianceReduction::Both;
    const std::size_t draws = antithetic ? paths_ / 2 : paths_;
    if (draws == 0) throw std::invalid_argument("MonteCarloEngine: too few paths");

    NormalStream z(seed_);
    Accumulator acc;

    std::vector<double> path(m + 1), anti(m + 1);

    for (std::size_t i = 0; i < draws; ++i) {
        path[0] = market.spot;
        anti[0] = market.spot;

        for (std::size_t k = 1; k <= m; ++k) {
            const double zi = z.next();
            path[k] = path[k - 1] * std::exp(drift + diffusion * zi);
            if (antithetic) anti[k] = anti[k - 1] * std::exp(drift - diffusion * zi);
        }

        double y = disc * payoff.from_path(path);
        if (antithetic) y = 0.5 * (y + disc * payoff.from_path(anti));
        acc.add(y);
    }

    PricingResult out;
    out.price = acc.mean();
    out.std_error = acc.std_error();
    out.has_std_error = true;
    return out;
}

PricingResult MonteCarloEngine::price_raw(const Instrument& instrument,
                                          const MarketData& market) const {
    return instrument.payoff().is_path_dependent() ? price_path(instrument, market)
                                                    : price_terminal(instrument, market);
}

PricingResult MonteCarloEngine::price(const Instrument& instrument,
                                      const MarketData& market) const {
    market.validate();
    PricingResult out = price_raw(instrument, market);
    if (!greeks_) return out;

    // Bump sizes: relative for spot, absolute for vol. Chosen well above Monte
    // Carlo noise once common random numbers cancel the bulk of it, but small
    // enough that the central difference stays second-order accurate.
    const double h_s = 0.01 * market.spot;
    const double h_v = 0.01;

    MarketData up = market, down = market;
    up.spot += h_s;
    down.spot -= h_s;

    // Same seed on every call => same draws => differences are not pure noise.
    const double p_up = price_raw(instrument, up).price;
    const double p_dn = price_raw(instrument, down).price;

    out.delta = (p_up - p_dn) / (2.0 * h_s);
    out.gamma = (p_up - 2.0 * out.price + p_dn) / (h_s * h_s);

    MarketData v_up = market, v_dn = market;
    v_up.vol += h_v;
    v_dn.vol = std::max(market.vol - h_v, 0.0);
    out.vega = (price_raw(instrument, v_up).price - price_raw(instrument, v_dn).price) /
               (v_up.vol - v_dn.vol);

    out.has_greeks = true;
    return out;
}

}  // namespace dpe
