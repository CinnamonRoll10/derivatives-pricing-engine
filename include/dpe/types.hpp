#pragma once

#include <stdexcept>
#include <string>

namespace dpe {

/// Right conveyed by a vanilla option.
enum class OptionType { Call, Put };

/// When the holder may exercise. Kept orthogonal to the payoff: a payoff says
/// *how much* you get, an exercise style says *when* you may take it.
enum class ExerciseStyle { European, American };

/// Direction the spot must travel to touch a barrier.
enum class BarrierDirection { Down, Up };

/// Whether touching the barrier kills the option or brings it to life.
enum class BarrierType { Out, In };

inline int sign(OptionType t) { return t == OptionType::Call ? 1 : -1; }

inline std::string to_string(OptionType t) {
    return t == OptionType::Call ? "Call" : "Put";
}

inline std::string to_string(ExerciseStyle e) {
    return e == ExerciseStyle::European ? "European" : "American";
}

/// Observable market state the models are calibrated to / priced off.
///
/// `vol` is the Black-Scholes flat volatility. `rate` and `dividend` are
/// continuously compounded. Pricing happens under the risk-neutral measure, so
/// the drift used everywhere is (rate - dividend) -- never a real-world mu.
struct MarketData {
    double spot{100.0};
    double rate{0.05};
    double dividend{0.0};
    double vol{0.20};

    void validate() const {
        if (spot <= 0.0) throw std::invalid_argument("MarketData: spot must be > 0");
        if (vol < 0.0) throw std::invalid_argument("MarketData: vol must be >= 0");
    }

    /// Risk-neutral drift of log-spot's exponential, i.e. E[S_T] = S_0 e^{drift*T}.
    double drift() const { return rate - dividend; }
};

/// Price plus whatever risk numbers an engine was able to produce.
///
/// Greeks are optional because not every engine computes every one: the
/// analytic engine fills all of them, Monte Carlo fills the ones it bumps.
struct PricingResult {
    double price{0.0};

    double delta{0.0};
    double gamma{0.0};
    double vega{0.0};
    double theta{0.0};
    double rho{0.0};
    bool has_greeks{false};

    /// Monte Carlo only: sample standard error of the price estimator.
    double std_error{0.0};
    bool has_std_error{false};

    /// 95% confidence half-width (1.96 * std_error).
    double ci_half_width() const { return 1.959963985 * std_error; }
};

}  // namespace dpe
