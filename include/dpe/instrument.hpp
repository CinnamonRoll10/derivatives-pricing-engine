#pragma once

#include <memory>
#include <stdexcept>
#include <utility>

#include "dpe/payoff.hpp"
#include "dpe/types.hpp"

namespace dpe {

/// An instrument is a payoff + an exercise style + a maturity.
///
/// Note what it is *not*: it carries no pricing method and no market data. The
/// same Instrument object is handed to every engine, which is what makes
/// method-vs-method benchmarking a fair comparison rather than four
/// near-duplicate re-specifications of the trade.
class Instrument {
public:
    Instrument(std::unique_ptr<Payoff> payoff, ExerciseStyle style, double maturity)
        : payoff_(std::move(payoff)), style_(style), maturity_(maturity) {
        if (!payoff_) throw std::invalid_argument("Instrument: null payoff");
        if (maturity_ <= 0.0) throw std::invalid_argument("Instrument: maturity must be > 0");
    }

    Instrument(const Instrument& other)
        : payoff_(other.payoff_->clone()), style_(other.style_), maturity_(other.maturity_) {}

    Instrument& operator=(const Instrument& other) {
        if (this != &other) {
            payoff_ = other.payoff_->clone();
            style_ = other.style_;
            maturity_ = other.maturity_;
        }
        return *this;
    }

    Instrument(Instrument&&) noexcept = default;
    Instrument& operator=(Instrument&&) noexcept = default;

    const Payoff& payoff() const { return *payoff_; }
    ExerciseStyle exercise() const { return style_; }
    double maturity() const { return maturity_; }

    bool is_american() const { return style_ == ExerciseStyle::American; }

private:
    std::unique_ptr<Payoff> payoff_;
    ExerciseStyle style_;
    double maturity_;
};

/// Convenience builders for the common cases.
inline Instrument make_vanilla(OptionType type, double strike, double maturity,
                               ExerciseStyle style = ExerciseStyle::European) {
    return Instrument(std::make_unique<VanillaPayoff>(type, strike), style, maturity);
}

inline Instrument make_barrier(OptionType type, double strike, double barrier,
                               BarrierDirection direction, BarrierType barrier_type,
                               double maturity, double rebate = 0.0) {
    return Instrument(
        std::make_unique<BarrierPayoff>(type, strike, barrier, direction, barrier_type, rebate),
        ExerciseStyle::European, maturity);
}

}  // namespace dpe
