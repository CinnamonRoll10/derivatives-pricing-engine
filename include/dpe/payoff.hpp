#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "dpe/types.hpp"

namespace dpe {

/// A payoff answers exactly one question: given the spot (or the whole path),
/// how much is this worth at exercise?
///
/// It knows nothing about volatility, discounting, exercise style or which
/// numerical scheme is pricing it. That separation is what lets a new payoff be
/// added without touching a single solver.
class Payoff {
public:
    virtual ~Payoff() = default;

    /// Value given terminal (or current) spot. Path-independent payoffs need
    /// only this; path-dependent ones still implement it for the terminal leg.
    virtual double operator()(double spot) const = 0;

    /// Path-dependent payoffs (barriers, Asians) override this. The default
    /// simply reads the terminal spot, so every path-independent payoff works
    /// unchanged inside a Monte Carlo engine.
    virtual double from_path(const std::vector<double>& path) const {
        return (*this)(path.back());
    }

    /// True when the payoff depends on more than the terminal spot. Engines use
    /// this to reject work they cannot do (an analytic formula cannot price a
    /// generic path-dependent claim).
    virtual bool is_path_dependent() const { return false; }

    virtual std::string name() const = 0;

    /// Virtual constructor. Lets an instrument own its payoff by value-semantics
    /// without knowing the concrete type -- the classic Joshi pattern.
    virtual std::unique_ptr<Payoff> clone() const = 0;
};

/// Standard call/put.
class VanillaPayoff : public Payoff {
public:
    VanillaPayoff(OptionType type, double strike) : type_(type), strike_(strike) {}

    double operator()(double spot) const override {
        return type_ == OptionType::Call ? std::max(spot - strike_, 0.0)
                                         : std::max(strike_ - spot, 0.0);
    }

    std::string name() const override { return "Vanilla" + to_string(type_); }
    std::unique_ptr<Payoff> clone() const override {
        return std::make_unique<VanillaPayoff>(*this);
    }

    OptionType type() const { return type_; }
    double strike() const { return strike_; }

private:
    OptionType type_;
    double strike_;
};

/// Cash-or-nothing digital. Included mainly because its discontinuity is what
/// makes Crank-Nicolson oscillate, which the convergence study exercises.
class DigitalPayoff : public Payoff {
public:
    DigitalPayoff(OptionType type, double strike, double cash = 1.0)
        : type_(type), strike_(strike), cash_(cash) {}

    double operator()(double spot) const override {
        const bool in = type_ == OptionType::Call ? (spot > strike_) : (spot < strike_);
        return in ? cash_ : 0.0;
    }

    std::string name() const override { return "Digital" + to_string(type_); }
    std::unique_ptr<Payoff> clone() const override {
        return std::make_unique<DigitalPayoff>(*this);
    }

    OptionType type() const { return type_; }
    double strike() const { return strike_; }
    double cash() const { return cash_; }

private:
    OptionType type_;
    double strike_;
    double cash_;
};

/// Knock-in / knock-out barrier on a vanilla payoff.
///
/// Path dependence lives entirely here: `from_path` decides whether the barrier
/// was breached and then defers to the underlying vanilla payoff. No engine
/// needs to know a barrier exists.
class BarrierPayoff : public Payoff {
public:
    BarrierPayoff(OptionType type, double strike, double barrier, BarrierDirection direction,
                  BarrierType barrier_type, double rebate = 0.0)
        : type_(type),
          strike_(strike),
          barrier_(barrier),
          direction_(direction),
          barrier_type_(barrier_type),
          rebate_(rebate),
          vanilla_(type, strike) {}

    /// Terminal-spot-only evaluation cannot see the path, so it can only be
    /// correct for the un-breached case. Engines must use from_path().
    double operator()(double spot) const override { return vanilla_(spot); }

    double from_path(const std::vector<double>& path) const override {
        bool touched = false;
        for (double s : path) {
            if (breaches(s)) {
                touched = true;
                break;
            }
        }
        const bool alive = (barrier_type_ == BarrierType::Out) ? !touched : touched;
        if (!alive) return rebate_;
        return vanilla_(path.back());
    }

    bool is_path_dependent() const override { return true; }

    bool breaches(double spot) const {
        return direction_ == BarrierDirection::Down ? (spot <= barrier_) : (spot >= barrier_);
    }

    std::string name() const override {
        std::string d = direction_ == BarrierDirection::Down ? "Down" : "Up";
        std::string b = barrier_type_ == BarrierType::Out ? "Out" : "In";
        return d + "And" + b + to_string(type_);
    }

    std::unique_ptr<Payoff> clone() const override {
        return std::make_unique<BarrierPayoff>(*this);
    }

    OptionType type() const { return type_; }
    double strike() const { return strike_; }
    double barrier() const { return barrier_; }
    BarrierDirection direction() const { return direction_; }
    BarrierType barrier_type() const { return barrier_type_; }
    double rebate() const { return rebate_; }

private:
    OptionType type_;
    double strike_;
    double barrier_;
    BarrierDirection direction_;
    BarrierType barrier_type_;
    double rebate_;
    VanillaPayoff vanilla_;
};

}  // namespace dpe
