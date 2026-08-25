#pragma once

#include "dpe/engine.hpp"

namespace dpe {

/// Closed-form Black-Scholes prices and Greeks.
///
/// This is the benchmark: every numerical scheme in the library is measured
/// against it, so it is written for accuracy rather than speed and handles the
/// degenerate cases (zero vol, zero time) explicitly instead of dividing by
/// zero and returning NaN.
namespace black_scholes {

/// d1 = [ln(S/K) + (r - q + s^2/2) T] / (s sqrt(T))
double d1(double spot, double strike, double rate, double dividend, double vol, double maturity);
double d2(double spot, double strike, double rate, double dividend, double vol, double maturity);

double price(OptionType type, double spot, double strike, double rate, double dividend,
             double vol, double maturity);

double delta(OptionType type, double spot, double strike, double rate, double dividend,
             double vol, double maturity);
double gamma(double spot, double strike, double rate, double dividend, double vol,
             double maturity);
/// Per 1.00 of vol (not per vol point) -- divide by 100 for the trader convention.
double vega(double spot, double strike, double rate, double dividend, double vol,
            double maturity);
/// Per year (not per day) -- divide by 365 for the trader convention.
double theta(OptionType type, double spot, double strike, double rate, double dividend,
             double vol, double maturity);
double rho(OptionType type, double spot, double strike, double rate, double dividend,
           double vol, double maturity);

/// Cash-or-nothing digital, closed form.
double digital_price(OptionType type, double spot, double strike, double rate, double dividend,
                     double vol, double maturity, double cash = 1.0);

/// Continuously-monitored down-and-out call via the method of images.
///
/// Valid for barrier <= strike and barrier < spot. The image solution is
///     C_do(S) = C(S) - (S/B)^{1-k} C(B^2/S),   k = 2(r-q)/vol^2
/// which vanishes at S = B by construction. Used to validate the Monte Carlo
/// barrier pricer, whose discrete monitoring biases it the other way.
double down_and_out_call(double spot, double strike, double barrier, double rate,
                         double dividend, double vol, double maturity);

}  // namespace black_scholes

class AnalyticBlackScholesEngine : public PricingEngine {
public:
    PricingResult price(const Instrument& instrument, const MarketData& market) const override;
    std::string name() const override { return "Analytic Black-Scholes"; }
    bool supports(const Instrument& instrument) const override;
};

}  // namespace dpe
