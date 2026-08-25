#include "dpe/engines/analytic_bs.hpp"

#include <cmath>
#include <stdexcept>

#include "dpe/math/normal.hpp"

namespace dpe {
namespace black_scholes {

using math::norm_cdf;
using math::norm_pdf;

namespace {
/// Guard against the degenerate corner where the diffusion term vanishes.
bool degenerate(double vol, double maturity) { return vol <= 0.0 || maturity <= 0.0; }

/// Intrinsic value discounted to today -- the correct limit as vol or T -> 0.
double forward_intrinsic(OptionType type, double spot, double strike, double rate,
                         double dividend, double maturity) {
    const double fwd = spot * std::exp(-dividend * maturity);
    const double disc_k = strike * std::exp(-rate * maturity);
    return type == OptionType::Call ? std::max(fwd - disc_k, 0.0) : std::max(disc_k - fwd, 0.0);
}
}  // namespace

double d1(double spot, double strike, double rate, double dividend, double vol, double maturity) {
    const double v = vol * std::sqrt(maturity);
    return (std::log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * maturity) / v;
}

double d2(double spot, double strike, double rate, double dividend, double vol, double maturity) {
    return d1(spot, strike, rate, dividend, vol, maturity) - vol * std::sqrt(maturity);
}

double price(OptionType type, double spot, double strike, double rate, double dividend,
             double vol, double maturity) {
    if (degenerate(vol, maturity))
        return forward_intrinsic(type, spot, strike, rate, dividend, maturity);

    const double D1 = d1(spot, strike, rate, dividend, vol, maturity);
    const double D2 = D1 - vol * std::sqrt(maturity);
    const double df_q = std::exp(-dividend * maturity);
    const double df_r = std::exp(-rate * maturity);

    if (type == OptionType::Call)
        return spot * df_q * norm_cdf(D1) - strike * df_r * norm_cdf(D2);
    return strike * df_r * norm_cdf(-D2) - spot * df_q * norm_cdf(-D1);
}

double delta(OptionType type, double spot, double strike, double rate, double dividend,
             double vol, double maturity) {
    if (degenerate(vol, maturity)) {
        const double fwd = spot * std::exp(-dividend * maturity);
        const double disc_k = strike * std::exp(-rate * maturity);
        const double df_q = std::exp(-dividend * maturity);
        if (type == OptionType::Call) return fwd > disc_k ? df_q : 0.0;
        return fwd < disc_k ? -df_q : 0.0;
    }
    const double D1 = d1(spot, strike, rate, dividend, vol, maturity);
    const double df_q = std::exp(-dividend * maturity);
    return type == OptionType::Call ? df_q * norm_cdf(D1) : df_q * (norm_cdf(D1) - 1.0);
}

double gamma(double spot, double strike, double rate, double dividend, double vol,
             double maturity) {
    if (degenerate(vol, maturity)) return 0.0;
    const double D1 = d1(spot, strike, rate, dividend, vol, maturity);
    return std::exp(-dividend * maturity) * norm_pdf(D1) / (spot * vol * std::sqrt(maturity));
}

double vega(double spot, double strike, double rate, double dividend, double vol,
            double maturity) {
    if (degenerate(vol, maturity)) return 0.0;
    const double D1 = d1(spot, strike, rate, dividend, vol, maturity);
    return spot * std::exp(-dividend * maturity) * norm_pdf(D1) * std::sqrt(maturity);
}

double theta(OptionType type, double spot, double strike, double rate, double dividend,
             double vol, double maturity) {
    if (degenerate(vol, maturity)) return 0.0;

    const double D1 = d1(spot, strike, rate, dividend, vol, maturity);
    const double D2 = D1 - vol * std::sqrt(maturity);
    const double df_q = std::exp(-dividend * maturity);
    const double df_r = std::exp(-rate * maturity);

    const double diffusion = -spot * df_q * norm_pdf(D1) * vol / (2.0 * std::sqrt(maturity));

    if (type == OptionType::Call)
        return diffusion - rate * strike * df_r * norm_cdf(D2) +
               dividend * spot * df_q * norm_cdf(D1);
    return diffusion + rate * strike * df_r * norm_cdf(-D2) -
           dividend * spot * df_q * norm_cdf(-D1);
}

double rho(OptionType type, double spot, double strike, double rate, double dividend, double vol,
           double maturity) {
    if (degenerate(vol, maturity)) return 0.0;
    const double D2 = d2(spot, strike, rate, dividend, vol, maturity);
    const double df_r = std::exp(-rate * maturity);
    if (type == OptionType::Call) return strike * maturity * df_r * norm_cdf(D2);
    return -strike * maturity * df_r * norm_cdf(-D2);
}

double digital_price(OptionType type, double spot, double strike, double rate, double dividend,
                     double vol, double maturity, double cash) {
    const double df_r = std::exp(-rate * maturity);
    if (degenerate(vol, maturity)) {
        const bool in = type == OptionType::Call ? (spot > strike) : (spot < strike);
        return in ? cash * df_r : 0.0;
    }
    const double D2 = d2(spot, strike, rate, dividend, vol, maturity);
    return cash * df_r * (type == OptionType::Call ? norm_cdf(D2) : norm_cdf(-D2));
}

double down_and_out_call(double spot, double strike, double barrier, double rate, double dividend,
                         double vol, double maturity) {
    if (barrier > strike)
        throw std::invalid_argument(
            "down_and_out_call: image solution requires barrier <= strike");
    if (spot <= barrier) return 0.0;  // already knocked out
    if (degenerate(vol, maturity))
        return forward_intrinsic(OptionType::Call, spot, strike, rate, dividend, maturity);

    const double k = 2.0 * (rate - dividend) / (vol * vol);
    const double vanilla = price(OptionType::Call, spot, strike, rate, dividend, vol, maturity);
    const double reflected =
        price(OptionType::Call, barrier * barrier / spot, strike, rate, dividend, vol, maturity);

    return vanilla - std::pow(spot / barrier, 1.0 - k) * reflected;
}

}  // namespace black_scholes

bool AnalyticBlackScholesEngine::supports(const Instrument& instrument) const {
    if (instrument.is_american()) return false;  // no closed form for American puts

    const Payoff& p = instrument.payoff();
    if (dynamic_cast<const VanillaPayoff*>(&p)) return true;
    if (dynamic_cast<const DigitalPayoff*>(&p)) return true;

    if (const auto* b = dynamic_cast<const BarrierPayoff*>(&p)) {
        // Only the case the image solution is valid for.
        return b->type() == OptionType::Call && b->direction() == BarrierDirection::Down &&
               b->barrier_type() == BarrierType::Out && b->barrier() <= b->strike() &&
               b->rebate() == 0.0;
    }
    return false;
}

PricingResult AnalyticBlackScholesEngine::price(const Instrument& instrument,
                                                const MarketData& market) const {
    market.validate();
    if (!supports(instrument))
        throw std::invalid_argument("AnalyticBlackScholesEngine: unsupported instrument");

    const double T = instrument.maturity();
    const Payoff& p = instrument.payoff();
    PricingResult out;

    if (const auto* v = dynamic_cast<const VanillaPayoff*>(&p)) {
        const auto type = v->type();
        const double K = v->strike();
        out.price = black_scholes::price(type, market.spot, K, market.rate, market.dividend,
                                         market.vol, T);
        out.delta = black_scholes::delta(type, market.spot, K, market.rate, market.dividend,
                                         market.vol, T);
        out.gamma = black_scholes::gamma(market.spot, K, market.rate, market.dividend,
                                         market.vol, T);
        out.vega = black_scholes::vega(market.spot, K, market.rate, market.dividend, market.vol, T);
        out.theta = black_scholes::theta(type, market.spot, K, market.rate, market.dividend,
                                         market.vol, T);
        out.rho = black_scholes::rho(type, market.spot, K, market.rate, market.dividend,
                                     market.vol, T);
        out.has_greeks = true;
        return out;
    }

    if (const auto* d = dynamic_cast<const DigitalPayoff*>(&p)) {
        out.price = black_scholes::digital_price(d->type(), market.spot, d->strike(), market.rate,
                                                 market.dividend, market.vol, T, d->cash());
        return out;
    }

    const auto* b = dynamic_cast<const BarrierPayoff*>(&p);
    out.price = black_scholes::down_and_out_call(market.spot, b->strike(), b->barrier(),
                                                 market.rate, market.dividend, market.vol, T);
    return out;
}

}  // namespace dpe
