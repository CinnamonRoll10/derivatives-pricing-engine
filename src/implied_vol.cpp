#include "dpe/implied_vol.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

#include "dpe/engines/analytic_bs.hpp"
#include "dpe/math/rootfind.hpp"

namespace dpe {

ImpliedVolResult implied_volatility(double market_price, OptionType type, double spot,
                                    double strike, double rate, double dividend, double maturity,
                                    double tol, double vol_lo, double vol_hi) {
    ImpliedVolResult out;

    if (maturity <= 0.0) {
        out.message = "maturity must be > 0";
        return out;
    }

    // No-arbitrage bounds. Outside these no volatility reproduces the quote, so
    // report failure rather than returning a meaningless root.
    const double fwd = spot * std::exp(-dividend * maturity);
    const double disc_k = strike * std::exp(-rate * maturity);
    const double lower_bound =
        type == OptionType::Call ? std::max(fwd - disc_k, 0.0) : std::max(disc_k - fwd, 0.0);
    const double upper_bound = type == OptionType::Call ? fwd : disc_k;

    if (market_price < lower_bound - 1e-12) {
        out.message = "price below intrinsic (no-arbitrage violation)";
        return out;
    }
    if (market_price > upper_bound + 1e-12) {
        out.message = "price above upper no-arbitrage bound";
        return out;
    }

    auto diff = [&](double v) {
        return black_scholes::price(type, spot, strike, rate, dividend, v, maturity) -
               market_price;
    };
    auto vega = [&](double v) {
        return black_scholes::vega(spot, strike, rate, dividend, v, maturity);
    };

    // Brenner-Subrahmanyam ATM approximation as the starting guess: for a
    // near-the-money option, price ~ 0.4 * S * vol * sqrt(T).
    double guess = std::sqrt(2.0 * M_PI / maturity) * market_price / spot;
    if (!std::isfinite(guess) || guess < vol_lo || guess > vol_hi) guess = 0.2;

    // Fill in the conditioning diagnostics for a recovered volatility.
    auto finalise = [&](double v, int iters, bool ok, const char* how) {
        out.vol = v;
        out.iterations = iters;
        out.converged = ok;
        out.message = how;
        out.vega = vega(v);
        // Vol resolution implied by the price tolerance. Guarded so a zero vega
        // reports "infinite uncertainty" rather than dividing by zero.
        out.vol_uncertainty = out.vega > 0.0 ? tol / out.vega
                                             : std::numeric_limits<double>::infinity();
        // "Well conditioned" promises the vol is pinned to within 1e-4, i.e. one
        // basis point of volatility. Anything looser is reported as such so a
        // caller never treats a noisy root as a usable quote.
        out.well_conditioned = ok && out.vol_uncertainty < 1e-4;
        return out;
    };

    const auto newton = math::newton(diff, vega, guess, tol, 100, 1e-8);

    if (newton.converged && newton.root > vol_lo && newton.root < vol_hi)
        return finalise(newton.root, newton.iterations, true, "newton");

    // Newton stalled -- almost always because vega collapsed. Bracket and use
    // Brent, which is guaranteed to converge on a sign-changing interval.
    out.used_fallback = true;
    const double f_lo = diff(vol_lo);
    const double f_hi = diff(vol_hi);

    if (f_lo * f_hi > 0.0) {
        out.message = "no sign change in [vol_lo, vol_hi]";
        return out;
    }

    try {
        const auto br = math::brent(diff, vol_lo, vol_hi, tol);
        return finalise(br.root, newton.iterations + br.iterations, br.converged,
                        br.converged ? "brent fallback" : "brent did not converge");
    } catch (const std::exception& e) {
        out.message = e.what();
    }
    return out;
}

void VolatilitySurface::build(const std::vector<VolQuote>& quotes, double spot, double rate,
                              double dividend) {
    points_.clear();
    std::set<double> ks, ts;

    for (const VolQuote& q : quotes) {
        const auto iv = implied_volatility(q.price, q.type, spot, q.strike, rate, dividend,
                                           q.maturity);
        SurfacePoint p;
        p.strike = q.strike;
        p.maturity = q.maturity;
        p.implied_vol = iv.vol;
        p.converged = iv.converged;
        p.used_fallback = iv.used_fallback;
        p.vega = iv.vega;
        p.well_conditioned = iv.well_conditioned;

        const double forward = spot * std::exp((rate - dividend) * q.maturity);
        p.moneyness = std::log(q.strike / forward);

        points_.push_back(p);
        ks.insert(q.strike);
        ts.insert(q.maturity);
    }

    strikes_.assign(ks.begin(), ks.end());
    maturities_.assign(ts.begin(), ts.end());
}

std::size_t VolatilitySurface::converged_count() const {
    return static_cast<std::size_t>(
        std::count_if(points_.begin(), points_.end(), [](const SurfacePoint& p) {
            return p.converged;
        }));
}

std::size_t VolatilitySurface::fallback_count() const {
    return static_cast<std::size_t>(
        std::count_if(points_.begin(), points_.end(), [](const SurfacePoint& p) {
            return p.used_fallback && p.converged;
        }));
}

std::size_t VolatilitySurface::usable_count() const {
    return static_cast<std::size_t>(
        std::count_if(points_.begin(), points_.end(), [](const SurfacePoint& p) {
            return p.well_conditioned;
        }));
}

double VolatilitySurface::interpolate(double strike, double maturity) const {
    if (points_.empty()) return 0.0;

    // Nearest-neighbour in maturity, linear in strike within that slice. Enough
    // for validation plots; not a production interpolator.
    double best_t = maturities_.empty() ? maturity : maturities_.front();
    for (double t : maturities_)
        if (std::abs(t - maturity) < std::abs(best_t - maturity)) best_t = t;

    std::vector<const SurfacePoint*> slice;
    for (const auto& p : points_)
        if (std::abs(p.maturity - best_t) < 1e-12 && p.well_conditioned) slice.push_back(&p);

    if (slice.empty()) return 0.0;
    std::sort(slice.begin(), slice.end(),
              [](const SurfacePoint* a, const SurfacePoint* b) { return a->strike < b->strike; });

    if (strike <= slice.front()->strike) return slice.front()->implied_vol;
    if (strike >= slice.back()->strike) return slice.back()->implied_vol;

    for (std::size_t i = 0; i + 1 < slice.size(); ++i) {
        if (strike <= slice[i + 1]->strike) {
            const double k0 = slice[i]->strike, k1 = slice[i + 1]->strike;
            const double w = (strike - k0) / (k1 - k0);
            return (1.0 - w) * slice[i]->implied_vol + w * slice[i + 1]->implied_vol;
        }
    }
    return slice.back()->implied_vol;
}

}  // namespace dpe
