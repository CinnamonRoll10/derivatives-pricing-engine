#pragma once

#include <string>
#include <vector>

#include "dpe/types.hpp"

namespace dpe {

struct ImpliedVolResult {
    double vol{0.0};
    int iterations{0};
    bool converged{false};
    /// True when Newton stalled on near-zero vega and Brent finished the job.
    bool used_fallback{false};
    std::string message;

    /// Vega at the recovered volatility -- the conditioning of the inversion.
    double vega{0.0};

    /// Roughly how much the volatility could move without changing the price by
    /// more than the solver tolerance, i.e. tol / vega.
    ///
    /// This matters more than it looks. For a deep in-the-money option vega can
    /// be ~1e-11, meaning a whole percentage point of volatility moves the price
    /// by less than 1e-11. The price then pins down *no* volatility at all: the
    /// root is found correctly, but the inverse problem itself is ill-posed.
    /// Reporting this stops a caller trusting a number that carries no
    /// information.
    double vol_uncertainty{0.0};

    /// False when vega is too small for the implied vol to be meaningful.
    bool well_conditioned{false};
};

/// Invert Black-Scholes for volatility given a market price.
///
/// Newton-Raphson first, because vega is available analytically and convergence
/// is quadratic near the money. But vega -> 0 for deep in- and out-of-the-money
/// quotes and for very short expiries, so the Newton step f/f' explodes exactly
/// where real option chains have the most strikes. When that happens we fall
/// back to Brent on a bracketed interval, which cannot diverge.
///
/// Returns converged=false when the quote is outside the no-arbitrage bounds,
/// since no volatility reproduces such a price.
ImpliedVolResult implied_volatility(double market_price, OptionType type, double spot,
                                    double strike, double rate, double dividend,
                                    double maturity, double tol = 1e-10,
                                    double vol_lo = 1e-6, double vol_hi = 5.0);

/// One quoted option in a chain.
struct VolQuote {
    double strike{0.0};
    double maturity{0.0};
    double price{0.0};
    OptionType type{OptionType::Call};
};

struct SurfacePoint {
    double strike{0.0};
    double maturity{0.0};
    double implied_vol{0.0};
    double moneyness{0.0};  ///< log(K / F), F being the forward
    bool converged{false};
    bool used_fallback{false};

    /// Vega at the recovered vol, and whether it is large enough for the number
    /// to mean anything. Deep wings on short expiries routinely invert to a
    /// "converged" root that carries no information -- a real chain is full of
    /// these, and quoting them would corrupt any fit built on the surface.
    double vega{0.0};
    bool well_conditioned{false};
};

/// Implied volatility surface built by inverting a chain quote by quote.
class VolatilitySurface {
public:
    void build(const std::vector<VolQuote>& quotes, double spot, double rate, double dividend);

    const std::vector<SurfacePoint>& points() const { return points_; }

    /// Bilinear interpolation in (strike, maturity). Deliberately simple: a
    /// production surface would fit a parametric form (SVI) to enforce
    /// no-arbitrage in both directions, which is out of scope here.
    double interpolate(double strike, double maturity) const;

    std::size_t converged_count() const;
    std::size_t fallback_count() const;
    /// Points a downstream model fit should actually be allowed to use.
    std::size_t usable_count() const;

private:
    std::vector<SurfacePoint> points_;
    std::vector<double> strikes_;
    std::vector<double> maturities_;
};

}  // namespace dpe
