#pragma once

#include <cmath>
#include <limits>

namespace dpe::math {

inline constexpr double kInvSqrt2Pi = 0.3989422804014327;
inline constexpr double kInvSqrt2 = 0.7071067811865476;

/// Standard normal probability density.
inline double norm_pdf(double x) { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }

/// Standard normal cumulative distribution.
///
/// std::erfc is accurate to near machine precision and does not lose relative
/// accuracy in the left tail the way 1 - Phi(-x) would, which matters for the
/// deep out-of-the-money quotes the implied-vol solver has to handle.
inline double norm_cdf(double x) { return 0.5 * std::erfc(-x * kInvSqrt2); }

/// Inverse standard normal CDF (Acklam's rational approximation).
///
/// Relative error < 1.15e-9 over the whole domain, which is well inside Monte
/// Carlo error for any practical path count.
inline double norm_inv_cdf(double p) {
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return std::numeric_limits<double>::infinity();

    static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                -2.759285104469687e+02, 1.383577518672690e+02,
                                -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                -1.556989798598866e+02, 6.680131188771972e+01,
                                -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                -2.400758277161838e+00, -2.549732539343734e+00,
                                4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                                2.445134137142996e+00, 3.754408661907416e+00};

    const double p_low = 0.02425;
    const double p_high = 1.0 - p_low;

    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p > p_high) {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const double q = p - 0.5;
    const double r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

/// Bivariate normal CDF (Drezner-Wesolowsky). Needed by some barrier formulas.
inline double bivariate_norm_cdf(double a, double b, double rho) {
    if (std::abs(rho) < 1e-14) return norm_cdf(a) * norm_cdf(b);

    static const double x[5] = {0.04691008, 0.23076534, 0.5, 0.76923466, 0.95308992};
    static const double w[5] = {0.018854042, 0.038088059, 0.0452707394, 0.038088059,
                                0.018854042};

    const double h1 = a;
    const double h2 = b;
    double lhk = 0.0;
    for (int i = 0; i < 5; ++i) {
        const double r = rho * x[i];
        const double denom = std::sqrt(1.0 - r * r);
        lhk += w[i] * std::exp((h1 * h2 * r - 0.5 * (h1 * h1 + h2 * h2)) / (denom * denom)) /
               denom;
    }
    return norm_cdf(a) * norm_cdf(b) + rho * lhk;
}

}  // namespace dpe::math
