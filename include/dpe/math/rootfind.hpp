#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace dpe::math {

struct RootResult {
    double root{0.0};
    int iterations{0};
    bool converged{false};
    /// True when Newton was abandoned and the bracketing fallback produced the root.
    bool used_fallback{false};
};

/// Brent's method: bisection's guaranteed convergence with superlinear speed
/// when the function is well behaved. Requires a sign-changing bracket.
inline RootResult brent(const std::function<double(double)>& f, double lo, double hi,
                        double tol = 1e-10, int max_iter = 200) {
    double a = lo, b = hi;
    double fa = f(a), fb = f(b);

    if (fa * fb > 0.0) throw std::invalid_argument("brent: root is not bracketed");

    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a, fc = fa;
    double d = b - a, e = d;
    RootResult out;

    for (int i = 0; i < max_iter; ++i) {
        out.iterations = i + 1;

        if (fb * fc > 0.0) {
            c = a;
            fc = fa;
            d = e = b - a;
        }
        if (std::abs(fc) < std::abs(fb)) {
            a = b;  b = c;  c = a;
            fa = fb; fb = fc; fc = fa;
        }

        const double tol1 = 2.0 * std::numeric_limits<double>::epsilon() * std::abs(b) + 0.5 * tol;
        const double xm = 0.5 * (c - b);

        if (std::abs(xm) <= tol1 || fb == 0.0) {
            out.root = b;
            out.converged = true;
            return out;
        }

        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            // Attempt inverse quadratic interpolation (or secant if only two points).
            const double s = fb / fa;
            double p, q;
            if (a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                const double qq = fa / fc;
                const double r = fb / fc;
                p = s * (2.0 * xm * qq * (qq - r) - (b - a) * (r - 1.0));
                q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q;
            p = std::abs(p);

            const double min1 = 3.0 * xm * q - std::abs(tol1 * q);
            const double min2 = std::abs(e * q);
            if (2.0 * p < std::min(min1, min2)) {
                e = d;
                d = p / q;
            } else {
                d = xm;  // interpolation rejected, fall back to bisection
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }

        a = b;
        fa = fb;
        b += (std::abs(d) > tol1) ? d : (xm > 0.0 ? tol1 : -tol1);
        fb = f(b);
    }

    out.root = b;
    out.converged = false;
    return out;
}

/// Newton-Raphson with a hard requirement on the derivative.
///
/// Returns converged=false rather than throwing so callers can decide to fall
/// back to a bracketing method. This is the whole point of the implied-vol
/// design: vega collapses toward zero for deep in/out-of-the-money quotes, the
/// Newton step blows up, and we must degrade gracefully instead of diverging.
inline RootResult newton(const std::function<double(double)>& f,
                         const std::function<double(double)>& df, double x0,
                         double tol = 1e-10, int max_iter = 100,
                         double min_derivative = 1e-8) {
    RootResult out;
    double x = x0;

    for (int i = 0; i < max_iter; ++i) {
        out.iterations = i + 1;
        const double fx = f(x);

        if (std::abs(fx) < tol) {
            out.root = x;
            out.converged = true;
            return out;
        }

        const double d = df(x);
        if (!std::isfinite(d) || std::abs(d) < min_derivative) {
            out.root = x;
            out.converged = false;  // derivative too flat -- caller should bracket
            return out;
        }

        const double step = fx / d;
        x -= step;

        if (!std::isfinite(x)) {
            out.converged = false;
            return out;
        }
        if (std::abs(step) < tol) {
            out.root = x;
            out.converged = true;
            return out;
        }
    }

    out.root = x;
    out.converged = false;
    return out;
}

}  // namespace dpe::math
