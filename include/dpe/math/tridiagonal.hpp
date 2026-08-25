#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

namespace dpe::math {

/// Tridiagonal system stored by its three diagonals.
///
///   lower[i] * x[i-1] + diag[i] * x[i] + upper[i] * x[i+1] = rhs[i]
///
/// lower[0] and upper[n-1] are unused.
struct TridiagonalSystem {
    std::vector<double> lower, diag, upper;

    explicit TridiagonalSystem(std::size_t n) : lower(n, 0.0), diag(n, 0.0), upper(n, 0.0) {}
    std::size_t size() const { return diag.size(); }
};

/// Thomas algorithm -- O(n) direct solve. Used by the implicit and
/// Crank-Nicolson schemes for European payoffs, where no constraint applies.
inline std::vector<double> solve_thomas(const TridiagonalSystem& A,
                                        const std::vector<double>& rhs) {
    const std::size_t n = A.size();
    if (rhs.size() != n) throw std::invalid_argument("solve_thomas: size mismatch");

    std::vector<double> c(n, 0.0), d(n, 0.0), x(n, 0.0);

    if (std::abs(A.diag[0]) < 1e-300) throw std::runtime_error("solve_thomas: zero pivot");
    c[0] = A.upper[0] / A.diag[0];
    d[0] = rhs[0] / A.diag[0];

    for (std::size_t i = 1; i < n; ++i) {
        const double m = A.diag[i] - A.lower[i] * c[i - 1];
        if (std::abs(m) < 1e-300) throw std::runtime_error("solve_thomas: zero pivot");
        c[i] = A.upper[i] / m;
        d[i] = (rhs[i] - A.lower[i] * d[i - 1]) / m;
    }

    x[n - 1] = d[n - 1];
    for (std::size_t i = n - 1; i-- > 0;) x[i] = d[i] - c[i] * x[i + 1];
    return x;
}

/// Projected SOR for the linear complementarity problem
///
///     A x >= rhs,   x >= constraint,   (A x - rhs)(x - constraint) = 0
///
/// This is what an American option actually is under a PDE discretisation: you
/// cannot simply solve the linear system and then clip to the payoff, because
/// clipping after the fact does not satisfy the complementarity condition at
/// the free boundary. PSOR enforces the constraint inside the iteration.
inline std::vector<double> solve_psor(const TridiagonalSystem& A, const std::vector<double>& rhs,
                                      const std::vector<double>& constraint,
                                      const std::vector<double>& initial_guess,
                                      double omega = 1.5, double tol = 1e-10,
                                      int max_iter = 10000) {
    const std::size_t n = A.size();
    if (rhs.size() != n || constraint.size() != n || initial_guess.size() != n)
        throw std::invalid_argument("solve_psor: size mismatch");

    std::vector<double> x = initial_guess;
    for (std::size_t i = 0; i < n; ++i) x[i] = std::max(x[i], constraint[i]);

    for (int iter = 0; iter < max_iter; ++iter) {
        double max_change = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            double sum = rhs[i];
            if (i > 0) sum -= A.lower[i] * x[i - 1];
            if (i + 1 < n) sum -= A.upper[i] * x[i + 1];

            const double gs = sum / A.diag[i];                  // Gauss-Seidel update
            const double relaxed = x[i] + omega * (gs - x[i]);   // over-relax
            const double projected = std::max(relaxed, constraint[i]);  // project onto payoff

            max_change = std::max(max_change, std::abs(projected - x[i]));
            x[i] = projected;
        }

        if (max_change < tol) break;
    }
    return x;
}

}  // namespace dpe::math
