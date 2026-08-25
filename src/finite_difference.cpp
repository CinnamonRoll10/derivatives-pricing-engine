#include "dpe/engines/finite_difference.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include "dpe/math/tridiagonal.hpp"

namespace dpe {

std::string to_string(FDScheme s) {
    switch (s) {
        case FDScheme::Explicit: return "explicit";
        case FDScheme::Implicit: return "implicit";
        case FDScheme::CrankNicolson: return "Crank-Nicolson";
    }
    return "unknown";
}

std::string FiniteDifferenceEngine::name() const {
    std::string n = "FD " + to_string(scheme_) + " (" + std::to_string(spot_steps_) + "x" +
                    std::to_string(time_steps_) + ")";
    if (scheme_ == FDScheme::CrankNicolson && rannacher_) n += " +Rannacher";
    return n;
}

bool FiniteDifferenceEngine::supports(const Instrument& instrument) const {
    const Payoff& p = instrument.payoff();
    if (dynamic_cast<const VanillaPayoff*>(&p)) return true;
    if (dynamic_cast<const DigitalPayoff*>(&p)) return true;
    if (const auto* b = dynamic_cast<const BarrierPayoff*>(&p)) {
        // Knock-outs become a Dirichlet condition on a shrunken domain. Knock-ins
        // would need the in-out parity relation, which is not implemented here.
        return b->barrier_type() == BarrierType::Out && !instrument.is_american();
    }
    return false;
}

namespace {

struct Boundaries {
    double s_lo{0.0};
    double s_hi{0.0};
    std::function<double(double)> lower;  ///< V(s_lo, tau)
    std::function<double(double)> upper;  ///< V(s_hi, tau)
};

/// Dirichlet conditions are payoff-specific and only asymptotically correct at
/// the upper edge, which is why the domain is pushed out to a multiple of the
/// strike rather than stopping just past it.
Boundaries make_boundaries(const Instrument& inst, const MarketData& mkt, double s_hi) {
    const Payoff& p = inst.payoff();
    const double r = mkt.rate;
    const double q = mkt.dividend;
    const bool american = inst.is_american();

    Boundaries b;
    b.s_lo = 0.0;
    b.s_hi = s_hi;

    if (const auto* v = dynamic_cast<const VanillaPayoff*>(&p)) {
        const double K = v->strike();
        if (v->type() == OptionType::Call) {
            b.lower = [](double) { return 0.0; };
            b.upper = [=](double tau) {
                const double euro = s_hi * std::exp(-q * tau) - K * std::exp(-r * tau);
                return american ? std::max(euro, s_hi - K) : euro;
            };
        } else {
            // Deep in-the-money put: European decays with the discount factor,
            // American is worth K immediately.
            b.lower = [=](double tau) { return american ? K : K * std::exp(-r * tau); };
            b.upper = [](double) { return 0.0; };
        }
        return b;
    }

    if (const auto* d = dynamic_cast<const DigitalPayoff*>(&p)) {
        const double cash = d->cash();
        if (d->type() == OptionType::Call) {
            b.lower = [](double) { return 0.0; };
            b.upper = [=](double tau) { return cash * std::exp(-r * tau); };
        } else {
            b.lower = [=](double tau) { return cash * std::exp(-r * tau); };
            b.upper = [](double) { return 0.0; };
        }
        return b;
    }

    const auto* bar = dynamic_cast<const BarrierPayoff*>(&p);
    const double K = bar->strike();
    const double rebate = bar->rebate();

    if (bar->direction() == BarrierDirection::Down) {
        // Domain shrinks to [B, S_hi]; knocking out is a Dirichlet condition.
        b.s_lo = bar->barrier();
        b.lower = [=](double) { return rebate; };
        if (bar->type() == OptionType::Call)
            b.upper = [=](double tau) {
                return s_hi * std::exp(-q * tau) - K * std::exp(-r * tau);
            };
        else
            b.upper = [](double) { return 0.0; };
    } else {
        b.s_hi = bar->barrier();
        b.upper = [=](double) { return rebate; };
        if (bar->type() == OptionType::Call)
            b.lower = [](double) { return 0.0; };
        else
            b.lower = [=](double tau) { return K * std::exp(-r * tau); };
    }
    return b;
}

}  // namespace

PricingResult FiniteDifferenceEngine::price(const Instrument& instrument,
                                            const MarketData& market) const {
    market.validate();
    if (!supports(instrument))
        throw std::invalid_argument("FiniteDifferenceEngine: unsupported instrument");
    if (spot_steps_ < 3 || time_steps_ < 1)
        throw std::invalid_argument("FiniteDifferenceEngine: grid too coarse");

    const Payoff& payoff = instrument.payoff();
    const double T = instrument.maturity();
    const double vol = market.vol;
    const double r = market.rate;
    const double mu = market.drift();

    // Far boundary well beyond the money so the asymptotic condition is benign.
    double strike_guess = market.spot;
    if (const auto* v = dynamic_cast<const VanillaPayoff*>(&payoff)) strike_guess = v->strike();
    else if (const auto* d = dynamic_cast<const DigitalPayoff*>(&payoff)) strike_guess = d->strike();
    else if (const auto* b = dynamic_cast<const BarrierPayoff*>(&payoff)) strike_guess = b->strike();

    const double s_hi_target = spot_max_multiple_ * std::max(market.spot, strike_guess);
    Boundaries bc = make_boundaries(instrument, market, s_hi_target);

    if (market.spot <= bc.s_lo || market.spot >= bc.s_hi) {
        // Spot already outside the live domain (e.g. knocked out at inception).
        PricingResult out;
        out.price = market.spot <= bc.s_lo ? bc.lower(T) : bc.upper(T);
        return out;
    }

    int M = spot_steps_;

    // Align the grid so spot falls exactly on a node: removes interpolation
    // error from the reported price and makes the convergence order clean.
    double ds = (bc.s_hi - bc.s_lo) / M;
    int j_spot = static_cast<int>(std::lround((market.spot - bc.s_lo) / ds));
    j_spot = std::max(1, std::min(M - 1, j_spot));
    ds = (market.spot - bc.s_lo) / j_spot;
    const double s_hi = bc.s_lo + M * ds;
    bc.s_hi = s_hi;

    std::vector<double> S(M + 1), V(M + 1), intrinsic(M + 1);
    for (int j = 0; j <= M; ++j) {
        S[j] = bc.s_lo + j * ds;
        V[j] = payoff(S[j]);
        intrinsic[j] = payoff(S[j]);
    }
    // Knock-out payoff at the barrier node itself is the rebate, not the vanilla.
    if (dynamic_cast<const BarrierPayoff*>(&payoff)) {
        V[0] = bc.lower(0.0);
        V[M] = bc.upper(0.0);
    }

    // Spatial operator coefficients (independent of time for constant params).
    std::vector<double> a(M), b(M), c(M);
    for (int j = 1; j <= M - 1; ++j) {
        const double s = S[j];
        const double diff = vol * vol * s * s / (ds * ds);
        const double conv = mu * s / ds;
        a[j] = 0.5 * (diff - conv);
        b[j] = -(diff + r);
        c[j] = 0.5 * (diff + conv);
    }

    // Build the timestep schedule. Rannacher replaces the first two CN steps
    // with four half-size implicit ones to damp the payoff-kink oscillation.
    struct Step { double dt; double theta; };
    std::vector<Step> schedule;
    const double dt_full = T / time_steps_;
    const double theta_main = scheme_ == FDScheme::Explicit      ? 0.0
                              : scheme_ == FDScheme::Implicit    ? 1.0
                                                                 : 0.5;

    if (scheme_ == FDScheme::CrankNicolson && rannacher_ && time_steps_ >= 4) {
        for (int i = 0; i < 4; ++i) schedule.push_back({0.5 * dt_full, 1.0});
        for (int i = 0; i < time_steps_ - 2; ++i) schedule.push_back({dt_full, theta_main});
    } else {
        for (int i = 0; i < time_steps_; ++i) schedule.push_back({dt_full, theta_main});
    }

    if (scheme_ == FDScheme::Explicit) {
        // Stability requires dt <= 1 / max_j(|b_j|) for this discretisation.
        double max_b = 0.0;
        for (int j = 1; j <= M - 1; ++j) max_b = std::max(max_b, std::abs(b[j]));
        if (max_b * dt_full > 1.0)
            throw std::runtime_error(
                "FiniteDifferenceEngine: explicit scheme violates its stability limit "
                "(need time_steps >= " +
                std::to_string(static_cast<long long>(std::ceil(max_b * T))) + ")");
    }

    const bool american = instrument.is_american();
    double tau = 0.0;
    std::vector<double> prev_V;

    for (const Step& st : schedule) {
        prev_V = V;
        const double dt = st.dt;
        const double th = st.theta;
        const double tau_next = tau + dt;

        const std::size_t n = static_cast<std::size_t>(M - 1);
        math::TridiagonalSystem A(n);
        std::vector<double> rhs(n, 0.0);

        for (int j = 1; j <= M - 1; ++j) {
            const std::size_t i = static_cast<std::size_t>(j - 1);
            A.lower[i] = -th * dt * a[j];
            A.diag[i] = 1.0 - th * dt * b[j];
            A.upper[i] = -th * dt * c[j];

            rhs[i] = V[j] + (1.0 - th) * dt *
                                (a[j] * V[j - 1] + b[j] * V[j] + c[j] * V[j + 1]);
        }

        // Dirichlet values at the new time level move to the right-hand side.
        const double lo_next = bc.lower(tau_next);
        const double hi_next = bc.upper(tau_next);
        rhs[0] += th * dt * a[1] * lo_next;
        rhs[n - 1] += th * dt * c[M - 1] * hi_next;

        std::vector<double> interior;
        if (american) {
            std::vector<double> constraint(n), guess(n);
            for (int j = 1; j <= M - 1; ++j) {
                constraint[j - 1] = intrinsic[j];
                guess[j - 1] = V[j];
            }
            interior = math::solve_psor(A, rhs, constraint, guess);
        } else {
            interior = math::solve_thomas(A, rhs);
        }

        V[0] = lo_next;
        V[M] = hi_next;
        for (int j = 1; j <= M - 1; ++j) V[j] = interior[j - 1];

        tau = tau_next;
    }

    PricingResult out;
    out.price = V[j_spot];

    // Delta and gamma come free from the final grid; theta then follows exactly
    // from the PDE itself rather than needing another solve:
    //   dV/dt = rV - (r-q) S V_S - (1/2) vol^2 S^2 V_SS
    out.delta = (V[j_spot + 1] - V[j_spot - 1]) / (2.0 * ds);
    out.gamma = (V[j_spot + 1] - 2.0 * V[j_spot] + V[j_spot - 1]) / (ds * ds);
    const double s = S[j_spot];
    out.theta = r * out.price - mu * s * out.delta - 0.5 * vol * vol * s * s * out.gamma;
    out.has_greeks = true;  // delta/gamma/theta only; vega and rho need bumping

    return out;
}

}  // namespace dpe
