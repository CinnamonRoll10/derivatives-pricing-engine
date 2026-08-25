// Produces the numerical evidence the project claims: cross-method agreement,
// Monte Carlo variance reduction, grid convergence order, and the American
// early-exercise premium. Every number quoted in the README comes from here.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dpe/engines/analytic_bs.hpp"
#include "dpe/engines/binomial.hpp"
#include "dpe/engines/finite_difference.hpp"
#include "dpe/engines/monte_carlo.hpp"
#include "dpe/implied_vol.hpp"
#include "dpe/instrument.hpp"

using namespace dpe;

namespace {

void header(const std::string& t) {
    std::cout << "\n" << std::string(78, '=') << "\n" << t << "\n" << std::string(78, '=') << "\n";
}

double elapsed_ms(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

/// Basis points of the reference price.
double bps(double value, double reference) {
    return 10000.0 * std::abs(value - reference) / std::abs(reference);
}

void study_cross_method_agreement() {
    header("1. Cross-method agreement on a European call (benchmark = closed form)");

    MarketData mkt{100.0, 0.05, 0.02, 0.20};
    auto opt = make_vanilla(OptionType::Call, 100.0, 1.0);

    const double exact = black_scholes::price(OptionType::Call, mkt.spot, 100.0, mkt.rate,
                                              mkt.dividend, mkt.vol, 1.0);

    std::printf("%-42s %14s %12s %10s\n", "method", "price", "diff (bps)", "ms");
    std::printf("%-42s %14.8f %12s %10s\n", "Closed-form Black-Scholes", exact, "-", "-");

    double worst = 0.0;

    auto run = [&](const PricingEngine& e) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto res = e.price(opt, mkt);
        const double ms = elapsed_ms(t0);
        const double d = bps(res.price, exact);
        worst = std::max(worst, d);
        std::printf("%-42s %14.8f %12.3f %10.1f\n", e.name().c_str(), res.price, d, ms);
    };

    run(BinomialEngine(5000));
    run(FiniteDifferenceEngine(800, 800, FDScheme::CrankNicolson));
    run(MonteCarloEngine(2000000, VarianceReduction::Both, 20240101));

    std::printf("\n  => all numerical methods agree with closed form to within %.2f bps\n", worst);
}

void study_variance_reduction() {
    header("2. Monte Carlo: O(1/sqrt(N)) convergence and variance reduction");

    MarketData mkt{100.0, 0.05, 0.0, 0.20};
    auto opt = make_vanilla(OptionType::Call, 100.0, 1.0);
    const double exact =
        black_scholes::price(OptionType::Call, 100.0, 100.0, 0.05, 0.0, 0.20, 1.0);

    std::cout << "\nStandard error vs path count (plain MC), exact = " << std::fixed
              << std::setprecision(8) << exact << "\n\n";
    std::printf("%12s %14s %14s %14s %12s\n", "paths", "price", "std error", "95% half-width",
                "SE ratio");

    double prev_se = 0.0;
    for (std::size_t n : {10000u, 50000u, 100000u, 500000u, 1000000u, 5000000u}) {
        MonteCarloEngine mc(n, VarianceReduction::None, 777);
        const auto r = mc.price(opt, mkt);
        std::printf("%12zu %14.8f %14.8f %14.8f %12s\n", n, r.price, r.std_error,
                    r.ci_half_width(),
                    prev_se > 0.0 ? (std::to_string(prev_se / r.std_error).substr(0, 6)).c_str()
                                  : "-");
        prev_se = r.std_error;
    }
    std::cout << "\n  (quadrupling paths should halve the standard error)\n";

    std::cout << "\nVariance reduction at a FIXED path budget of 1,000,000 paths:\n\n";
    std::printf("%-24s %14s %14s %14s %10s\n", "technique", "price", "std error", "vs plain",
                "ms");

    double se_plain = 0.0;
    for (auto vr : {VarianceReduction::None, VarianceReduction::Antithetic,
                    VarianceReduction::ControlVariate, VarianceReduction::Both}) {
        MonteCarloEngine mc(1000000, vr, 20240101);
        const auto t0 = std::chrono::steady_clock::now();
        const auto r = mc.price(opt, mkt);
        const double ms = elapsed_ms(t0);

        if (vr == VarianceReduction::None) se_plain = r.std_error;
        const double reduction = 100.0 * (1.0 - r.std_error / se_plain);

        std::printf("%-24s %14.8f %14.8f %13.1f%% %10.1f\n", to_string(vr).c_str(), r.price,
                    r.std_error, reduction, ms);
    }
    std::cout << "\n  (equal path count => equal cost, so this is a like-for-like comparison)\n";
}

void study_grid_convergence() {
    header("3. Finite difference: grid convergence order");

    MarketData mkt{100.0, 0.05, 0.0, 0.20};
    auto opt = make_vanilla(OptionType::Call, 100.0, 1.0);
    const double exact =
        black_scholes::price(OptionType::Call, 100.0, 100.0, 0.05, 0.0, 0.20, 1.0);

    for (auto scheme : {FDScheme::Implicit, FDScheme::CrankNicolson}) {
        std::cout << "\n" << to_string(scheme) << " (halving both dS and dt each row)\n\n";
        std::printf("%10s %10s %16s %14s %12s\n", "spot pts", "time pts", "price", "abs error",
                    "ratio");

        double prev_err = 0.0;
        for (int n : {50, 100, 200, 400, 800}) {
            FiniteDifferenceEngine fd(n, n, scheme);
            const double p = fd.price(opt, mkt).price;
            const double err = std::abs(p - exact);
            std::printf("%10d %10d %16.10f %14.3e %12s\n", n, n, p, err,
                        prev_err > 0.0 ? std::to_string(prev_err / err).substr(0, 6).c_str() : "-");
            prev_err = err;
        }
    }
    std::cout << "\n  (ratio ~2 => 1st order, ratio ~4 => 2nd order)\n";

    std::cout << "\nRannacher start-up on a DIGITAL payoff (discontinuous => CN oscillates).\n"
                 "Spatial grid is refined at a FIXED coarse time grid, which is exactly the\n"
                 "regime where Crank-Nicolson fails: its amplification factor tends to -1 for\n"
                 "high-frequency modes, so refining space injects modes CN does not damp.\n\n";

    MarketData m2{100.0, 0.05, 0.0, 0.20};
    Instrument dig(std::make_unique<DigitalPayoff>(OptionType::Call, 100.0, 1.0),
                   ExerciseStyle::European, 1.0);
    const double dig_exact =
        black_scholes::digital_price(OptionType::Call, 100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 1.0);
    std::printf("exact digital price = %.10f\n\n", dig_exact);

    std::printf("%8s %8s | %14s %14s | %14s %14s\n", "spot pts", "time pts", "CN price",
                "+Rannacher", "CN gamma", "+Rannacher");
    for (int M : {200, 400, 800, 1600}) {
        for (int N : {10, 25}) {
            const auto raw = FiniteDifferenceEngine(M, N, FDScheme::CrankNicolson, false).price(dig, m2);
            const auto ran = FiniteDifferenceEngine(M, N, FDScheme::CrankNicolson, true).price(dig, m2);
            std::printf("%8d %8d | %14.8f %14.8f | %14.6f %14.6f\n", M, N, raw.price, ran.price,
                        raw.gamma, ran.gamma);
        }
    }
    std::cout << "\n  (raw CN gamma changes SIGN and blows up as the spatial grid refines;\n"
                 "   four half-size implicit steps at the start damp it completely)\n";
}

void study_american() {
    header("4. American early exercise: tree vs PSOR finite difference");

    MarketData mkt{100.0, 0.05, 0.0, 0.30};
    const double K = 110.0, T = 1.0;

    auto euro = make_vanilla(OptionType::Put, K, T, ExerciseStyle::European);
    auto amer = make_vanilla(OptionType::Put, K, T, ExerciseStyle::American);

    const double bs_euro =
        black_scholes::price(OptionType::Put, mkt.spot, K, mkt.rate, mkt.dividend, mkt.vol, T);
    const double tree_euro = BinomialEngine(5000).price(euro, mkt).price;
    const double tree_amer = BinomialEngine(5000).price(amer, mkt).price;
    const double psor_amer =
        FiniteDifferenceEngine(800, 800, FDScheme::CrankNicolson).price(amer, mkt).price;

    std::printf("%-46s %16.8f\n", "European put, closed form", bs_euro);
    std::printf("%-46s %16.8f\n", "European put, binomial tree", tree_euro);
    std::printf("%-46s %16.8f\n", "American put, binomial tree", tree_amer);
    std::printf("%-46s %16.8f\n", "American put, Crank-Nicolson + PSOR", psor_amer);
    std::printf("\n%-46s %16.8f\n", "early-exercise premium (tree)", tree_amer - bs_euro);
    std::printf("%-46s %16.3f\n", "tree vs PSOR disagreement (bps)", bps(psor_amer, tree_amer));
    std::cout << "\n  (two independent methods agreeing on the premium is the real check)\n";
}

void study_barrier() {
    header("5. Barrier option: closed form vs PDE vs discretely-monitored MC");

    MarketData mkt{100.0, 0.05, 0.0, 0.20};
    const double K = 100.0, B = 90.0, T = 1.0;

    const double closed = black_scholes::down_and_out_call(mkt.spot, K, B, mkt.rate, mkt.dividend,
                                                          mkt.vol, T);
    const double vanilla =
        black_scholes::price(OptionType::Call, mkt.spot, K, mkt.rate, mkt.dividend, mkt.vol, T);

    auto bar = make_barrier(OptionType::Call, K, B, BarrierDirection::Down, BarrierType::Out, T);
    const double pde =
        FiniteDifferenceEngine(1000, 1000, FDScheme::CrankNicolson).price(bar, mkt).price;

    std::printf("%-52s %14.8f\n", "vanilla call (no barrier)", vanilla);
    std::printf("%-52s %14.8f\n", "down-and-out call, closed form (continuous)", closed);
    std::printf("%-52s %14.8f  (%.2f bps)\n", "down-and-out call, Crank-Nicolson PDE", pde,
                bps(pde, closed));

    std::cout << "\nDiscrete monitoring bias in Monte Carlo, validated against the\n"
                 "Broadie-Glasserman-Kou continuity correction, which predicts that a\n"
                 "discretely-monitored barrier behaves like a continuous one at a barrier\n"
                 "shifted away from the spot by exp(-0.5826 * vol * sqrt(T/m)):\n\n";
    std::printf("%8s %14s %12s %14s %12s %10s\n", "dates", "MC price", "MC se", "BGK predicted",
                "difference", "< 2 se?");

    const double beta = 0.5825971579;  // -zeta(1/2)/sqrt(2*pi)
    for (std::size_t steps : {12u, 50u, 250u, 1000u, 4000u}) {
        MonteCarloEngine mc(1000000, VarianceReduction::Antithetic, 20240101, steps);
        const auto r = mc.price(bar, mkt);

        const double b_adj =
            B * std::exp(-beta * mkt.vol * std::sqrt(T / static_cast<double>(steps)));
        const double predicted = black_scholes::down_and_out_call(mkt.spot, K, b_adj, mkt.rate,
                                                                  mkt.dividend, mkt.vol, T);
        const double diff = r.price - predicted;
        std::printf("%8zu %14.8f %12.8f %14.8f %12.8f %10s\n", steps, r.price, r.std_error,
                    predicted, diff, std::abs(diff) < 2.0 * r.std_error ? "yes" : "NO");
    }
    std::cout << "\n  (a discretely-monitored knock-out is always worth MORE: fewer observation\n"
                 "   dates mean fewer chances to breach. Matching BGK at every frequency shows\n"
                 "   the gap is the predicted discretisation bias, not an implementation error)\n";
}

void study_greeks() {
    header("6. Greeks: analytic vs finite difference vs Monte Carlo");

    MarketData mkt{100.0, 0.05, 0.01, 0.25};
    const double K = 105.0, T = 0.75;
    auto opt = make_vanilla(OptionType::Call, K, T);

    const double a_delta =
        black_scholes::delta(OptionType::Call, mkt.spot, K, mkt.rate, mkt.dividend, mkt.vol, T);
    const double a_gamma =
        black_scholes::gamma(mkt.spot, K, mkt.rate, mkt.dividend, mkt.vol, T);
    const double a_vega = black_scholes::vega(mkt.spot, K, mkt.rate, mkt.dividend, mkt.vol, T);

    const auto fd = FiniteDifferenceEngine(800, 800, FDScheme::CrankNicolson).price(opt, mkt);

    MonteCarloEngine mc(2000000, VarianceReduction::Antithetic, 20240101);
    mc.set_compute_greeks(true);
    const auto mcr = mc.price(opt, mkt);

    std::printf("%-10s %16s %16s %16s\n", "greek", "analytic", "FD grid", "MC (CRN bump)");
    std::printf("%-10s %16.8f %16.8f %16.8f\n", "delta", a_delta, fd.delta, mcr.delta);
    std::printf("%-10s %16.8f %16.8f %16.8f\n", "gamma", a_gamma, fd.gamma, mcr.gamma);
    std::printf("%-10s %16.8f %16s %16.8f\n", "vega", a_vega, "n/a", mcr.vega);
    std::cout << "\n  (MC gamma is the hardest: it is a second difference of a noisy\n"
                 "   estimator, which is why common random numbers are essential)\n";
}

void study_implied_vol() {
    header("7. Implied volatility: Newton-Raphson with Brent fallback");

    const double S = 100, r = 0.03, q = 0.0;

    auto true_vol = [](double K, double T) {
        const double m = std::log(K / 100.0);
        return 0.20 + 0.40 * m * m - 0.06 * m + 0.015 * T;
    };

    std::vector<VolQuote> quotes;
    for (double T : {0.05, 0.25, 1.0, 2.0}) {
        for (double K : {50.0, 70.0, 85.0, 100.0, 115.0, 130.0, 180.0}) {
            VolQuote qt;
            qt.strike = K;
            qt.maturity = T;
            qt.type = OptionType::Call;
            qt.price = black_scholes::price(OptionType::Call, S, K, r, q, true_vol(K, T), T);
            quotes.push_back(qt);
        }
    }

    VolatilitySurface surf;
    surf.build(quotes, S, r, q);

    std::printf("quotes inverted   : %zu / %zu\n", surf.converged_count(), quotes.size());
    std::printf("needed fallback   : %zu  (Newton stalled on near-zero vega)\n",
                surf.fallback_count());
    std::printf("usable for a fit  : %zu  (rest are ill-conditioned, see below)\n",
                surf.usable_count());

    double max_err_usable = 0.0, max_err_all = 0.0;
    for (const auto& p : surf.points()) {
        if (!p.converged) continue;
        const double e = std::abs(p.implied_vol - true_vol(p.strike, p.maturity));
        max_err_all = std::max(max_err_all, e);
        if (p.well_conditioned) max_err_usable = std::max(max_err_usable, e);
    }
    std::printf("\nmax error over WELL-CONDITIONED points : %.3e vol points\n", max_err_usable);
    std::printf("max error over ALL converged points    : %.3e vol points\n", max_err_all);
    std::cout << "\n  The second number looks alarming and is entirely expected: in the deep\n"
                 "  wings at short expiry, vega is ~1e-11, so the quote pins down no vol at\n"
                 "  all. The solver still returns a root that reprices correctly -- it simply\n"
                 "  flags that the root carries no information. A real chain is full of these,\n"
                 "  and feeding them to a smile fit would corrupt it.\n";

    std::cout << "\nRecovered smile ('  --  ' = ill-conditioned, excluded):\n\n";
    const std::vector<double> ks = {50.0, 70.0, 85.0, 100.0, 115.0, 130.0, 180.0};
    std::printf("%10s", "T \\ K");
    for (double K : ks) std::printf("%9.0f", K);
    std::cout << "\n";
    for (double T : {0.05, 0.25, 1.0, 2.0}) {
        std::printf("%10.2f", T);
        for (double K : ks) {
            const SurfacePoint* hit = nullptr;
            for (const auto& p : surf.points())
                if (std::abs(p.strike - K) < 1e-9 && std::abs(p.maturity - T) < 1e-9) hit = &p;
            if (hit && hit->well_conditioned) std::printf("%9.4f", hit->implied_vol);
            else std::printf("%9s", "--");
        }
        std::cout << "\n";
    }
    std::printf("%10s", "true");
    for (double K : ks) std::printf("%9.4f", true_vol(K, 1.0));
    std::cout << "   <- input smile at T=1.00\n";
}

}  // namespace

int main() {
    std::cout << "Derivatives Pricing Engine -- numerical validation report\n";
    study_cross_method_agreement();
    study_variance_reduction();
    study_grid_convergence();
    study_american();
    study_barrier();
    study_greeks();
    study_implied_vol();
    std::cout << "\n";
    return 0;
}
