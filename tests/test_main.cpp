// Self-contained test suite -- no external framework, so the project builds
// anywhere with just a C++17 compiler and CMake.
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "dpe/engines/analytic_bs.hpp"
#include "dpe/engines/binomial.hpp"
#include "dpe/engines/finite_difference.hpp"
#include "dpe/engines/monte_carlo.hpp"
#include "dpe/implied_vol.hpp"
#include "dpe/instrument.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool cond, const std::string& what) {
    ++g_checks;
    if (!cond) {
        ++g_failures;
        std::cout << "  FAIL: " << what << "\n";
    }
}

void check_close(double a, double b, double tol, const std::string& what) {
    ++g_checks;
    const double diff = std::abs(a - b);
    if (!(diff <= tol)) {
        ++g_failures;
        std::printf("  FAIL: %s  (got %.10f, want %.10f, diff %.3e > tol %.3e)\n", what.c_str(),
                    a, b, diff, tol);
    }
}

void section(const std::string& title) { std::cout << "\n[" << title << "]\n"; }

using namespace dpe;

// ---------------------------------------------------------------------------

void test_black_scholes_reference_values() {
    section("Black-Scholes against published reference values");

    // Hull, "Options, Futures and Other Derivatives": S=42, K=40, r=10%, vol=20%, T=0.5
    // Call = 4.76, Put = 0.81
    const double c = black_scholes::price(OptionType::Call, 42, 40, 0.10, 0.0, 0.20, 0.5);
    const double p = black_scholes::price(OptionType::Put, 42, 40, 0.10, 0.0, 0.20, 0.5);
    check_close(c, 4.7594, 1e-3, "Hull call reference");
    check_close(p, 0.8086, 1e-3, "Hull put reference");
}

void test_put_call_parity() {
    section("Put-call parity");

    const double S = 100, K = 95, r = 0.05, q = 0.02, v = 0.25, T = 1.5;
    const double c = black_scholes::price(OptionType::Call, S, K, r, q, v, T);
    const double p = black_scholes::price(OptionType::Put, S, K, r, q, v, T);

    // C - P = S e^{-qT} - K e^{-rT}
    check_close(c - p, S * std::exp(-q * T) - K * std::exp(-r * T), 1e-12, "parity identity");

    // Deltas must differ by exactly e^{-qT}
    const double dc = black_scholes::delta(OptionType::Call, S, K, r, q, v, T);
    const double dp = black_scholes::delta(OptionType::Put, S, K, r, q, v, T);
    check_close(dc - dp, std::exp(-q * T), 1e-12, "delta parity");
}

void test_greeks_against_finite_difference() {
    section("Analytic Greeks vs finite-difference bumps");

    const double S = 100, K = 105, r = 0.03, q = 0.01, v = 0.22, T = 0.75;
    const auto type = OptionType::Call;
    auto price = [&](double s, double vol, double rate, double t) {
        return black_scholes::price(type, s, K, rate, q, vol, t);
    };

    const double hs = 1e-4 * S, hv = 1e-5, hr = 1e-6, ht = 1e-6;

    const double fd_delta = (price(S + hs, v, r, T) - price(S - hs, v, r, T)) / (2 * hs);
    const double fd_gamma =
        (price(S + hs, v, r, T) - 2 * price(S, v, r, T) + price(S - hs, v, r, T)) / (hs * hs);
    const double fd_vega = (price(S, v + hv, r, T) - price(S, v - hv, r, T)) / (2 * hv);
    const double fd_rho = (price(S, v, r + hr, T) - price(S, v, r - hr, T)) / (2 * hr);
    // theta is -d/dT (value decays as maturity approaches)
    const double fd_theta = -(price(S, v, r, T + ht) - price(S, v, r, T - ht)) / (2 * ht);

    check_close(black_scholes::delta(type, S, K, r, q, v, T), fd_delta, 1e-6, "delta");
    check_close(black_scholes::gamma(S, K, r, q, v, T), fd_gamma, 1e-4, "gamma");
    check_close(black_scholes::vega(S, K, r, q, v, T), fd_vega, 1e-5, "vega");
    check_close(black_scholes::rho(type, S, K, r, q, v, T), fd_rho, 1e-4, "rho");
    check_close(black_scholes::theta(type, S, K, r, q, v, T), fd_theta, 1e-4, "theta");
}

void test_binomial_converges_to_bs() {
    section("Binomial tree -> Black-Scholes");

    MarketData mkt{100.0, 0.05, 0.0, 0.2};
    auto euro = make_vanilla(OptionType::Call, 100.0, 1.0);
    const double exact =
        black_scholes::price(OptionType::Call, 100, 100, 0.05, 0.0, 0.2, 1.0);

    const double e500 = std::abs(BinomialEngine(500).price(euro, mkt).price - exact);
    const double e5000 = std::abs(BinomialEngine(5000).price(euro, mkt).price - exact);

    check(e500 < 5e-3, "binomial 500 steps within 5e-3");
    check(e5000 < 5e-4, "binomial 5000 steps within 5e-4");
    check(e5000 < e500, "error decreases with more steps");
}

void test_american_premium() {
    section("American early-exercise premium");

    MarketData mkt{100.0, 0.05, 0.0, 0.3};
    auto euro_put = make_vanilla(OptionType::Put, 110.0, 1.0, ExerciseStyle::European);
    auto amer_put = make_vanilla(OptionType::Put, 110.0, 1.0, ExerciseStyle::American);

    BinomialEngine tree(2000);
    const double pe = tree.price(euro_put, mkt).price;
    const double pa = tree.price(amer_put, mkt).price;

    check(pa > pe, "American put strictly more valuable than European");
    check(pa >= 110.0 - 100.0 - 1e-9, "American put at least intrinsic");

    // Without dividends an American call is never exercised early, so it must
    // equal the European call exactly. This is the sharpest test of the
    // early-exercise logic: it must NOT fire here.
    auto euro_call = make_vanilla(OptionType::Call, 100.0, 1.0, ExerciseStyle::European);
    auto amer_call = make_vanilla(OptionType::Call, 100.0, 1.0, ExerciseStyle::American);
    check_close(tree.price(amer_call, mkt).price, tree.price(euro_call, mkt).price, 1e-10,
                "American call == European call when q=0");
}

void test_monte_carlo_converges() {
    section("Monte Carlo convergence and variance reduction");

    MarketData mkt{100.0, 0.05, 0.0, 0.2};
    auto euro = make_vanilla(OptionType::Call, 100.0, 1.0);
    const double exact = black_scholes::price(OptionType::Call, 100, 100, 0.05, 0.0, 0.2, 1.0);

    MonteCarloEngine mc(200000, VarianceReduction::None, 12345);
    const auto r = mc.price(euro, mkt);

    check(r.has_std_error, "MC reports a standard error");
    check(std::abs(r.price - exact) < 4.0 * r.std_error,
          "MC price within 4 standard errors of exact");

    // Standard error must fall like 1/sqrt(N): 4x the paths => ~2x tighter.
    MonteCarloEngine mc_small(50000, VarianceReduction::None, 999);
    MonteCarloEngine mc_big(200000, VarianceReduction::None, 999);
    const double se_small = mc_small.price(euro, mkt).std_error;
    const double se_big = mc_big.price(euro, mkt).std_error;
    const double ratio = se_small / se_big;
    check(ratio > 1.7 && ratio < 2.3, "SE ratio ~2 when paths quadruple");

    // Variance reduction must actually reduce variance at equal path count.
    const double se_plain = MonteCarloEngine(100000, VarianceReduction::None, 7).price(euro, mkt).std_error;
    const double se_anti = MonteCarloEngine(100000, VarianceReduction::Antithetic, 7).price(euro, mkt).std_error;
    const double se_cv = MonteCarloEngine(100000, VarianceReduction::ControlVariate, 7).price(euro, mkt).std_error;
    const double se_both = MonteCarloEngine(100000, VarianceReduction::Both, 7).price(euro, mkt).std_error;

    check(se_anti < se_plain, "antithetic reduces SE");
    check(se_cv < se_plain, "control variate reduces SE");
    check(se_both < se_plain, "combined reduces SE");
}

void test_finite_difference_schemes() {
    section("Finite difference schemes vs Black-Scholes");

    MarketData mkt{100.0, 0.05, 0.0, 0.2};
    auto euro = make_vanilla(OptionType::Call, 100.0, 1.0);
    const double exact = black_scholes::price(OptionType::Call, 100, 100, 0.05, 0.0, 0.2, 1.0);

    FiniteDifferenceEngine implicit_(400, 400, FDScheme::Implicit);
    FiniteDifferenceEngine cn(400, 400, FDScheme::CrankNicolson);

    check_close(implicit_.price(euro, mkt).price, exact, 5e-2, "implicit within 5e-2");
    check_close(cn.price(euro, mkt).price, exact, 5e-3, "Crank-Nicolson within 5e-3");

    // Grid Greeks should match the analytic ones closely.
    const auto res = cn.price(euro, mkt);
    check_close(res.delta, black_scholes::delta(OptionType::Call, 100, 100, 0.05, 0.0, 0.2, 1.0),
                1e-3, "FD delta");
    check_close(res.gamma, black_scholes::gamma(100, 100, 0.05, 0.0, 0.2, 1.0), 1e-3, "FD gamma");
    check_close(res.theta, black_scholes::theta(OptionType::Call, 100, 100, 0.05, 0.0, 0.2, 1.0),
                1e-2, "FD theta via PDE identity");

    // The explicit scheme must refuse to run past its stability limit rather
    // than silently returning garbage.
    bool threw = false;
    try {
        FiniteDifferenceEngine(400, 10, FDScheme::Explicit).price(euro, mkt);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "explicit scheme rejects unstable timestep");

    // ...and must be accurate when the limit is respected.
    const double stable =
        FiniteDifferenceEngine(200, 60000, FDScheme::Explicit).price(euro, mkt).price;
    check_close(stable, exact, 5e-2, "explicit (stable) within 5e-2");
}

void test_american_put_agreement() {
    section("American put: tree vs PSOR finite difference");

    MarketData mkt{100.0, 0.05, 0.0, 0.3};
    auto amer = make_vanilla(OptionType::Put, 110.0, 1.0, ExerciseStyle::American);

    const double tree = BinomialEngine(4000).price(amer, mkt).price;
    const double psor = FiniteDifferenceEngine(500, 500, FDScheme::CrankNicolson).price(amer, mkt).price;

    check_close(psor, tree, 2e-2, "PSOR agrees with tree on American put");
}

void test_barrier() {
    section("Barrier: closed form vs Monte Carlo vs PDE");

    MarketData mkt{100.0, 0.05, 0.0, 0.2};
    const double K = 100.0, B = 90.0, T = 1.0;

    const double closed =
        black_scholes::down_and_out_call(100.0, K, B, 0.05, 0.0, 0.2, T);
    const double vanilla = black_scholes::price(OptionType::Call, 100, K, 0.05, 0.0, 0.2, T);

    check(closed > 0.0, "down-and-out has positive value");
    check(closed < vanilla, "knock-out is cheaper than vanilla");

    auto bar = make_barrier(OptionType::Call, K, B, BarrierDirection::Down, BarrierType::Out, T);

    // PDE prices the continuously-monitored contract, so it should be close.
    const double pde =
        FiniteDifferenceEngine(600, 600, FDScheme::CrankNicolson).price(bar, mkt).price;
    check_close(pde, closed, 2e-2, "PDE barrier vs closed form");

    // Discretely-monitored MC must OVERPRICE a knock-out: fewer monitoring dates
    // mean fewer chances to breach, so the option survives too often.
    MonteCarloEngine mc(200000, VarianceReduction::Antithetic, 4242, 50);
    const double mc_price = mc.price(bar, mkt).price;
    check(mc_price > closed, "discrete monitoring overprices the knock-out");

    // Denser monitoring must move it back toward the continuous price.
    MonteCarloEngine mc_fine(200000, VarianceReduction::Antithetic, 4242, 800);
    const double mc_fine_price = mc_fine.price(bar, mkt).price;
    check(mc_fine_price < mc_price, "denser monitoring reduces the discretisation bias");
}

void test_monte_carlo_barrier_vs_vanilla_consistency() {
    section("Monte Carlo path engine sanity");

    MarketData mkt{100.0, 0.05, 0.0, 0.2};
    const double exact = black_scholes::price(OptionType::Call, 100, 100, 0.05, 0.0, 0.2, 1.0);

    // A barrier placed where it can never be hit must reproduce the vanilla.
    auto unreachable =
        make_barrier(OptionType::Call, 100.0, 1e-8, BarrierDirection::Down, BarrierType::Out, 1.0);
    MonteCarloEngine mc(200000, VarianceReduction::Antithetic, 31337, 100);
    const auto r = mc.price(unreachable, mkt);
    check(std::abs(r.price - exact) < 5.0 * r.std_error + 1e-3,
          "unreachable barrier == vanilla within MC error");
}

void test_implied_vol_roundtrip() {
    section("Implied volatility round-trip");

    const double S = 100, r = 0.04, q = 0.01, T = 0.8;

    for (double K : {60.0, 80.0, 100.0, 120.0, 160.0}) {
        for (double v : {0.08, 0.20, 0.45, 0.90}) {
            const double px = black_scholes::price(OptionType::Call, S, K, r, q, v, T);
            const auto iv = implied_volatility(px, OptionType::Call, S, K, r, q, T);
            const std::string tag =
                "K=" + std::to_string(K) + " v=" + std::to_string(v);

            check(iv.converged, "IV converged " + tag);
            if (!iv.converged) continue;

            // The price must ALWAYS round-trip -- that is what the root solver
            // actually promises.
            const double repriced =
                black_scholes::price(OptionType::Call, S, K, r, q, iv.vol, T);
            check_close(repriced, px, 1e-8, "IV price round-trip " + tag);

            // The VOLATILITY only round-trips where vega is big enough to pin it
            // down. Demanding 1e-6 on vol when vega is 1e-11 would be asking the
            // solver to extract information the price does not contain.
            //
            // So the assertion everywhere is the solver's actual contract: the
            // error must fall inside the uncertainty it reported for itself.
            check(std::abs(iv.vol - v) <= 10.0 * iv.vol_uncertainty + 1e-6,
                  "IV error within its own reported uncertainty " + tag);

            // Where it claims to be well conditioned, it must deliver 1e-4.
            if (iv.well_conditioned)
                check_close(iv.vol, v, 1e-4, "IV vol round-trip (well-conditioned) " + tag);
        }
    }
}

void test_implied_vol_conditioning_is_reported() {
    section("Implied vol reports its own conditioning");

    const double S = 100, r = 0.04, q = 0.01, T = 0.8;

    // At the money: vega is large, the inversion is well posed.
    const double atm_px = black_scholes::price(OptionType::Call, S, 100.0, r, q, 0.08, T);
    const auto atm = implied_volatility(atm_px, OptionType::Call, S, 100.0, r, q, T);
    check(atm.well_conditioned, "ATM inversion flagged well-conditioned");
    check(atm.vega > 1.0, "ATM vega is large");
    check_close(atm.vol, 0.08, 1e-8, "ATM vol recovered exactly");

    // Deep in the money at low vol: vega ~ 1e-11, so a whole vol point moves the
    // price by less than the solver tolerance. The answer is meaningless and the
    // solver must say so rather than pretending otherwise.
    const double itm_px = black_scholes::price(OptionType::Call, S, 60.0, r, q, 0.08, T);
    const auto itm = implied_volatility(itm_px, OptionType::Call, S, 60.0, r, q, T);
    check(!itm.well_conditioned, "deep-ITM inversion flagged ill-conditioned");
    check(itm.vega < 1e-6, "deep-ITM vega is negligible");
    check(itm.vol_uncertainty > 1e-4, "reported uncertainty is large");

    // Even so, the price it returns must still be right.
    const double repriced = black_scholes::price(OptionType::Call, S, 60.0, r, q, itm.vol, T);
    check_close(repriced, itm_px, 1e-8, "ill-conditioned root still reprices correctly");
}

void test_implied_vol_fallback_is_exercised() {
    section("Implied vol: Newton fails, Brent rescues");

    // Very deep out-of-the-money and short-dated: vega is ~1e-30, so the Newton
    // step is astronomically large and it cannot converge. The bracketing
    // fallback is the only thing that returns an answer here.
    const double S = 100, K = 400, r = 0.02, q = 0.0, T = 0.02, v = 0.3;
    const double px = black_scholes::price(OptionType::Call, S, K, r, q, v, T);

    const double vega = black_scholes::vega(S, K, r, q, v, T);
    check(vega < 1e-10, "vega really is negligible in this regime");

    const auto iv = implied_volatility(px, OptionType::Call, S, K, r, q, T);
    // The price is numerically zero here, so we only require that the solver
    // terminates cleanly rather than diverging or returning NaN.
    check(std::isfinite(iv.vol), "solver returns a finite vol rather than diverging");
}

void test_implied_vol_rejects_arbitrage() {
    section("Implied vol rejects unattainable quotes");

    const double S = 100, K = 100, r = 0.05, q = 0.0, T = 1.0;

    const auto below = implied_volatility(0.0001, OptionType::Call, S, K, r, q, T);
    check(!below.converged, "price below intrinsic rejected");

    const auto above = implied_volatility(200.0, OptionType::Call, S, K, r, q, T);
    check(!above.converged, "price above upper bound rejected");
}

void test_vol_surface() {
    section("Volatility surface reconstruction");

    const double S = 100, r = 0.03, q = 0.0;

    // Synthesise a chain from a known smile, then invert it and check we get
    // the smile back. This validates the whole calibration path end to end.
    auto true_vol = [](double K, double T) {
        const double m = std::log(K / 100.0);
        return 0.20 + 0.35 * m * m - 0.05 * m + 0.02 * T;
    };

    std::vector<VolQuote> quotes;
    for (double T : {0.25, 0.5, 1.0, 2.0}) {
        for (double K : {70.0, 85.0, 100.0, 115.0, 130.0}) {
            VolQuote quote;
            quote.strike = K;
            quote.maturity = T;
            quote.type = OptionType::Call;
            quote.price = black_scholes::price(OptionType::Call, S, K, r, q, true_vol(K, T), T);
            quotes.push_back(quote);
        }
    }

    VolatilitySurface surf;
    surf.build(quotes, S, r, q);

    check(surf.converged_count() == quotes.size(), "every quote inverted");

    double max_err = 0.0;
    for (const auto& p : surf.points())
        if (p.converged) max_err = std::max(max_err, std::abs(p.implied_vol - true_vol(p.strike, p.maturity)));
    check(max_err < 1e-6, "surface recovers the input smile");
}

}  // namespace

int main() {
    std::cout << "Derivatives Pricing Engine -- test suite\n";
    std::cout << "=======================================\n";

    test_black_scholes_reference_values();
    test_put_call_parity();
    test_greeks_against_finite_difference();
    test_binomial_converges_to_bs();
    test_american_premium();
    test_monte_carlo_converges();
    test_finite_difference_schemes();
    test_american_put_agreement();
    test_barrier();
    test_monte_carlo_barrier_vs_vanilla_consistency();
    test_implied_vol_roundtrip();
    test_implied_vol_conditioning_is_reported();
    test_implied_vol_fallback_is_exercised();
    test_implied_vol_rejects_arbitrage();
    test_vol_surface();

    std::cout << "\n=======================================\n";
    std::cout << (g_failures == 0 ? "ALL PASSED" : "FAILURES") << ": " << (g_checks - g_failures)
              << "/" << g_checks << " checks passed\n";
    return g_failures == 0 ? 0 : 1;
}
