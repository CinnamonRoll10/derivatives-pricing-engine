#pragma once

#include <string>

#include "dpe/instrument.hpp"
#include "dpe/types.hpp"

namespace dpe {

/// Strategy interface for "turn an instrument + market into a price".
///
/// Every numerical method implements this, so the convergence study can hold a
/// vector<unique_ptr<PricingEngine>> and benchmark them against one another
/// without any method-specific code at the call site.
class PricingEngine {
public:
    virtual ~PricingEngine() = default;

    virtual PricingResult price(const Instrument& instrument, const MarketData& market) const = 0;

    virtual std::string name() const = 0;

    /// Engines advertise what they can handle; callers check before pricing
    /// rather than discovering a silent wrong answer. The analytic engine, for
    /// instance, cannot price an American put or a generic path-dependent claim.
    virtual bool supports(const Instrument& instrument) const = 0;
};

}  // namespace dpe
