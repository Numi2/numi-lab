#pragma once
#include "metalrobo/MetalDiscreteElasticRod.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace rod_local_research {

// Experimental host-mediated adapter for the standalone uncoupled API.
// It does not participate in MetalWorld, contact, or resident state.
inline metalrobo::MetalDiscreteElasticRodDiagnostics run(
    const metalrobo::DiscreteElasticRodModel& model,
    const metalrobo::MetalDiscreteElasticRodInput& input,
    metalrobo::MetalDiscreteElasticRodResult& output,
    const metalrobo::MetalDiscreteElasticRodConfig& config) {
    using namespace metalrobo;
    MetalDiscreteElasticRodDiagnostics rejected;
    auto fail = [&](MetalDiscreteElasticRodHostStatus status, const char* message) {
        rejected.status=status; rejected.message=message; rejected.published=false;
        return rejected;
    };
    const std::size_t count=input.states.size();
    if (config.tool.enabled || config.step.enableSelfCollision ||
        input.rigidBodyCount!=0 || !input.rigidBodies.empty() || !input.rigidBindings.empty() ||
        input.toolModel!=nullptr || !input.toolPairs.empty() || !input.previousToolContacts.empty()) {
        return fail(MetalDiscreteElasticRodHostStatus::invalidConfiguration,
                    "local-frame research adapter does not support contact or rigid coupling");
    }
    if (count==0 || input.attachmentCount>std::numeric_limits<std::size_t>::max()/count ||
        input.attachments.size()!=count*input.attachmentCount) {
        return fail(MetalDiscreteElasticRodHostStatus::invalidAttachment,
                    "local-frame input packing is invalid");
    }
    std::vector<DiscreteElasticRodState> states(input.states.begin(),input.states.end());
    std::vector<DiscreteRodAttachment> attachments(input.attachments.begin(),input.attachments.end());
    std::vector<std::array<double,3>> origins(count);
    for (std::size_t env=0;env<count;++env) {
        if (states[env].positions.empty()) {
            return fail(MetalDiscreteElasticRodHostStatus::invalidState,"local-frame state is empty");
        }
        origins[env]=states[env].positions[0];
        for (double v:origins[env]) if (!std::isfinite(v))
            return fail(MetalDiscreteElasticRodHostStatus::invalidState,"local origin is non-finite");
        for (auto& p:states[env].positions) for (std::size_t k=0;k<3;++k) {
            p[k]-=origins[env][k];
            if (!std::isfinite(p[k]))
                return fail(MetalDiscreteElasticRodHostStatus::invalidState,"local position is non-finite");
        }
        for (std::size_t j=0;j<input.attachmentCount;++j) {
            auto& a=attachments[env*input.attachmentCount+j];
            for (std::size_t k=0;k<3;++k) {
                a.targetPosition[k]-=origins[env][k];
                if (!std::isfinite(a.targetPosition[k]))
                    return fail(MetalDiscreteElasticRodHostStatus::invalidAttachment,"local target is non-finite");
            }
        }
    }
    MetalDiscreteElasticRodInput localInput{};
    localInput.states=states;
    localInput.attachmentCount=input.attachmentCount;
    localInput.attachments=attachments;
    MetalDiscreteElasticRodResult staged;
    auto d=runMetalDiscreteElasticRod(model,localInput,staged,config);
    if (!d.succeeded() || !d.published) return d;
    if (staged.states.size()!=count) {
        d.status=MetalDiscreteElasticRodHostStatus::internalFailure;
        d.message="local-frame result has invalid environment count"; d.published=false; return d;
    }
    for (std::size_t env=0;env<count;++env)
        for (auto& p:staged.states[env].positions)
            for (std::size_t k=0;k<3;++k) {
                p[k]+=origins[env][k];
                if (!std::isfinite(p[k])) {
                    d.status=MetalDiscreteElasticRodHostStatus::internalFailure;
                    d.message="world reconstruction is non-finite"; d.published=false; return d;
                }
            }
    output=std::move(staged);
    return d;
}
}
