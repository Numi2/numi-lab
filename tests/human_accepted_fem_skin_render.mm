#include "NumiHumanAcceptedFEMSkinRenderPass.hpp"
void check(numi::human::AcceptedFEMSkinRenderPass& pass,
           const metalrobo::MetalHybridComputeEncoderCallbacks& callbacks,
           void* mesh) {
    (void)pass.encode(callbacks, mesh, 0u);
}
