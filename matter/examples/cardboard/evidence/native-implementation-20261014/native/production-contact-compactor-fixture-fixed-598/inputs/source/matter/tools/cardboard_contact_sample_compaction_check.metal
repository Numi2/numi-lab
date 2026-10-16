#include "../src/matter_kernels.metal"
namespace numi_matter_metal {
kernel void nm_contact_sample_compaction_fixture_append(
    constant NMMatterDispatchGPU& dispatch [[buffer(0)]],
    constant uint& activeCapacity [[buffer(1)]],
    device const uint* activeMask [[buffer(2)]],
    device const NMContactPairGPU* pairs [[buffer(3)]],
    device const NMContactSampleGPU* sourceSamples [[buffer(4)]],
    device NMContactSampleGPU* compactSamples [[buffer(5)]],
    device uint* sampleSlotsByPair [[buffer(6)]],
    device atomic_uint* activeCounts [[buffer(7)]],
    device NMMatterStatusGPU* statuses [[buffer(8)]],
    uint global [[thread_position_in_grid]]
) {
    const uint total = dispatch.environmentCount * dispatch.contactPairCount;
    if (global >= total || dispatch.contactPairCount == 0u) return;
    const uint environment = global / dispatch.contactPairCount;
    const uint pairIndex = global - environment * dispatch.contactPairCount;
    if (activeMask[global] == 0u) return;
    const NMContactSampleGPU sample = sourceSamples[global];
    const NMContactPairGPU pair = pairs[pairIndex];
    if ((sample.identity.w & NM_CONTACT_VALID) == 0u ||
        sample.identity.x != pair.continuumNode ||
        sample.identity.y != pair.rigidProxy ||
        sample.identity.z != pair.objectIndex) {
        recordFailure(statuses[environment], NM_STATUS_CONTACT_FAILURE,
            pair.objectIndex, pairIndex,
            float4(float(sample.identity.x), float(sample.identity.y),
                   float(sample.identity.z), float(sample.identity.w)));
        return;
    }
    const uint slot = atomic_fetch_add_explicit(
        activeCounts + environment, 1u, memory_order_relaxed);
    if (slot >= activeCapacity) {
        recordFailure(statuses[environment], NM_STATUS_CAPACITY_OVERFLOW,
            pair.objectIndex, pairIndex,
            float4(float(slot), float(activeCapacity),
                   float(dispatch.contactPairCount), 6.0f));
        return;
    }
    compactSamples[environment * activeCapacity + slot] = sample;
    sampleSlotsByPair[global] = slot;
}
}
