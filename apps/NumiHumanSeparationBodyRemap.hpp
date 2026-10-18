#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace metalrobo::separation {

inline std::uint32_t mapSourceBodyIndex(
    const std::vector<std::uint32_t>& map, const std::uint32_t sourceBody) {
    if (sourceBody == MR_INVALID_INDEX) return sourceBody;
    if (map.empty()) return sourceBody;
    if (sourceBody >= map.size())
        throw std::runtime_error("separation body index escapes the source-to-runtime map");
    return map[sourceBody];
}

template<class Rigid, class Muscles, class Bones, class Hood,
         class Tissues, class Skin, class Anatomy, class Contacts>
void remapLoadedBodyReferences(
    const std::vector<std::uint32_t>& map, Rigid& rigid, Muscles& muscles,
    Bones* bones, Hood* hood, Tissues* tissues, Skin* skin, Anatomy* anatomy,
    Contacts* contacts) {
    if (map.empty()) throw std::runtime_error("separation body map is empty");
    const auto apply = [&map](std::uint32_t& index) {
        index = mapSourceBodyIndex(map, index);
    };
    for (auto& index : rigid.sourceToCore) apply(index);
    for (auto& site : muscles.sites) apply(site.bodyIndex);
    for (auto& wrap : muscles.wraps) apply(wrap.bodyIndex);
    for (auto& site : muscles.referenceSites) apply(site.bodyIndex);
    for (auto& wrap : muscles.referenceWraps) apply(wrap.bodyIndex);
    for (auto& site : muscles.gpuSites) apply(site.bodyIndex);
    for (auto& wrap : muscles.gpuWraps) apply(wrap.bodyIndex);
    for (auto& binding : muscles.tendonPayload.bindings) apply(binding.bodyIndex);
    for (auto& triangle : muscles.tendonPayload.triangles) apply(triangle.bodyIndex);
    for (auto& envelope : muscles.tendonPayload.envelopes) apply(envelope.bodyIndex);
    if (bones)
        for (auto& record : bones->records) apply(record.bodyIndex);
    if (hood) {
        for (auto& node : hood->nodes) apply(node.bodyIndex);
        for (auto& input : hood->inputs) apply(input.proximalBodyIndex);
    }
    if (tissues) {
        for (auto& record : tissues->records)
            for (std::uint32_t binding = 0u; binding < record.bindingCount; ++binding)
                apply(record.bodyIndex[binding]);
#ifdef NUMI_HUMAN_RESTING_SCENE
        const auto remapInfluenceMap = [&apply](auto& influenceMap) {
            for (auto& [stableId, vertices] : influenceMap) {
                (void)stableId;
                for (auto& vertex : vertices)
                    for (auto& influence : vertex) apply(influence.body.x);
            }
        };
        remapInfluenceMap(tissues->parentInfluences);
        remapInfluenceMap(tissues->anatomyParentInfluences);
#endif
    }
    if (skin)
        for (auto& binding : skin->bindings) apply(binding.bodyIndex);
    if (anatomy)
        for (auto& record : anatomy->records) apply(record.bodyIndex);
    if (contacts)
        for (auto& record : contacts->records) apply(record.bodyIndex);
}

} // namespace metalrobo::separation
