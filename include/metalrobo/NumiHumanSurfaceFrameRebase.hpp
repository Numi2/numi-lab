#pragma once
#include "metalrobo/NumiHumanTissueMass.hpp"
#include <array>
#include <cmath>
#include <map>
#include <span>
#include <stdexcept>
#include <vector>

// Load-time frame conversion only. Source geometry and hashes remain source
// identities; the caller fingerprints the executed mass partitions separately.
// Rotations, normals, anatomical IDs, topology and skin weights do not change.
template<class Bones, class Skin, class Tissues, class Anatomy>
void rebaseNumiHumanSurfaceFrames(
    Bones* bones, Skin* skin, Tissues* tissues, Anatomy* anatomy,
    std::span<const metalrobo::NumiHumanTissueMassPartition> partitions) {
    std::map<unsigned,std::array<double,3>> offsets;
    for (const auto& p : partitions) {
        for (double x:p.remainingCOMOffsetM)
            if (!std::isfinite(x)) throw std::runtime_error("nonfinite surface COM rebase");
        if (!offsets.emplace(p.donorBody,p.remainingCOMOffsetM).second)
            throw std::runtime_error("repeated surface COM donor");
    }
    const auto shift=[&](unsigned body,float& x,float& y,float& z) {
        const auto i=offsets.find(body); if(i==offsets.end())return;
        x=float(double(x)-i->second[0]);
        y=float(double(y)-i->second[1]);
        z=float(double(z)-i->second[2]);
    };
    if(bones) for(auto& b:bones->records)
        shift(b.bodyIndex,b.translationX,b.translationY,b.translationZ);
    if(skin) for(auto& b:skin->bindings)
        shift(b.bodyIndex,b.translationX,b.translationY,b.translationZ);
    if(tissues) {
        for(auto& r:tissues->records) for(unsigned b=0;b<r.bindingCount;++b)
            shift(r.bodyIndex[b],r.binding[b].translation[0],
                  r.binding[b].translation[1],r.binding[b].translation[2]);
        if constexpr(requires{ tissues->parentInfluences; tissues->anatomyParentInfluences; }) {
            const auto parents=[&](auto& surfaces) {
                for(auto& [id,vertices]:surfaces) {
                    (void)id;
                    for(auto& terms:vertices) for(auto& term:terms)
                        shift(term.body.x,term.positionAndWeight.x,
                              term.positionAndWeight.y,term.positionAndWeight.z);
                }
            };
            parents(tissues->parentInfluences);
            parents(tissues->anatomyParentInfluences);
        }
    }
    if(anatomy) {
        // A source allocation can be referenced by more than one display row.
        // Convert it once and reject ambiguous shared-body ownership.
        std::map<unsigned,unsigned> owners;
        for(const auto& r:anatomy->records) {
            if(r.firstVertex>anatomy->vertices.size() ||
               r.vertexCount>anatomy->vertices.size()-r.firstVertex)
                throw std::runtime_error("anatomy COM rebase vertex range");
            for(unsigned v=r.firstVertex;v<r.firstVertex+r.vertexCount;++v) {
                const auto [i,inserted]=owners.emplace(v,r.bodyIndex);
                if(!inserted&&i->second!=r.bodyIndex)
                    throw std::runtime_error("anatomy COM rebase ambiguous vertex body");
            }
        }
        for(const auto& [v,body]:owners) {
            auto& x=anatomy->vertices[v];
            shift(body,x.positionX,x.positionY,x.positionZ);
        }
    }
}
