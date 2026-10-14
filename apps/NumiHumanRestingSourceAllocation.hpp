#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Source compilers retain unused ancestry slots. Recover the complete allocation
// from ordered source indices, rather than assuming the first rendered vertex
// is source vertex zero. No coordinates or deformation ownership are changed.
inline std::uint32_t numiHumanRestingSourceAllocationBase(
    const std::vector<std::uint32_t>& sourceIndices, std::size_t sourceFirstIndex,
    std::uint32_t sourceFirstVertex, std::uint32_t sourceVertexCount,
    const std::vector<std::uint32_t>& meshIndices, std::size_t meshFirstIndex,
    std::size_t indexCount, std::size_t meshVertexCount) {
    const auto requireRange=[](bool valid) {
        if(!valid)throw std::runtime_error("resting native topology differs from its source allocation");
    };
    requireRange(sourceVertexCount>0&&indexCount>0&&indexCount%3==0);
    requireRange(sourceFirstIndex<=sourceIndices.size()&&
        indexCount<=sourceIndices.size()-sourceFirstIndex&&
        meshFirstIndex<=meshIndices.size()&&indexCount<=meshIndices.size()-meshFirstIndex);
    const auto sourceFirst=sourceIndices[sourceFirstIndex];
    requireRange(sourceFirst>=sourceFirstVertex&&
        sourceFirst-sourceFirstVertex<sourceVertexCount);
    const auto offset=sourceFirst-sourceFirstVertex;
    const auto meshFirst=meshIndices[meshFirstIndex];
    requireRange(meshFirst>=offset);
    const auto base=meshFirst-offset;
    requireRange(base<=meshVertexCount&&sourceVertexCount<=meshVertexCount-base);
    for(std::size_t j=0;j<indexCount;++j) {
        const auto sourceIndex=sourceIndices[sourceFirstIndex+j];
        requireRange(sourceIndex>=sourceFirstVertex&&
            sourceIndex-sourceFirstVertex<sourceVertexCount);
        // Subtraction also avoids overflowing a constructed global vertex ID.
        const auto meshIndex=meshIndices[meshFirstIndex+j];
        requireRange(meshIndex>=base&&
            meshIndex-base==sourceIndex-sourceFirstVertex);
    }
    return base;
}
