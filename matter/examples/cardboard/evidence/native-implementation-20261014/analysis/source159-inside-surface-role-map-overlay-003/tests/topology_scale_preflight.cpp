#include "cardboard_inside_clearance.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace numi::cardboard::inside_clearance_detail;

void require(bool ok,const char* message) {
    if (!ok) throw std::runtime_error(message);
}

std::array<std::uint32_t,3> sortedFace(const std::array<std::uint32_t,4>& ids,
                                      const std::uint32_t opposite) {
    std::array<std::uint32_t,3> key{};
    std::size_t out=0u;
    for (std::uint32_t i=0u;i<4u;++i) if (i!=opposite) key[out++]=ids[i];
    std::sort(key.begin(),key.end());
    return key;
}

void run() {
    constexpr std::uint32_t nodeCount=528444u;
    constexpr std::uint32_t planarVertexCount=263720u;
    constexpr std::uint32_t boundaryVertexCount=34862u;
    constexpr std::uint32_t interiorVertexCount=planarVertexCount-boundaryVertexCount;
    constexpr std::uint32_t planarTriangleCount=492576u;
    constexpr std::uint32_t tetrahedronCount=1477728u;
    constexpr std::uint32_t surfaceFaceCount=1054876u;
    static_assert(boundaryVertexCount>=3u);
    static_assert(planarTriangleCount==boundaryVertexCount-2u+2u*interiorVertexCount);
    static_assert(tetrahedronCount==3u*planarTriangleCount);
    static_assert(surfaceFaceCount==2u*planarTriangleCount+2u*boundaryVertexCount);
    static_assert(2u*planarVertexCount<=nodeCount);

    // Build a deterministic combinatorial triangulation of a disk: a boundary
    // fan followed by interior-vertex insertion into existing triangles. This
    // is a topology/workload fixture only; it carries no spatial coordinates,
    // authored blank roles, accepted state, or physical qualification.
    std::vector<std::array<std::uint32_t,3>> active;
    active.reserve(planarTriangleCount);
    for (std::uint32_t i=1u;i+1u<boundaryVertexCount;++i)
        active.push_back({0u,i,i+1u});
    for (std::uint32_t v=boundaryVertexCount;v<planarVertexCount;++v) {
        require(!active.empty(),"triangulation insertion exhausted active faces");
        const auto old=active.back();
        active.pop_back();
        active.push_back({old[0],old[1],v});
        active.push_back({old[1],old[2],v});
        active.push_back({old[2],old[0],v});
    }
    require(active.size()==planarTriangleCount,"planar topology count mismatch");

    std::vector<NMTetrahedronGPU> tets;
    tets.reserve(tetrahedronCount);
    for (const auto& triangle:active) {
        auto sortedTriangle=triangle;
        std::sort(sortedTriangle.begin(),sortedTriangle.end());
        const std::uint32_t a=sortedTriangle[0],b=sortedTriangle[1],c=sortedTriangle[2];
        const std::array<std::array<std::uint32_t,4>,3> prism{{
            {{a,b,c,c+planarVertexCount}},
            {{a,b,b+planarVertexCount,c+planarVertexCount}},
            {{a,a+planarVertexCount,b+planarVertexCount,c+planarVertexCount}}
        }};
        for (const auto& ids:prism) {
            auto sorted=ids;
            std::sort(sorted.begin(),sorted.end());
            require(std::adjacent_find(sorted.begin(),sorted.end())==sorted.end(),
                    "generated prism tetrahedron repeats a node");
            NMTetrahedronGPU tet{};
            tet.nodes={ids[0],ids[1],ids[2],ids[3]};
            tet.identity={0u,0u,0u,0u};
            tets.push_back(tet);
        }
    }
    require(tets.size()==tetrahedronCount,"tetrahedron construction count mismatch");

    std::uint64_t hashCapacity=0u,hashBytes=0u,workspaceBytes=0u;
    require(ExteriorFaceIndex::capacityFor(tets.size(),hashCapacity,hashBytes),
            "nominal tetrahedron count rejected by checked topology index");
    require(estimateWorkspaceBytes(hashBytes,nodeCount,surfaceFaceCount,workspaceBytes),
            "nominal topology/workspace estimate exceeds the declared 1 GiB cap");
    ExteriorFaceIndex index(hashCapacity);
    for (std::uint32_t ti=0u;ti<tets.size();++ti) {
        const auto& tet=tets[ti];
        const std::array<std::uint32_t,4> ids{tet.nodes.x,tet.nodes.y,tet.nodes.z,tet.nodes.w};
        for (std::uint32_t opposite=0u;opposite<4u;++opposite) {
            const auto key=sortedFace(ids,opposite);
            require(index.insert(key,ti,opposite),"nominal tetrahedron face insertion exceeded bounds");
        }
    }
    std::vector<NMFEMSurfaceFaceGPU> faces;
    faces.reserve(surfaceFaceCount);
    require(index.forEachExterior([&](const FaceKey&,const BoundaryRef& ref) {
        NMFEMSurfaceFaceGPU face{};
        face.adjacency={ref.tet,NM_INVALID_INDEX,0u,NM_TOPOLOGY_ACTIVE};
        face.sides={ref.opposite,0u,static_cast<std::uint32_t>(faces.size()),0u};
        faces.push_back(face);
        return faces.size()<=surfaceFaceCount;
    }),"exterior face enumeration exceeded its checked face count");
    require(faces.size()==surfaceFaceCount,"exterior surface count differs from disk-extrusion identity");
    for (const auto& face:faces) {
        require(face.adjacency.x<tets.size()&&face.sides.x<4u&&
                face.adjacency.y==NM_INVALID_INDEX&&face.adjacency.z==0u&&
                (face.adjacency.w&NM_TOPOLOGY_ACTIVE)!=0u,
                "generated surface record has invalid object/tet/face roles");
        const auto& tet=tets[face.adjacency.x];
        const std::array<std::uint32_t,4> ids{tet.nodes.x,tet.nodes.y,tet.nodes.z,tet.nodes.w};
        const auto key=sortedFace(ids,face.sides.x);
        BoundaryRef ref{};
        require(index.markExteriorSeen(key,face.adjacency.x,face.sides.x,ref),
                "generated surface record failed exact tet/opposite-corner mapping");
    }
    require(index.allExteriorFacesSeen(),"surface fixture omitted an exterior topological face");

    const std::uint64_t residentFixtureBytes=
        static_cast<std::uint64_t>(tets.capacity())*sizeof(NMTetrahedronGPU)+
        static_cast<std::uint64_t>(faces.capacity())*sizeof(NMFEMSurfaceFaceGPU)+
        static_cast<std::uint64_t>(active.capacity())*sizeof(active[0])+hashBytes;
    std::cout<<"{\"schema\":\"numi.cardboard.inside_topology_scale_preflight.v1\","
             <<"\"fixture_kind\":\"synthetic_combinatorial_disk_extrusion_no_geometry\","
             <<"\"node_count\":"<<nodeCount<<",\"used_node_count\":"<<2u*planarVertexCount<<"," 
             <<"\"planar_boundary_vertices\":"<<boundaryVertexCount<<","
             <<"\"planar_triangles\":"<<planarTriangleCount<<","
             <<"\"tetrahedra\":"<<tets.size()<<",\"surface_faces\":"<<faces.size()<<"," 
             <<"\"hash_capacity\":"<<hashCapacity<<",\"hash_bytes\":"<<hashBytes<<"," 
             <<"\"workspace_estimate_bytes\":"<<workspaceBytes<<"," 
             <<"\"fixture_resident_bytes\":"<<residentFixtureBytes<<"," 
             <<"\"topology_probe_steps\":"<<index.probeSteps()<<"," 
             <<"\"surface_mappings_verified\":"<<faces.size()<<"," 
             <<"\"all_exterior_faces_seen\":true,\"geometry_or_material_evaluated\":false}\n";
}
}

int main() {
    try { run(); return 0; }
    catch (const std::exception& e) {
        std::cerr<<"topology scale preflight: FAIL: "<<e.what()<<'\n';
        return 1;
    }
}
