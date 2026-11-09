#pragma once
#include "numi/matter/matter.hpp"
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace numi::matter::detail {
// Reference triangle coordinates and weights. Weights sum to one, so each
// physical contribution is referenceArea * weight * integrand. The positive
// seven-point rule integrates every triangle polynomial through degree five.
struct FEMPlaneQuadraturePoint {
    std::array<double, 3> barycentric;
    double weight;
};
inline constexpr std::array<FEMPlaneQuadraturePoint, 7> femPlaneQuadrature{{
    {{{1.0/3.0,1.0/3.0,1.0/3.0}},0.225},
    {{{0.059715871789770,0.470142064105115,0.470142064105115}},0.132394152788506},
    {{{0.470142064105115,0.059715871789770,0.470142064105115}},0.132394152788506},
    {{{0.470142064105115,0.470142064105115,0.059715871789770}},0.132394152788506},
    {{{0.797426985353087,0.101286507323456,0.101286507323456}},0.125939180544827},
    {{{0.101286507323456,0.797426985353087,0.101286507323456}},0.125939180544827},
    {{{0.101286507323456,0.101286507323456,0.797426985353087}},0.125939180544827},
}};
struct FEMPlaneReferenceTriangle {
    std::array<unsigned,3> nodes;
    double referenceArea;
};

// Reference-area mass-lumping of a *single planar* exposed face patch.
// Nodes which do not belong to the patch receive zero contact area.
// Reject arbitrary meshes: their projected patches require a full surface
// quadrature and are outside this deliberately narrow research owner.
inline bool femFixedPlaneReferenceAreas(const CompiledWorld& world,
        std::vector<float>& areas, std::string& error,
        std::vector<FEMPlaneReferenceTriangle>* triangles = nullptr) {
    if (world.contact.rigidProxies.size()!=1u) {
        error="fixed-area plane model requires exactly one plane";return false;
    }
    const auto &p=world.contact.rigidProxies[0];
    const double length=std::sqrt(double(p.localCenterAndRadius.x)*p.localCenterAndRadius.x+
        double(p.localCenterAndRadius.y)*p.localCenterAndRadius.y+
        double(p.localCenterAndRadius.z)*p.localCenterAndRadius.z);
    if (!(length>0) || !std::isfinite(length)) {
        error="invalid reference plane normal";return false;
    }
    using V=std::array<double,3>;
    const V normal{p.localCenterAndRadius.x/length,p.localCenterAndRadius.y/length,p.localCenterAndRadius.z/length};
    auto dot=[](V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];};
    auto sub=[](V a,V b){return V{a[0]-b[0],a[1]-b[1],a[2]-b[2]};};
    auto cross=[](V a,V b){return V{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};};
    const auto coord=[&](unsigned i){const auto& n=world.fem.nodes[i].restAndFixed;return V{n.x,n.y,n.z};};
    areas.assign(world.fem.nodes.size(),0.f);
    if (triangles) triangles->clear();
    double patchHeight=0;bool found=false;
    for(const auto& f:world.fem.surfaceFaces) {
        if(f.adjacency.y!=NM_INVALID_INDEX)continue;
        if(f.adjacency.x>=world.fem.tetrahedra.size()||f.sides.x>3u){error="invalid surface face";return false;}
        const auto t=world.fem.tetrahedra[f.adjacency.x].nodes;
        const unsigned ids[4]{t.x,t.y,t.z,t.w};
        for(auto id:ids)if(id>=areas.size()){error="invalid face node";return false;}
        unsigned face[3],n=0;for(unsigned i=0;i<4;++i)if(i!=f.sides.x)face[n++]=ids[i];
        const V a=coord(face[0]),b=coord(face[1]),c=coord(face[2]),opposite=coord(ids[f.sides.x]);
        V v=cross(sub(b,a),sub(c,a));double magnitude=std::sqrt(dot(v,v));
        if(!(magnitude>0)||!std::isfinite(magnitude)){error="degenerate surface triangle";return false;}
        if(dot(v,sub(opposite,a))>0)for(auto& x:v)x=-x;
        const double projected=-dot(v,normal);
        if(projected<=magnitude*1e-6)continue;
        if(projected<magnitude*(1.-1e-6)){error="contact patch must be parallel to the plane";return false;}
        for(auto id:face) {
            const auto x=coord(id);const double height=dot(x,normal);
            if(!found){patchHeight=height;found=true;}
            if(std::abs(height-patchHeight)>1e-7){error="contact patch must be coplanar";return false;}
            areas[id]+=float(projected/6.0);
        }
        if (triangles) triangles->push_back({{face[0],face[1],face[2]},projected/2.0});
    }
    if(!found){error="no plane-facing contact patch";return false;}
    for(float area:areas)if(!std::isfinite(area)||area<0){error="invalid quadrature area";return false;}
    return true;
}
} // namespace numi::matter::detail
