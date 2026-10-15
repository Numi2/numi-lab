#include "cardboard_shipping_closure_mesh.hpp"
#include "cardboard_shipping_protocol.hpp"
#include "cardboard_shipping_driver.hpp"
#include "cardboard_starch_interface_mesh.hpp"
#include "numi/matter/mortar_shared_edge_guard.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <map>
#include <limits>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace numi::cardboard;
namespace {
constexpr float kClipTolerance = 1.0e-7f;
constexpr float kAreaFloor = 1.0e-10f;
constexpr float kOverlapTolerance = 1.0e-10f;
constexpr std::uint32_t kMaxCells = 64u;
struct V2 { float x=0, y=0; };
struct L2 { long double x=0, y=0; };
struct V3 { float x=0, y=0, z=0; };
using FaceMap = std::array<V3,3>;
using Pair = std::pair<std::uint32_t,std::uint32_t>;
struct Record {
    Pair pair{};
    std::uint32_t stable=0;
    std::string source;
    std::uint32_t sourceIndex=0;
    FaceMap left{};
    FaceMap right{};
};
struct Cell { std::array<V2,3> p{}; bool covered=false; std::uint32_t patch=0; };
struct Partition { std::vector<Cell> cells; std::string error; };
float cross2(V2 a,V2 b) { return a.x*b.y-a.y*b.x; }
long double cross2(L2 a,L2 b) { return a.x*b.y-a.y*b.x; }
V2 sub(V2 a,V2 b) { return {a.x-b.x,a.y-b.y}; }
float area2(const std::vector<V3>& p) {
    float s=0;
    for (std::size_t i=0;i<p.size();++i) {
        const auto& a=p[i]; const auto& b=p[(i+1)%p.size()];
        s += cross2(V2{a.x,a.y},V2{b.x,b.y});
    }
    return s;
}
V3 bary(V2 p) { return {p.x,p.y,1.0f-p.x-p.y}; }
float signedArea(const FaceMap& t) {
    return cross2(V2{t[1].x-t[0].x,t[1].y-t[0].y},
                  V2{t[2].x-t[0].x,t[2].y-t[0].y});
}
std::vector<V3> clip(const std::vector<V3>& input,V3 e0,V3 e1,
                     float orient,bool inside,std::string& error) {
    std::vector<V3> out;
    if (input.empty()) return out;
    auto side=[&](V3 p) { return orient*cross2(
        V2{e1.x-e0.x,e1.y-e0.y},V2{p.x-e0.x,p.y-e0.y}); };
    for (std::size_t i=0;i<input.size();++i) {
        const V3 a=input[i], b=input[(i+1)%input.size()];
        const float sa=side(a), sb=side(b);
        const bool ai=inside ? sa>=-kClipTolerance : sa<=kClipTolerance;
        const bool bi=inside ? sb>=-kClipTolerance : sb<=kClipTolerance;
        if (ai) out.push_back(a);
        if (ai!=bi) {
            if (out.size()>=6u) { error="clip_output_over_6"; return {}; }
            const float denominator=sa-sb;
            if (std::abs(denominator)<=1.0e-20f) {
                error="clip_denominator_floor"; return {};
            }
            const float f=std::clamp(sa/denominator,0.0f,1.0f);
            const V2 xy{a.x+f*(b.x-a.x),a.y+f*(b.y-a.y)};
            out.push_back(bary(xy));
        }
        if (out.size()>6u) { error="clip_output_over_6"; return {}; }
    }
    return out;
}
float intersectionArea(const FaceMap& a,const FaceMap& b,std::string& err) {
    const float orient=signedArea(b);
    if (!std::isfinite(orient)||std::abs(orient)<=kAreaFloor) {
        err="invalid_prior_or_current_patch_area"; return INFINITY;
    }
    std::vector<V3> poly(a.begin(),a.end());
    for (std::uint32_t e=0;e<3u;++e) {
        poly=clip(poly,b[e],b[(e+1u)%3u],orient>0?1.0f:-1.0f,true,err);
        if (!err.empty()) return INFINITY;
        if (poly.size()<3u) return 0.0f;
    }
    return 0.5f*std::abs(area2(poly));
}
// Independent geometry classification for diagnostics only: preserve the
// exact promoted FP32 map coordinates but clip with long-double arithmetic and
// no geometric epsilon. This is not the native acceptance predicate.
long double preciseIntersectionArea(const FaceMap& a,const FaceMap& b) {
    std::vector<L2> polygon;
    for (const auto& p:a) polygon.push_back({p.x,p.y});
    const auto orient=[](L2 x,L2 y,L2 z) {
        return cross2(L2{y.x-x.x,y.y-x.y},L2{z.x-x.x,z.y-x.y});
    };
    const long double signedOrientation=orient(
        {b[0].x,b[0].y},{b[1].x,b[1].y},{b[2].x,b[2].y});
    if (signedOrientation==0.0L)
        return std::numeric_limits<long double>::infinity();
    const long double sign=signedOrientation>0.0L?1.0L:-1.0L;
    for (std::size_t edgeIndex=0u;edgeIndex<3u && !polygon.empty();++edgeIndex) {
        const L2 e0{b[edgeIndex].x,b[edgeIndex].y};
        const L2 e1{b[(edgeIndex+1u)%3u].x,b[(edgeIndex+1u)%3u].y};
        const L2 edge{e1.x-e0.x,e1.y-e0.y};
        std::vector<L2> output;
        for (std::size_t i=0u;i<polygon.size();++i) {
            const L2 p=polygon[i], q=polygon[(i+1u)%polygon.size()];
            const long double sp=sign*cross2(edge,{p.x-e0.x,p.y-e0.y});
            const long double sq=sign*cross2(edge,{q.x-e0.x,q.y-e0.y});
            const bool pInside=sp>=0.0L, qInside=sq>=0.0L;
            if (pInside) output.push_back(p);
            if (pInside!=qInside) {
                const long double denominator=sp-sq;
                if (denominator==0.0L)
                    return std::numeric_limits<long double>::infinity();
                const long double t=sp/denominator;
                output.push_back({p.x+t*(q.x-p.x),p.y+t*(q.y-p.y)});
            }
        }
        polygon=std::move(output);
    }
    if (polygon.size()<3u) return 0.0L;
    long double twiceArea=0.0L;
    for (std::size_t i=0u;i<polygon.size();++i)
        twiceArea+=cross2(polygon[i],polygon[(i+1u)%polygon.size()]);
    return 0.5L*std::abs(twiceArea);
}
NMMortarGuardPoint3 guardPoint(const V3& p) { return {p.x,p.y,p.z}; }
bool exactOppositeSharedEdge(const FaceMap& a,const FaceMap& b) {
    return nmMortarExactOppositeSharedEdge(
        guardPoint(a[0]),guardPoint(a[1]),guardPoint(a[2]),
        guardPoint(b[0]),guardPoint(b[1]),guardPoint(b[2]));
}
bool appendTriangulation(Partition& out,const std::vector<V3>& poly,
                         bool covered,std::uint32_t patch) {
    if (poly.size()<3u) return true;
    if (poly.size()>6u) { out.error="triangulation_polygon_over_6"; return false; }
    for (std::size_t v=1;v+1<poly.size();++v) {
        const float a=cross2(V2{poly[v].x-poly[0].x,poly[v].y-poly[0].y},
                             V2{poly[v+1].x-poly[0].x,poly[v+1].y-poly[0].y});
        if (!std::isfinite(a)) { out.error="nonfinite_triangle_area"; return false; }
        if (std::abs(a)<=kAreaFloor) continue;
        if (out.cells.size()>=kMaxCells) { out.error="partition_capacity_64"; return false; }
        Cell c{};
        c.covered=covered; c.patch=covered?patch:UINT32_MAX;
        c.p[0]={poly[0].x,poly[0].y};
        if (a>0) { c.p[1]={poly[v].x,poly[v].y}; c.p[2]={poly[v+1].x,poly[v+1].y}; }
        else { c.p[1]={poly[v+1].x,poly[v+1].y}; c.p[2]={poly[v].x,poly[v].y}; }
        out.cells.push_back(c);
    }
    return true;
}
Partition buildPartition(const std::vector<FaceMap>& patches,
                        const std::vector<std::uint32_t>& ids) {
    Partition part;
    Cell parent{}; parent.p={V2{1,0},V2{0,1},V2{0,0}};
    part.cells.push_back(parent);
    for (std::size_t pi=0;pi<patches.size();++pi) {
        const auto& patch=patches[pi];
        const float sa=signedArea(patch);
        if (!std::isfinite(sa)||std::abs(sa)<=kAreaFloor) {
            part.error="patch_area_invalid"; return part;
        }
        for (const V3 q:patch) {
            if (!std::isfinite(q.x)||!std::isfinite(q.y)||!std::isfinite(q.z)||
                q.x < -1.0e-5f || q.y < -1.0e-5f || q.z < -1.0e-5f ||
                q.x > 1.00001f || q.y > 1.00001f || q.z > 1.00001f ||
                std::abs((q.x+q.y)+q.z-1.0f)>1.0e-4f) {
                part.error="patch_barycentric_range_or_sum"; return part;
            }
        }
        for (std::size_t prior=0;prior<pi;++prior) {
            std::string err;
            const float overlap=intersectionArea(patches[prior],patch,err);
            if (!err.empty()||!std::isfinite(overlap)||overlap>kOverlapTolerance) {
                part.error=err.empty()?"patch_pair_overlap":err; return part;
            }
        }
        Partition next;
        for (const auto& c:part.cells) if (c.covered) {
            if (next.cells.size()>=kMaxCells) { part.error="partition_capacity_64"; return part; }
            next.cells.push_back(c);
        }
        std::vector<V3> covered(patch.begin(),patch.end());
        if (!appendTriangulation(next,covered,true,ids[pi])) return next;
        for (const auto& source:part.cells) {
            if (source.covered) continue;
            std::vector<V3> inside={bary(source.p[0]),bary(source.p[1]),bary(source.p[2])};
            const float orient=sa>0?1.0f:-1.0f;
            for (std::uint32_t edge=0;edge<3u;++edge) {
                std::string err;
                auto outside=clip(inside,patch[edge],patch[(edge+1u)%3u],orient,false,err);
                if (!err.empty()) { next.error=err; return next; }
                if (!appendTriangulation(next,outside,false,UINT32_MAX)) return next;
                inside=clip(inside,patch[edge],patch[(edge+1u)%3u],orient,true,err);
                if (!err.empty()) { next.error=err; return next; }
                if (inside.size()<3u) break;
            }
        }
        if (next.cells.empty()||next.cells.size()>kMaxCells) {
            next.error="partition_empty_or_overflow"; return next;
        }
        part=std::move(next);
    }
    return part;
}
float totalArea(const Partition& p) {
    float sum=0;
    for (const auto& c:p.cells)
        sum += 0.5f*std::abs(cross2(sub(c.p[1],c.p[0]),sub(c.p[2],c.p[0])));
    return sum;
}
std::uint32_t opposite(const std::array<std::uint32_t,4>& tet,
                       const std::array<std::uint32_t,3>& face) {
    auto f=face; std::sort(f.begin(),f.end());
    for (std::uint32_t o=0;o<4;++o) {
        std::array<std::uint32_t,3> c{}; std::uint32_t k=0;
        for (std::uint32_t i=0;i<4;++i) if (i!=o) c[k++]=tet[i];
        std::sort(c.begin(),c.end()); if (c==f) return o;
    }
    throw std::runtime_error("face nodes do not match tetrahedron");
}
std::array<std::uint32_t,3> primitiveNodes(const std::array<std::uint32_t,4>& tet,
                                          std::uint32_t o) {
    std::array<std::uint32_t,3> out{}; std::size_t j=0;
    for (std::uint32_t i=0;i<4;++i) if (i!=o) out[j++]=tet[i];
    return out;
}
std::array<double,3> mapped(const std::array<std::uint32_t,3>& nodes,
                            const std::array<std::array<double,3>,3>& map,
                            std::size_t v,const std::vector<BoxBlankPoint>& points) {
    std::array<double,3> x{};
    for (std::size_t c=0;c<3;++c) {
        x[0]+=map[v][c]*points.at(nodes[c]).x;
        x[1]+=map[v][c]*points.at(nodes[c]).y;
        x[2]+=map[v][c]*points.at(nodes[c]).z;
    }
    return x;
}
std::array<double,3> sub3(std::array<double,3>a,std::array<double,3>b) {
    return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};
}
std::array<double,3> cross3(std::array<double,3>a,std::array<double,3>b) {
    return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}
double dot3(std::array<double,3>a,std::array<double,3>b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
std::array<std::array<double,3>,6> orientMaps(
    const std::array<std::uint32_t,3>& aNodes,const std::array<std::uint32_t,3>& bNodes,
    std::array<std::array<double,3>,3> aMap,std::array<std::array<double,3>,3> bMap,
    const std::vector<BoxBlankPoint>& points) {
    std::array<std::array<double,3>,3> a{},b{};
    for (std::size_t v=0;v<3;++v) { a[v]=mapped(aNodes,aMap,v,points); b[v]=mapped(bNodes,bMap,v,points); }
    auto normal=cross3(sub3(a[1],a[0]),sub3(a[2],a[0]));
    std::array<double,3> ca{},cb{};
    for (std::size_t v=0;v<3;++v) for (std::size_t k=0;k<3;++k) {
        ca[k]+=a[v][k]/3.0; cb[k]+=b[v][k]/3.0;
    }
    if (dot3(normal,sub3(cb,ca))<0.0) {
        std::swap(aMap[1],aMap[2]); std::swap(bMap[1],bMap[2]);
    }
    return {aMap[0],aMap[1],aMap[2],bMap[0],bMap[1],bMap[2]};
}
FaceMap toPrimitiveMap(const std::array<std::uint32_t,3>& faceNodes,
                       const std::array<std::array<double,3>,3>& faceMap,
                       const std::array<std::uint32_t,3>& primitive) {
    FaceMap out{};
    for (std::size_t v=0;v<3;++v) {
        std::array<float,3> src{};
        for (std::size_t c=0;c<3;++c) src[c]=static_cast<float>(faceMap[v][c]);
        std::array<float,3> barycentric{};
        for (std::size_t fc=0;fc<3;++fc) {
            bool found=false;
            for (std::size_t pc=0;pc<3;++pc) if (primitive[pc]==faceNodes[fc]) {
                barycentric[pc]=src[fc]; found=true; break;
            }
            if (!found) throw std::runtime_error("mortar face node absent from primitive");
        }
        out[v]={barycentric[0],barycentric[1],barycentric[2]};
    }
    return out;
}
}

int main() {
    ShippingBoxDesignDefaults defaults; defaults.subdivisionsPerBand=16u;
    const std::array<double,3> zero{0.,0.,0.};
    const auto selected=selectShippingDesign(defaults,nullptr,&zero,false);
    ShippingCorrugatedRecipe recipe; recipe.nxPerPitch=8u; recipe.thicknessSlices=1u;
    const auto geometry=buildShippingBoxBlankGeometry(selected.design,recipe);
    StarchInterfaceMesh starchSource;
    for (const auto& p:geometry.blank.points) starchSource.referenceNodesM.push_back({p.x,p.y,p.z});
    for (const auto& st:geometry.blank.tetrahedra) {
        StarchInterfaceTet t;
        for (std::size_t i=0;i<4;++i) t.nodeIds[i]=st.nodes[i];
        t.materialIndex=static_cast<std::uint8_t>(st.material);
        t.densityKgPerM3=t.materialIndex==kStarchMaterialLiner?766.0:(t.materialIndex==kStarchMaterialMedium?679.0:1500.0);
        starchSource.tetrahedra.push_back(t);
    }
    const auto starch=prepareStarchInterfaceMesh(starchSource);
    std::vector<BoxBlankPoint> nodes;
    for (const auto& p:starch.referenceNodesM) nodes.push_back({p.x,p.y,p.z});
    std::vector<ShippingClosureTet> tets;
    for (const auto& st:starch.tetrahedra) {
        ShippingClosureTet t; for (std::size_t i=0;i<4;++i) t.nodes[i]=static_cast<std::uint32_t>(st.nodeIds[i]);
        t.materialIndex=st.materialIndex; tets.push_back(t);
    }
    const auto hma=buildShippingClosureMortarMesh(geometry.blank,nodes,tets,0.010,0.0001,1u,
        static_cast<std::uint32_t>(starch.pairedFaces.size()+1u));
    std::map<Pair,std::vector<Record>> groups;
    std::uint32_t stable=1u;
    for (std::uint32_t i=0;i<starch.pairedFaces.size();++i,++stable) {
        const auto& s=starch.pairedFaces[i];
        const auto& aTet=tets.at(s.paperTetId).nodes; const auto& bTet=tets.at(s.glueTetId).nodes;
        std::array<std::uint32_t,3> aFace{},bFace{};
        for (std::size_t c=0;c<3;++c) { aFace[c]=static_cast<std::uint32_t>(s.paperNodeIds[c]); bFace[c]=static_cast<std::uint32_t>(s.glueNodeIds[c]); }
        const auto pa=4u*static_cast<std::uint32_t>(s.paperTetId)+opposite(aTet,aFace);
        const auto pb=4u*static_cast<std::uint32_t>(s.glueTetId)+opposite(bTet,bFace);
        std::array<std::array<double,3>,3> identity{}; for (std::size_t v=0;v<3;++v) identity[v][v]=1.0;
        auto maps=orientMaps(aFace,bFace,identity,identity,nodes);
        std::array<std::array<double,3>,3> ma{maps[0],maps[1],maps[2]},mb{maps[3],maps[4],maps[5]};
        const bool swapped=pa>pb;
        Record r; r.pair=std::minmax(pa,pb); r.stable=stable; r.source="starch"; r.sourceIndex=i;
        r.left=toPrimitiveMap(swapped?bFace:aFace,swapped?mb:ma,primitiveNodes(swapped?bTet:aTet,opposite(swapped?bTet:aTet,swapped?bFace:aFace)));
        r.right=toPrimitiveMap(swapped?aFace:bFace,swapped?ma:mb,primitiveNodes(swapped?aTet:bTet,opposite(swapped?aTet:bTet,swapped?aFace:bFace)));
        groups[r.pair].push_back(r);
    }
    for (std::uint32_t i=0;i<hma.patches.size();++i,++stable) {
        const auto& p=hma.patches[i];
        const auto& aTet=tets.at(p.firstTetrahedron).nodes; const auto& bTet=tets.at(p.secondTetrahedron).nodes;
        const auto pa=4u*p.firstTetrahedron+opposite(aTet,p.sideANodes);
        const auto pb=4u*p.secondTetrahedron+opposite(bTet,p.sideBNodes);
        auto maps=orientMaps(p.sideANodes,p.sideBNodes,p.sideAVertexBarycentric,p.sideBVertexBarycentric,nodes);
        std::array<std::array<double,3>,3> ma{maps[0],maps[1],maps[2]},mb{maps[3],maps[4],maps[5]};
        const bool swapped=pa>pb;
        Record r; r.pair=std::minmax(pa,pb); r.stable=p.stableIdentifier; r.source="hma"; r.sourceIndex=i;
        r.left=toPrimitiveMap(swapped?p.sideBNodes:p.sideANodes,swapped?mb:ma,primitiveNodes(swapped?bTet:aTet,opposite(swapped?bTet:aTet,swapped?p.sideBNodes:p.sideANodes)));
        r.right=toPrimitiveMap(swapped?p.sideANodes:p.sideBNodes,swapped?ma:mb,primitiveNodes(swapped?aTet:bTet,opposite(swapped?aTet:bTet,swapped?p.sideANodes:p.sideBNodes)));
        groups[r.pair].push_back(r);
    }
    std::map<std::string,std::uint64_t> failures;
    std::uint32_t maxMultiplicity=0,maxCells=0;
    Pair maxCellPair{}; float maxAreaError=0.0f;
    std::uint64_t audited=0, valid=0;
    bool emittedFirstFailure=false;
    std::uint64_t sourceRejectedOverlapPairs=0u, preciseOverlapPairs=0u,
        toleranceOnlyOverlapPairs=0u, sourceAreaDegenerateSides=0u,
        sourceOnlyExactEdgeCertifiedPairs=0u,
        sourceOnlyExactEdgeUncertifiedPairs=0u;
    for (auto& [pair,records]:groups) {
        ++audited;
        std::sort(records.begin(),records.end(),[](const Record&a,const Record&b){return a.stable<b.stable;});
        maxMultiplicity=std::max(maxMultiplicity,static_cast<std::uint32_t>(records.size()));
        std::vector<FaceMap> left,right; std::vector<std::uint32_t> ids;
        for (const auto& r:records) { left.push_back(r.left); right.push_back(r.right); ids.push_back(r.stable); }
        for (const auto& r:records) {
            if (std::abs(signedArea(r.left))<=kAreaFloor) ++sourceAreaDegenerateSides;
            if (std::abs(signedArea(r.right))<=kAreaFloor) ++sourceAreaDegenerateSides;
        }
        for (std::size_t i=0u;i<records.size();++i) {
            for (std::size_t j=i+1u;j<records.size();++j) {
                std::string leftError, rightError;
                const float leftSource=intersectionArea(records[i].left,records[j].left,leftError);
                const float rightSource=intersectionArea(records[i].right,records[j].right,rightError);
                const bool leftSourceHit=leftError.empty() && std::isfinite(leftSource) &&
                    leftSource>kOverlapTolerance;
                const bool rightSourceHit=rightError.empty() && std::isfinite(rightSource) &&
                    rightSource>kOverlapTolerance;
                const long double leftPrecise=preciseIntersectionArea(records[i].left,records[j].left);
                const long double rightPrecise=preciseIntersectionArea(records[i].right,records[j].right);
                const bool leftPreciseHit=std::isfinite(leftPrecise) &&
                    leftPrecise>static_cast<long double>(kOverlapTolerance);
                const bool rightPreciseHit=std::isfinite(rightPrecise) &&
                    rightPrecise>static_cast<long double>(kOverlapTolerance);
                const bool sourceHit=leftSourceHit||rightSourceHit;
                const bool preciseHit=leftPreciseHit||rightPreciseHit;
                const bool leftEdgeProof=exactOppositeSharedEdge(
                    records[i].left,records[j].left);
                const bool rightEdgeProof=exactOppositeSharedEdge(
                    records[i].right,records[j].right);
                if (sourceHit) ++sourceRejectedOverlapPairs;
                if (preciseHit) ++preciseOverlapPairs;
                if (sourceHit&&!preciseHit) {
                    ++toleranceOnlyOverlapPairs;
                    const bool certified=(!leftSourceHit||leftEdgeProof) &&
                        (!rightSourceHit||rightEdgeProof);
                    if (certified) ++sourceOnlyExactEdgeCertifiedPairs;
                    else ++sourceOnlyExactEdgeUncertifiedPairs;
                    auto emitUnproven = [&](const char* side, const FaceMap& a,
                                            const FaceMap& b, float area,
                                            bool hit, bool edgeProof) {
                        if (!hit || edgeProof) return;
                        std::cout << "UNPROVEN," << side << ',' << records[i].stable
                                  << ',' << records[j].stable << ',' << std::hexfloat << area;
                        for (const auto& v : a) std::cout << ',' << v.x << ',' << v.y << ',' << v.z;
                        for (const auto& v : b) std::cout << ',' << v.x << ',' << v.y << ',' << v.z;
                        std::cout << std::defaultfloat << '\n';
                    };
                    emitUnproven("left", records[i].left, records[j].left,
                                 leftSource, leftSourceHit, leftEdgeProof);
                    emitUnproven("right", records[i].right, records[j].right,
                                 rightSource, rightSourceHit, rightEdgeProof);
                }
            }
        }
        auto lp=buildPartition(left,ids); auto rp=buildPartition(right,ids);
        if (!lp.error.empty()) {
            ++failures["left:"+lp.error];
            if (!emittedFirstFailure) {
                emittedFirstFailure=true;
                std::cout << std::setprecision(10);
                std::cout << "first_failure_side=left reason="<<lp.error<<" pair="<<pair.first<<','<<pair.second<<" records="<<records.size()<<"\n";
                for (std::size_t i=0; i<records.size(); ++i) for (std::size_t j=i+1; j<records.size(); ++j) {
                    std::string err;
                    const float overlap=intersectionArea(records[i].left,records[j].left,err);
                    if (!err.empty() || overlap>0.0f)
                        std::cout << "left_overlap stable="<<records[i].stable<<','<<records[j].stable<<" area="<<overlap<<" error="<<err<<"\n";
                }
                for (const auto& r:records) {
                    std::cout << "record stable="<<r.stable<<" source="<<r.source<<" index="<<r.sourceIndex;
                    std::cout << std::hexfloat;
                    for (const auto& q:r.left) std::cout << " left="<<q.x<<','<<q.y<<','<<q.z;
                    for (const auto& q:r.right) std::cout << " right="<<q.x<<','<<q.y<<','<<q.z;
                    std::cout << std::defaultfloat;
                    std::cout << '\n';
                }
            }
            continue;
        }
        if (!rp.error.empty()) {
            ++failures["right:"+rp.error];
            if (!emittedFirstFailure) {
                emittedFirstFailure=true;
                std::cout << std::setprecision(10);
                std::cout << "first_failure_side=right reason="<<rp.error<<" pair="<<pair.first<<','<<pair.second<<" records="<<records.size()<<"\n";
                for (std::size_t i=0; i<records.size(); ++i) for (std::size_t j=i+1; j<records.size(); ++j) {
                    std::string err;
                    const float overlap=intersectionArea(records[i].right,records[j].right,err);
                    if (!err.empty() || overlap>0.0f)
                        std::cout << "right_overlap stable="<<records[i].stable<<','<<records[j].stable<<" area="<<overlap<<" error="<<err<<"\n";
                }
                for (const auto& r:records) {
                    std::cout << "record stable="<<r.stable<<" source="<<r.source<<" index="<<r.sourceIndex;
                    for (const auto& q:r.left) std::cout << " left="<<q.x<<','<<q.y<<','<<q.z;
                    for (const auto& q:r.right) std::cout << " right="<<q.x<<','<<q.y<<','<<q.z;
                    std::cout << '\n';
                }
            }
            continue;
        }
        ++valid;
        if (lp.cells.size()>maxCells) { maxCells=static_cast<std::uint32_t>(lp.cells.size()); maxCellPair=pair; }
        if (rp.cells.size()>maxCells) { maxCells=static_cast<std::uint32_t>(rp.cells.size()); maxCellPair=pair; }
        maxAreaError=std::max(maxAreaError,std::abs(totalArea(lp)-0.5f));
        maxAreaError=std::max(maxAreaError,std::abs(totalArea(rp)-0.5f));
    }
    std::cout << "nodes="<<starch.referenceNodesM.size()<<" tets="<<starch.tetrahedra.size()
              <<" starch_faces="<<starch.pairedFaces.size()<<" hma_patches="<<hma.patches.size()
              <<" unique_parent_pairs="<<groups.size()<<" max_multiplicity="<<maxMultiplicity
              <<" audited_pairs="<<audited<<" partition_valid_pairs="<<valid
              <<" max_partition_cells="<<maxCells<<" max_cell_pair="<<maxCellPair.first<<','<<maxCellPair.second
              <<" max_parent_area_error="<<maxAreaError<<"\n";
    std::cout << "overlap_classification source_FP32_reject_pairs="<<sourceRejectedOverlapPairs
              <<" long_double_zero_clip_FP32_map_pairs_over_threshold="<<preciseOverlapPairs
              <<" source_only_tolerance_sliver_pairs="<<toleranceOnlyOverlapPairs
              <<" source_only_exact_shared_edge_proven_pairs="<<sourceOnlyExactEdgeCertifiedPairs
              <<" source_only_unproven_pairs="<<sourceOnlyExactEdgeUncertifiedPairs
              <<" FP32_patch_side_areas_at_or_below_native_floor="<<sourceAreaDegenerateSides<<"\n";
    for (const auto& [reason,count]:failures) std::cout << "failure_"<<reason<<'='<<count<<'\n';
    return failures.empty()?0:2;
}
