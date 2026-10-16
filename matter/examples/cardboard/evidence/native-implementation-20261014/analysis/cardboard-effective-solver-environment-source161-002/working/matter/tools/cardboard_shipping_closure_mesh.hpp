#pragma once

// Source-coordinate preparation for the explicit HMA overlap patches in the
// shipping-box protocol.  This creates zero-thickness mortar triangles over a
// declared adhesive footprint; it does not supply or calibrate an HMA law.

#include "cardboard_box_blank.hpp"
#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace numi::cardboard {

struct ShippingClosureTet {
    std::array<std::uint32_t, 4> nodes{};
    std::uint32_t materialIndex = 0u;
};

struct ShippingClosurePatchGroup {
    std::string patchId;
    std::vector<std::uint32_t> stableIdentifiers;
    double expectedAdhesiveAreaM2 = 0.0;
    double preparedAdhesiveAreaM2 = 0.0;
};

struct ShippingClosureMortarMesh {
    std::vector<numi::matter::CohesiveMortarPatchSource> patches;
    std::vector<ShippingClosurePatchGroup> groups;
};

namespace shipping_closure_detail {

struct Point2 {
    double x = 0.0;
    double y = 0.0;
};
struct Point3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};
struct Rect2 {
    double x0 = 0.0;
    double y0 = 0.0;
    double x1 = 0.0;
    double y1 = 0.0;
};
struct BoundaryFace {
    std::array<std::uint32_t, 3> nodes{};
    std::uint32_t tetrahedron = 0u;
    std::uint32_t material = 0u;
    double surfaceZ = 0.0;
};
struct CandidateFace {
    BoundaryFace face;
    std::array<Point2, 3> projected{};
};
struct Domain {
    enum class Kind : std::uint8_t { wall, tab, flap } kind = Kind::wall;
    std::uint32_t panel = 0u;
    bool top = false;
    // Only the manufactured outside surfaces are adhesive candidates.
    double surfaceZ = 0.0;
};
struct PatchDefinition {
    std::string id;
    Domain sideA;
    Domain sideB;
    bool lapPlane = false;
    Rect2 footprint;
    double expectedArea = 0.0;
};

inline Point3 add(Point3 a, Point3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline Point3 sub(Point3 a, Point3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline Point3 mul(double s, Point3 a) { return {s*a.x,s*a.y,s*a.z}; }
inline double dot(Point3 a, Point3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline Point3 cross(Point3 a, Point3 b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
inline double cross2(Point2 a, Point2 b, Point2 c) {
    return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
}
inline double area2(const std::vector<Point2>& polygon) {
    double sum = 0.0;
    for (std::size_t i = 0u; i < polygon.size(); ++i) {
        const auto& a = polygon[i];
        const auto& b = polygon[(i+1u)%polygon.size()];
        sum += a.x*b.y-a.y*b.x;
    }
    return sum;
}
inline Point3 rotate(Point3 value, Point3 axis, double angle) {
    axis = mul(1.0/std::sqrt(dot(axis,axis)), axis);
    return add(add(mul(std::cos(angle),value),
                   mul(std::sin(angle),cross(axis,value))),
               mul((1.0-std::cos(angle))*dot(axis,value),axis));
}
inline Point2 project(Point3 p, bool lapPlane) {
    return lapPlane ? Point2{p.y,p.z} : Point2{p.x,p.z};
}
inline Point3 panelAxis(const std::uint32_t panel) {
    switch (panel) {
        case 0u: return {1.0,0.0,0.0};
        case 1u: return {0.0,0.0,-1.0};
        case 2u: return {-1.0,0.0,0.0};
        case 3u: return {0.0,0.0,1.0};
        default: throw std::invalid_argument("shipping closure panel index is invalid");
    }
}
inline Point3 panelOrigin(const FEFCO0201Blank& blank,
                          const std::uint32_t panel) {
    const double length = blank.config.panelLength;
    const double width = blank.config.panelWidth;
    switch (panel) {
        case 0u: return {0.0,0.0,width};
        case 1u: return {length,0.0,width};
        case 2u: return {length,0.0,0.0};
        case 3u: return {0.0,0.0,0.0};
        default: throw std::invalid_argument("shipping closure panel index is invalid");
    }
}
inline Point3 panelNormal(const std::uint32_t panel) {
    const Point3 eu = panelAxis(panel);
    return cross(eu,{0.0,1.0,0.0});
}
inline Point3 wallPoint(const FEFCO0201Blank& blank,
                        const std::uint32_t panel,
                        const BoxBlankPoint& p) {
    const double u = p.x-blank.panels.at(panel).xStart;
    const double v = p.y-blank.bottomScoreY;
    return add(add(panelOrigin(blank,panel),mul(u,panelAxis(panel))),
               add(Point3{0.0,v,0.0},mul(p.z,panelNormal(panel))));
}
inline Point3 tabPoint(const FEFCO0201Blank& blank,
                       const BoxBlankPoint& p) {
    // The tab is folded from panel 0's leading edge around the vertical root
    // to the outside of panel 3.  The one-caliper translation stacks the tab
    // outside the panel rather than overlapping their bulk volumes.
    const double u = p.x-blank.config.manufacturerJointWidth;
    const double v = p.y-blank.bottomScoreY;
    const Point3 turned = rotate({u,0.0,p.z},{0.0,1.0,0.0},-0.5*3.14159265358979323846);
    return add({0.0,v,blank.config.panelWidth},
               add(turned,{-blank.boardCaliper,0.0,0.0}));
}
inline Point3 flapPoint(const FEFCO0201Blank& blank,
                        const std::uint32_t panel,
                        const bool top,
                        const BoxBlankPoint& p) {
    const double hingeY = top ? blank.topScoreY : blank.bottomScoreY;
    const Point3 hinge = wallPoint(blank,panel,{p.x,hingeY,0.0});
    const Point3 base = wallPoint(blank,panel,p);
    const double angle = top ? -0.5*3.14159265358979323846
                             :  0.5*3.14159265358979323846;
    Point3 folded = add(hinge,rotate(sub(base,hinge),panelAxis(panel),angle));
    if (panel == 0u || panel == 2u) {
        const Point3 outwardAfterFold = rotate(panelNormal(panel),panelAxis(panel),angle);
        folded = add(folded,mul(blank.boardCaliper,outwardAfterFold));
    }
    return folded;
}
inline Point3 mapPoint(const FEFCO0201Blank& blank,
                       const Domain& domain,
                       const BoxBlankPoint& point) {
    if (domain.kind == Domain::Kind::tab) return tabPoint(blank,point);
    if (domain.kind == Domain::Kind::wall) return wallPoint(blank,domain.panel,point);
    return flapPoint(blank,domain.panel,domain.top,point);
}

inline std::array<std::uint32_t,3> faceNodes(
    const std::array<std::uint32_t,4>& tet, const std::uint32_t omitted) {
    std::array<std::uint32_t,3> face{};
    std::size_t cursor=0u;
    for (std::uint32_t i=0u;i<4u;++i) if (i!=omitted) face[cursor++]=tet[i];
    std::sort(face.begin(),face.end());
    return face;
}
inline double toleranceFor(const FEFCO0201Blank& blank) {
    return 64.0*std::numeric_limits<double>::epsilon()*
        std::max({1.0,blank.perimeterSpan,blank.blankHeight,blank.boardCaliper});
}
inline bool inDomain(const FEFCO0201Blank& blank, const Domain& domain,
                     const BoxBlankPoint& p, const double tol) {
    if (std::abs(p.z-domain.surfaceZ)>tol) return false;
    if (domain.kind==Domain::Kind::tab)
        return p.x>=-tol && p.x<=blank.config.manufacturerJointWidth+tol &&
            p.y>=blank.bottomScoreY-tol && p.y<=blank.topScoreY+tol;
    if (domain.kind==Domain::Kind::wall) {
        const auto& panel=blank.panels.at(domain.panel);
        return p.x>=panel.xStart-tol && p.x<=panel.xEnd+tol &&
            p.y>=blank.bottomScoreY-tol && p.y<=blank.topScoreY+tol;
    }
    const auto& panel=blank.panels.at(domain.panel);
    return p.x>=panel.xStart-tol && p.x<=panel.xEnd+tol &&
        (domain.top
            ? p.y>=blank.topScoreY-tol && p.y<=blank.blankHeight+tol
            : p.y>=-tol && p.y<=blank.bottomScoreY+tol);
}
inline std::vector<BoundaryFace> boundaryFaces(
    const FEFCO0201Blank& blank,
    const std::vector<BoxBlankPoint>& nodes,
    const std::vector<ShippingClosureTet>& tets,
    const double tolerance) {
    std::map<std::array<std::uint32_t,3>,std::vector<std::pair<std::uint32_t,std::uint32_t>>> incidence;
    for (std::uint32_t ti=0u;ti<tets.size();++ti)
        for (std::uint32_t opposite=0u;opposite<4u;++opposite)
            incidence[faceNodes(tets[ti].nodes,opposite)].push_back({ti,opposite});
    std::vector<BoundaryFace> result;
    result.reserve(incidence.size());
    for (const auto& [key,owners]:incidence) {
        if (owners.size()>2u) throw std::invalid_argument("shipping closure mesh has a nonmanifold face");
        if (owners.size()!=1u) continue;
        const auto [ti,opposite]=owners.front();
        (void)opposite;
        const double z0=nodes.at(key[0]).z;
        const bool bottom=std::abs(z0)<=tolerance;
        const bool top=std::abs(z0-blank.boardCaliper)<=tolerance;
        if (!bottom && !top) continue;
        if (std::any_of(key.begin(),key.end(),[&](const auto node) {
            return std::abs(nodes.at(node).z-z0)>tolerance;
        })) continue;
        result.push_back({key,ti,tets[ti].materialIndex,bottom?0.0:blank.boardCaliper});
    }
    return result;
}

inline std::vector<Point2> trianglePolygon(const std::array<Point2,3>& tri) {
    std::vector<Point2> result{tri[0],tri[1],tri[2]};
    if (area2(result)<0.0) std::swap(result[1],result[2]);
    return result;
}
inline std::vector<Point2> clipConvex(std::vector<Point2> subject,
                                      const std::vector<Point2>& clip,
                                      const double eps) {
    for (std::size_t edge=0u;edge<clip.size() && !subject.empty();++edge) {
        const Point2 a=clip[edge], b=clip[(edge+1u)%clip.size()];
        std::vector<Point2> output;
        Point2 previous=subject.back();
        double previousSide=cross2(a,b,previous);
        for (const Point2 current:subject) {
            const double currentSide=cross2(a,b,current);
            const bool currentInside=currentSide>=-eps;
            const bool previousInside=previousSide>=-eps;
            if (currentInside!=previousInside) {
                const double denom=previousSide-currentSide;
                if (std::abs(denom)>std::numeric_limits<double>::min()) {
                    const double t=previousSide/denom;
                    output.push_back({previous.x+t*(current.x-previous.x),
                                      previous.y+t*(current.y-previous.y)});
                }
            }
            if (currentInside) output.push_back(current);
            previous=current;
            previousSide=currentSide;
        }
        subject=std::move(output);
    }
    std::vector<Point2> clean;
    for (const Point2 p:subject) {
        if (clean.empty() || std::hypot(p.x-clean.back().x,p.y-clean.back().y)>eps)
            clean.push_back(p);
    }
    if (clean.size()>1u && std::hypot(clean.front().x-clean.back().x,
                                      clean.front().y-clean.back().y)<=eps)
        clean.pop_back();
    if (clean.size()>=3u && area2(clean)<0.0) std::reverse(clean.begin(),clean.end());
    if (clean.size()<3u || std::abs(area2(clean))<=eps*eps) return {};
    return clean;
}
inline std::vector<Point2> intersectTriangleAndRect(
    const std::array<Point2,3>& triangle, const Rect2& rect, const double eps) {
    std::vector<Point2> rectangle{{rect.x0,rect.y0},{rect.x1,rect.y0},
                                  {rect.x1,rect.y1},{rect.x0,rect.y1}};
    return clipConvex(trianglePolygon(triangle),rectangle,eps);
}
inline std::array<double,3> barycentric(Point2 p,
                                        const std::array<Point2,3>& tri) {
    const double det=cross2(tri[0],tri[1],tri[2]);
    if (!std::isfinite(det) || std::abs(det)<=1.0e-24)
        throw std::invalid_argument("shipping closure parent face is degenerate in its projection");
    const double b=cross2(tri[0],p,tri[2])/det;
    const double c=cross2(tri[0],tri[1],p)/det;
    const double a=1.0-b-c;
    std::array<double,3> weights{a,b,c};
    for (double& weight:weights) {
        if (weight<0.0 && weight>-1.0e-10) weight=0.0;
        if (weight>1.0 && weight<1.0+1.0e-10) weight=1.0;
    }
    const double sum=weights[0]+weights[1]+weights[2];
    if (!(sum>0.0) || !std::isfinite(sum))
        throw std::invalid_argument("shipping closure barycentric map is invalid");
    for (double& weight:weights) weight/=sum;
    return weights;
}
inline std::array<double,3> clipBarycentric(
    Point2 p, const std::array<Point2,3>& tri) {
    return barycentric(p,tri);
}

inline std::vector<PatchDefinition> definitions(
    const FEFCO0201Blank& blank,
    const double flapStripWidth) {
    const double length=blank.config.panelLength;
    const double width=blank.config.panelWidth;
    const double flap=blank.topFlapDepth;
    const double height=blank.config.wallHeight;
    const double joint=blank.config.manufacturerJointWidth;
    std::vector<PatchDefinition> result;
    const auto addLap=[&] {
        PatchDefinition d;
        d.id="manufacturer-hma-lap";
        d.lapPlane=true;
        d.sideA={Domain::Kind::wall,3u,false,blank.boardCaliper};
        d.sideB={Domain::Kind::tab,0u,false,0.0};
        // The manufactured overlap is 25 mm wide; the declared adhesive
        // footprint is a centered 10 mm strip along the full wall height.
        const double lapStripCenter = width - 0.5 * joint;
        d.footprint={0.0,lapStripCenter-0.5*flapStripWidth,
                     height,lapStripCenter+0.5*flapStripWidth};
        d.expectedArea=height*flapStripWidth;
        result.push_back(std::move(d));
    };
    addLap();
    for (const bool top:{false,true}) {
        for (const std::uint32_t major:{0u,2u}) {
            for (const std::uint32_t minor:{1u,3u}) {
                PatchDefinition d;
                d.id=std::string(top?"top":"bottom")+"-major-"+
                    std::to_string(major)+"-minor-"+std::to_string(minor);
                d.lapPlane=false;
                d.sideA={Domain::Kind::flap,minor,top,blank.boardCaliper};
                d.sideB={Domain::Kind::flap,major,top,0.0};
                // Side panel 1 meets the right ends of both major flaps;
                // side panel 3 meets their left ends.
                const double x0=minor==1u?length-flap:0.0;
                const double x1=minor==1u?length:flap;
                const double z0=major==0u?width-flap:0.0;
                const double z1=major==0u?width:flap;
                const double stripCenter=0.5*(z0+z1);
                d.footprint={x0,stripCenter-0.5*flapStripWidth,
                             x1,stripCenter+0.5*flapStripWidth};
                // A scored corner slot removes half of its kerf from each
                // adjacent flap. The p0/minor3 left outside corner has no
                // slot; the other three corner overlaps lose kerf/2.
                const bool slottedCorner = minor==1u || major==2u;
                d.expectedArea=(flap-(slottedCorner
                    ? 0.5*blank.config.slotKerf : 0.0))*flapStripWidth;
                result.push_back(std::move(d));
            }
        }
    }
    return result;
}

inline std::vector<CandidateFace> candidates(
    const FEFCO0201Blank& blank,
    const std::vector<BoxBlankPoint>& nodes,
    const std::vector<BoundaryFace>& boundary,
    const Domain& domain,
    const bool lapPlane,
    const double eps) {
    std::vector<CandidateFace> result;
    for (const auto& face:boundary) {
        if (std::abs(face.surfaceZ-domain.surfaceZ)>eps || face.material!=0u) continue;
        bool inside=true;
        CandidateFace candidate;
        candidate.face=face;
        for (std::size_t i=0u;i<3u;++i) {
            if (!inDomain(blank,domain,nodes.at(face.nodes[i]),eps)) {
                inside=false;
                break;
            }
            candidate.projected[i]=project(
                mapPoint(blank,domain,nodes.at(face.nodes[i])),lapPlane);
        }
        if (inside) result.push_back(candidate);
    }
    return result;
}

} // namespace shipping_closure_detail

inline std::array<double,3> shippingClosureTabFoldTarget(
    const FEFCO0201Blank& blank,
    const BoxBlankPoint& point) {
    const auto mapped=shipping_closure_detail::tabPoint(blank,point);
    return {mapped.x,mapped.y,mapped.z};
}

inline ShippingClosureMortarMesh buildShippingClosureMortarMesh(
    const FEFCO0201Blank& blank,
    const std::vector<BoxBlankPoint>& nodes,
    const std::vector<ShippingClosureTet>& tetrahedra,
    const double hmaStripWidthM,
    const double activationGapToleranceM,
    const std::uint32_t cohesiveMaterialIndex,
    const std::uint32_t firstStableIdentifier) {
    using namespace shipping_closure_detail;
    const auto positive=[](const double x){return std::isfinite(x)&&x>0.0;};
    if (nodes.size()<4u || tetrahedra.empty() ||
        !positive(blank.config.panelLength) || !positive(blank.config.panelWidth) ||
        !positive(blank.config.wallHeight) || !positive(blank.boardCaliper) ||
        !positive(hmaStripWidthM) || !positive(activationGapToleranceM) ||
        hmaStripWidthM>=blank.topFlapDepth ||
        firstStableIdentifier==0u)
        throw std::invalid_argument("shipping closure mortar inputs are invalid");
    for (const auto& tet:tetrahedra)
        for (const auto node:tet.nodes)
            if (node>=nodes.size()) throw std::invalid_argument("shipping closure tet node is out of range");
    const double eps=toleranceFor(blank);
    const auto boundary=boundaryFaces(blank,nodes,tetrahedra,eps);
    const auto defs=definitions(blank,hmaStripWidthM);
    ShippingClosureMortarMesh result;
    std::uint64_t nextStable=firstStableIdentifier;
    for (const auto& definition:defs) {
        ShippingClosurePatchGroup group;
        group.patchId=definition.id;
        group.expectedAdhesiveAreaM2=definition.expectedArea;
        auto sideA=candidates(blank,nodes,boundary,definition.sideA,
                              definition.lapPlane,eps);
        auto sideB=candidates(blank,nodes,boundary,definition.sideB,
                              definition.lapPlane,eps);
        for (const auto& a:sideA) {
            const double aminx=std::min({a.projected[0].x,a.projected[1].x,a.projected[2].x});
            const double amaxx=std::max({a.projected[0].x,a.projected[1].x,a.projected[2].x});
            const double aminy=std::min({a.projected[0].y,a.projected[1].y,a.projected[2].y});
            const double amaxy=std::max({a.projected[0].y,a.projected[1].y,a.projected[2].y});
            if (amaxx<definition.footprint.x0-eps || aminx>definition.footprint.x1+eps ||
                amaxy<definition.footprint.y0-eps || aminy>definition.footprint.y1+eps) continue;
            for (const auto& b:sideB) {
                const double bminx=std::min({b.projected[0].x,b.projected[1].x,b.projected[2].x});
                const double bmaxx=std::max({b.projected[0].x,b.projected[1].x,b.projected[2].x});
                const double bminy=std::min({b.projected[0].y,b.projected[1].y,b.projected[2].y});
                const double bmaxy=std::max({b.projected[0].y,b.projected[1].y,b.projected[2].y});
                if (bmaxx<aminx-eps || bminx>amaxx+eps || bmaxy<aminy-eps || bminy>amaxy+eps) continue;
                auto polygon=intersectTriangleAndRect(a.projected,definition.footprint,eps);
                if (polygon.empty()) continue;
                polygon=clipConvex(std::move(polygon),trianglePolygon(b.projected),eps);
                if (polygon.empty()) continue;
                for (std::size_t i=1u;i+1u<polygon.size();++i) {
                    const std::array<Point2,3> vertices{polygon[0],polygon[i],polygon[i+1u]};
                    const double area=0.5*std::abs(cross2(vertices[0],vertices[1],vertices[2]));
                    if (!(area>1.0e-20)) continue;
                    if (nextStable>std::numeric_limits<std::uint32_t>::max())
                        throw std::length_error("shipping HMA stable identifiers exceed Matter's 32-bit source API");
                    numi::matter::CohesiveMortarPatchSource patch;
                    patch.sideANodes=a.face.nodes;
                    patch.sideBNodes=b.face.nodes;
                    patch.firstTetrahedron=a.face.tetrahedron;
                    patch.secondTetrahedron=b.face.tetrahedron;
                    patch.materialIndex=a.face.material;
                    patch.stableIdentifier=static_cast<std::uint32_t>(nextStable++);
                    patch.cohesiveMaterialIndex=cohesiveMaterialIndex;
                    patch.activationGapToleranceM=activationGapToleranceM;
                    for (std::size_t vertex=0u;vertex<3u;++vertex) {
                        patch.sideAVertexBarycentric[vertex]=barycentric(vertices[vertex],a.projected);
                        patch.sideBVertexBarycentric[vertex]=barycentric(vertices[vertex],b.projected);
                    }
                    for (std::size_t ai=0u;ai<3u;++ai)
                        for (std::size_t bi=0u;bi<3u;++bi)
                            if (patch.sideANodes[ai]==patch.sideBNodes[bi])
                                throw std::invalid_argument("shipping HMA patch sides share source nodes");
                    group.stableIdentifiers.push_back(patch.stableIdentifier);
                    group.preparedAdhesiveAreaM2+=area;
                    result.patches.push_back(std::move(patch));
                }
            }
        }
        const double allowed=2.0e-8*std::max(group.expectedAdhesiveAreaM2,1.0e-18);
        if (group.stableIdentifiers.empty() ||
            std::abs(group.preparedAdhesiveAreaM2-group.expectedAdhesiveAreaM2)>allowed)
            throw std::invalid_argument("shipping HMA overlay does not cover the authored adhesive footprint for " + group.patchId);
        result.groups.push_back(std::move(group));
    }
    return result;
}

inline double shippingMortarAcceptedMaximumGapM(
    const numi::matter::CohesiveMortarPatchSource& patch,
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    const std::uint32_t environment,
    const std::uint32_t nodeCount) {
    if (nodeCount==0u || acceptedNodes.size()<(static_cast<std::size_t>(environment)+1u)*nodeCount)
        throw std::invalid_argument("shipping accepted mortar audit has a truncated FEM state");
    const auto mapped=[&](const std::array<std::uint32_t,3>& ids,
                          const std::array<std::array<double,3>,3>& weights) {
        std::array<shipping_closure_detail::Point3,3> points{};
        for (std::size_t vertex=0u;vertex<3u;++vertex)
            for (std::size_t corner=0u;corner<3u;++corner) {
                const auto& node=acceptedNodes.at(static_cast<std::size_t>(environment)*nodeCount+ids[corner]);
                points[vertex].x+=weights[vertex][corner]*node.positionAndMass.x;
                points[vertex].y+=weights[vertex][corner]*node.positionAndMass.y;
                points[vertex].z+=weights[vertex][corner]*node.positionAndMass.z;
            }
        return points;
    };
    const auto a=mapped(patch.sideANodes,patch.sideAVertexBarycentric);
    const auto b=mapped(patch.sideBNodes,patch.sideBVertexBarycentric);
    double maximum=0.0;
    for (std::size_t i=0u;i<3u;++i)
        maximum=std::max(maximum,std::sqrt(dot(sub(a[i],b[i]),sub(a[i],b[i]))));
    return maximum;
}

} // namespace numi::cardboard
