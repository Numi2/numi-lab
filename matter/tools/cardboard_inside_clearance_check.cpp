#include "cardboard_inside_clearance.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace numi::cardboard;
using namespace inside_clearance_detail;

void check(bool ok,const char* message) {
    if (!ok) throw std::runtime_error(message);
}

struct Fixture {
    FEFCO0201Blank blank;
    std::vector<NMFEMNodeStateGPU> nodes;
    std::vector<NMTetrahedronGPU> tets;
    std::vector<NMFEMSurfaceFaceGPU> faces;
};

Fixture makeFixture(bool gapMajor=false,double buttGap=0.0) {
    Fixture f;
    f.blank.config={0.30,0.20,0.20,0.025,0.002,1u};
    f.blank.perimeterSpan=1.025;
    f.blank.boardCaliper=0.01;
    f.blank.bottomScoreY=0.20;
    f.blank.topScoreY=0.40;
    f.blank.topFlapDepth=0.20;
    f.blank.blankHeight=0.60;
    const std::array<double,5> edge{0.025,0.325,0.525,0.825,1.025};
    for (std::uint32_t p=0;p<4u;++p)
        f.blank.panels.push_back({p,edge[p],edge[p+1],0.20,0.40,p==0u||p==2u});
    f.blank.scoreLines.push_back({BoxScoreKind::manufacturerJointRoot,0u,{0.025,0.20,0},{0.025,0.40,0},90.0,false});
    for (std::uint32_t p=0;p<3u;++p)
        f.blank.scoreLines.push_back({BoxScoreKind::bodyCorner,p,{edge[p+1],0.20,0},{edge[p+1],0.40,0},90.0,false});
    for (std::uint32_t p=0;p<4u;++p) {
        const auto& b=f.blank.panels[p];
        f.blank.scoreLines.push_back({BoxScoreKind::bottomFlap,p,{b.xStart,0.20,0},{b.xEnd,0.20,0},90.0,false});
        f.blank.scoreLines.push_back({BoxScoreKind::topFlap,p,{b.xStart,0.40,0},{b.xEnd,0.40,0},-90.0,false});
    }
    for (bool top:{false,true}) for (std::uint32_t p:{1u,3u,0u,2u}) {
        const auto kind=top?BoxScoreKind::topFlap:BoxScoreKind::bottomFlap;
        const auto it=std::find_if(f.blank.scoreLines.begin(),f.blank.scoreLines.end(),[&](const auto& s){return s.kind==kind&&s.panelIndex==p;});
        check(it!=f.blank.scoreLines.end(),"fixture score line missing");
        f.blank.assembly.actions.push_back({BoxAssemblyAction::Kind::foldFlap,
            {BoxAssemblyFeatureKind::scoreLine,static_cast<std::uint32_t>(it-f.blank.scoreLines.begin())},
            kNoBoxScoreLineIndex,top?-90.0:90.0,"synthetic",true});
    }

    const auto current=[&](std::uint32_t panel,BoxBlankFaceRegion region,
                           double x,double y,double z) -> V3 {
        const auto& b=f.blank.panels[panel];
        const double s=(x-b.xStart)/(b.xEnd-b.xStart);
        if (region==BoxBlankFaceRegion::wall) {
            const double h=(y-f.blank.bottomScoreY)/f.blank.config.wallHeight*0.20;
            if (panel==0u) return {s*0.30,h,z==0.0?0.0:-0.01};
            if (panel==1u) return {0.30+(z==0.0?0.0:0.01),h,s*0.20};
            if (panel==2u) return {(1.0-s)*0.30,h,0.20+(z==0.0?0.0:0.01)};
            return {z==0.0?0.0:-0.01,h,(1.0-s)*0.20};
        }
        const bool top=region==BoxBlankFaceRegion::topFlap;
        const double r=top?(y-f.blank.topScoreY)/0.20:(f.blank.bottomScoreY-y)/0.20;
        double xx=0.0,zz=0.0;
        if (panel==0u||panel==2u) {
            const double span=(gapMajor?0.40:1.0)*0.10;
            xx=panel==0u?s*0.30:(1.0-s)*0.30;
            zz=panel==0u?r*span:0.20-r*span+r*buttGap;
        } else {
            xx=panel==1u?0.30-r*0.15:r*0.15;
            zz=panel==1u?s*0.20:(1.0-s)*0.20;
        }
        const double endY=top?0.20:0.0;
        const double thicknessSide=top?0.01:-0.01;
        return {xx,endY+(z==0.0?0.0:thicknessSide),zz};
    };
    constexpr std::array<std::array<unsigned,4>,6> split{{
        {{0,1,3,7}},{{0,3,2,7}},{{0,2,6,7}},{{0,6,4,7}},{{0,4,5,7}},{{0,5,1,7}}}};
    for (std::uint32_t p=0;p<4u;++p) for (const auto region:{BoxBlankFaceRegion::wall,
            BoxBlankFaceRegion::bottomFlap,BoxBlankFaceRegion::topFlap}) {
        const auto& b=f.blank.panels[p];
        const double x0=b.xStart,x1=b.xEnd;
        double y0=0.0,y1=0.0;
        if (region==BoxBlankFaceRegion::wall) { y0=0.20;y1=0.40; }
        else if (region==BoxBlankFaceRegion::bottomFlap) { y0=0.0;y1=0.20; }
        else { y0=0.40;y1=0.60; }
        const std::array<BoxBlankPoint,8> rest{{
            {x0,y0,0},{x1,y0,0},{x0,y1,0},{x1,y1,0},
            {x0,y0,0.01},{x1,y0,0.01},{x0,y1,0.01},{x1,y1,0.01}}};
        const std::uint32_t base=static_cast<std::uint32_t>(f.nodes.size());
        for (const auto& q:rest) {
            NMFEMNodeStateGPU n{};
            const V3 c=current(p,region,q.x,q.y,q.z);
            n.restAndFixed={static_cast<float>(q.x),static_cast<float>(q.y),static_cast<float>(q.z),0.0f};
            n.positionAndMass={static_cast<float>(c.x),static_cast<float>(c.y),static_cast<float>(c.z),1.0f};
            f.nodes.push_back(n);
        }
        for (const auto& ids:split) {
            NMTetrahedronGPU t{};
            t.nodes={base+ids[0],base+ids[1],base+ids[2],base+ids[3]};
            t.identity={0u,0u,0u,0u};
            f.tets.push_back(t);
        }
    }
    struct Ref { std::uint32_t tet=0,opposite=0,count=0; };
    std::map<std::array<std::uint32_t,3>,Ref> incidence;
    for (std::uint32_t ti=0;ti<f.tets.size();++ti) {
        const auto& t=f.tets[ti]; const std::array<std::uint32_t,4> ids{t.nodes.x,t.nodes.y,t.nodes.z,t.nodes.w};
        for (std::uint32_t opp=0;opp<4u;++opp) {
            std::array<std::uint32_t,3> k{}; std::size_t n=0;
            for (std::uint32_t c=0;c<4u;++c) if (c!=opp) k[n++]=ids[c];
            std::sort(k.begin(),k.end()); auto& r=incidence[k];++r.count;r.tet=ti;r.opposite=opp;
        }
    }
    for (const auto& [key,r]:incidence) if (r.count==1u) {
        NMFEMSurfaceFaceGPU face{};
        face.adjacency={r.tet,NM_INVALID_INDEX,0u,NM_TOPOLOGY_ACTIVE};
        face.sides={r.opposite,0u,static_cast<std::uint32_t>(f.faces.size()),0u};
        f.faces.push_back(face);
    }
    return f;
}

std::array<Triangle2,2> rectangleTriangles(double x0,double x1,double y0,double y1) {
    return {{{{V2{x0,y0},V2{x1,y0},V2{x1,y1}}},
             {{V2{x0,y0},V2{x1,y1},V2{x0,y1}}}}};
}

void run() {
    std::uint64_t shippingHashCapacity=0u,shippingHashBytes=0u,shippingWorkspaceBytes=0u;
    check(ExteriorFaceIndex::capacityFor(1477728u,shippingHashCapacity,shippingHashBytes)&&
          shippingHashCapacity==16777216u&&shippingHashBytes==268435456u&&
          estimateWorkspaceBytes(shippingHashBytes,528444u,1054876u,shippingWorkspaceBytes)&&
          shippingWorkspaceBytes<=1024ull*1024ull*1024ull,
          "nominal shipping topology does not fit the checked topology/workspace plan");
    std::uint64_t overCapacity=0u,overCapacityBytes=0u;
    check(!ExteriorFaceIndex::capacityFor(1600001u,overCapacity,overCapacityBytes),
          "topology index accepted a tetrahedron count above its explicit cap");
    ExteriorFaceIndex collisionIndex(256u);
    std::vector<FaceKey> collidingKeys;
    for (std::uint32_t i=1u;collidingKeys.size()<ExteriorFaceIndex::maxProbesPerKey+1u;++i) {
        FaceKey key{i,i+1000000u,i+2000000u};
        if (ExteriorFaceIndex::bucketForTest(key,256u)==17u) collidingKeys.push_back(key);
    }
    for (std::uint32_t i=0u;i<ExteriorFaceIndex::maxProbesPerKey;++i)
        check(collisionIndex.insert(collidingKeys[i],i,0u),
              "bounded topology index rejected a collision below its probe limit");
    check(!collisionIndex.insert(collidingKeys.back(),129u,0u),
          "bounded topology index accepted a pathological collision chain beyond its limit");
    Plane left{{0,0,0},{1,0,0}},right{{0.30,0,0},{-1,0,0}},reversed{{-0.30,0,0},{-1,0,0}};
    double signedGap=0.0;
    check(positiveOpposingPlaneGap(left,right,signedGap)&&std::abs(signedGap-0.30)<1.0e-12&&
          !positiveOpposingPlaneGap(left,reversed,signedGap),
          "wall clearance accepted an absolute gap with normals facing away from one another");

    std::vector<Triangle2> denseCell;
    denseCell.reserve(5002u);
    const Triangle2 clustered{{V2{0.5000,0.5000},V2{0.5001,0.5000},V2{0.5000,0.5001}}};
    denseCell.push_back({{V2{0.0,0.0},V2{0.001,0.0},V2{0.0,0.001}}});
    denseCell.push_back({{V2{1.0,1.0},V2{0.999,1.0},V2{1.0,0.999}}});
    denseCell.insert(denseCell.end(),5000u,clustered);
    TriangleGrid denseGrid(denseCell);
    std::vector<std::uint32_t> denseCandidates;
    bool denseCandidateOverflow=false;
    check(denseGrid.valid&&
          !denseGrid.query(clustered,denseCandidates,nullptr,&denseCandidateOverflow)&&
          denseCandidateOverflow&&denseCandidates.size()==maximumGridQueryCandidates&&
          denseCandidates.capacity()<=maximumGridQueryCandidates,
          "spatial-grid candidate overflow was not surfaced without exceeding its bounded result storage");
    std::uint64_t densePairs=0u;
    check(!projectedPanelNonoverlap(denseCell,1.0e-12,densePairs),
          "wall-overlap validator admitted a spatial query whose candidate list was truncated");

    auto f=makeFixture();
    auto measured=measureAcceptedInsideClearance(f.blank,f.nodes,
        static_cast<std::uint32_t>(f.nodes.size()),f.tets,f.faces,{0.30,0.20,0.20});
    check(measured.gatePassed,"exact synthetic accepted box failed the inside-clearance gate");
    check(std::abs(measured.measuredInsideM[0]-0.30)<1.0e-5&&
          std::abs(measured.measuredInsideM[1]-0.20)<1.0e-5&&
          std::abs(measured.measuredInsideM[2]-0.20)<1.0e-5,
          "synthetic measured clear dimensions differ from actual geometry");
    check(measured.topologyHashCapacity>0u&&measured.estimatedWorkspaceBytes>0u&&
          measured.tetrahedronCountObserved==f.tets.size()&&
          measured.surfaceFaceCountObserved==f.faces.size(),
          "accepted map did not report its checked topology/workspace accounting");
    check(std::abs(measured.wallDomainAreaM2[0]-0.06)<1.0e-8&&
          measured.wallUncoveredAreaM2[0]<=measured.wallAreaRoundoffBoundM2[0],
          "panel-local wall domain absorbed adjacent flap/neighbor nodes or exceeded its area floor");
    check(measured.topologyComplete&&measured.roleCoverageComplete&&
          measured.faceProvenanceMapComplete&&
          measured.faceRecords.size()==measured.broadFaceCount&&
          measured.authoredSurfaceCoverageComplete&&measured.overlapComplete,
          "synthetic role/coverage/overlap map incomplete");
    std::set<std::uint32_t> mappedFaceIndices;
    std::size_t cavityFaces=0u,outerFaces=0u;
    for (const auto& face:measured.faceRecords) {
        check(face.surfaceFaceIndex<f.faces.size()&&face.tetrahedronIndex<f.tets.size()&&
              face.panelIndex<4u&&face.region!=BoxBlankFaceRegion::none&&
              face.broadSide!=BoxBlankBroadSide::none,
              "face provenance record has an invalid source index or role");
        check(mappedFaceIndices.insert(face.surfaceFaceIndex).second,
              "face provenance map repeated a compiled exterior face");
        if (face.cavityFacing) ++cavityFaces; else ++outerFaces;
    }
    check(cavityFaces>0u&&outerFaces>0u,
          "cavity-side selection did not distinguish the two broad material sides");
    for (const auto& region:measured.insideRestCoverageFraction)
        for (const double fraction:region)
            check(std::abs(fraction-1.0)<1.0e-5,
                  "synthetic accepted surface does not match authored panel area");
    for (const auto& end:measured.minorFlapCoverage)
        for (const double coverage:end)
            check(coverage>=0.99&&coverage<=1.000001,
                  "major-flap overlap union double-counted or missed minor flap area");

    const auto wrongTarget=measureAcceptedInsideClearance(f.blank,f.nodes,
        static_cast<std::uint32_t>(f.nodes.size()),f.tets,f.faces,{0.304,0.20,0.20});
    check(!wrongTarget.gatePassed&&wrongTarget.planarGeometryValid&&
          !wrongTarget.dimensionsWithinTolerance,
          "inside target outside the explicit one-percent gate was admitted");

    // Inner-first flap pieces can all be covered while the last-closing major
    // pair still leaves a central opening. Check the complete measured wall
    // footprint independently of minor-flap coverage.
    std::array<std::vector<Triangle2>,2> sideMajor;
    for (const auto t:rectangleTriangles(0.0,0.4,0.0,1.0)) sideMajor[0].push_back(t);
    for (const auto t:rectangleTriangles(0.6,1.0,0.0,1.0)) sideMajor[1].push_back(t);
    std::vector<Triangle2> sideMinors;
    for (const auto t:rectangleTriangles(0.0,0.4,0.0,1.0)) sideMinors.push_back(t);
    for (const auto t:rectangleTriangles(0.6,1.0,0.0,1.0)) sideMinors.push_back(t);
    double sideMinorCoverage=0.0,sideMinorArea=0.0;
    std::uint64_t sideMinorPairs=0u;
    check(flapUnionCoverage(sideMajor,sideMinors,1.0e-12,sideMinorCoverage,sideMinorArea,sideMinorPairs)&&
          std::abs(sideMinorCoverage-1.0)<1.0e-12,
          "central-opening counterexample did not retain complete minor-flap coverage");
    const auto cavityFootprint=rectangleTriangles(0.0,1.0,0.0,1.0);
    double endFootCoverage=0.0,endFootArea=0.0;
    std::uint64_t endFootPairs=0u;
    check(majorUnionFootprintCoverage(sideMajor,cavityFootprint,1.0e-12,
          endFootCoverage,endFootArea,endFootPairs)&&
          std::abs(endFootCoverage-0.8)<1.0e-12&&endFootCoverage<0.99,
          "major-flap union failed to detect a central opening despite 100% minor coverage");
    std::array<std::vector<Triangle2>,2> overlapMajor;
    for (const auto t:rectangleTriangles(0.0,0.6,0.0,1.0)) overlapMajor[0].push_back(t);
    for (const auto t:rectangleTriangles(0.4,1.0,0.0,1.0)) overlapMajor[1].push_back(t);
    double overlapCoverage=0.0,overlapFootArea=0.0;
    std::uint64_t overlapPairs=0u;
    check(majorUnionFootprintCoverage(overlapMajor,cavityFootprint,1.0e-12,
          overlapCoverage,overlapFootArea,overlapPairs)&&
          std::abs(overlapCoverage-1.0)<1.0e-12&&overlapCoverage<=1.000001,
          "bonded major-flap overlap was double-counted as more than full footprint coverage");

    std::vector<Triangle2> subdividedCover;
    constexpr std::uint32_t coverSide=70u;
    subdividedCover.reserve(2u*coverSide*coverSide);
    for (std::uint32_t y=0u;y<coverSide;++y) for (std::uint32_t x=0u;x<coverSide;++x) {
        const double x0=static_cast<double>(x)/coverSide;
        const double x1=static_cast<double>(x+1u)/coverSide;
        const double y0=static_cast<double>(y)/coverSide;
        const double y1=static_cast<double>(y+1u)/coverSide;
        subdividedCover.push_back({{V2{x0,y0},V2{x1,y0},V2{x1,y1}}});
        subdividedCover.push_back({{V2{x0,y0},V2{x1,y1},V2{x0,y1}}});
    }
    double subdividedCoverage=0.0,subdividedArea=0.0;
    std::uint64_t subdividedPairs=0u,subdividedWork=0u;
    const auto subdividedDomainArray=rectangleTriangles(0.0,1.0,0.0,1.0);
    const std::vector<Triangle2> subdividedDomain(subdividedDomainArray.begin(),
                                                  subdividedDomainArray.end());
    check(singleSetCoverageOnDomain(subdividedCover,subdividedDomain,
          1.0e-14,subdividedCoverage,subdividedArea,subdividedPairs,subdividedWork)&&
          std::abs(subdividedCoverage-1.0)<1.0e-10&&subdividedWork<maximumGridWorkUnits,
          "bounded candidate-overflow subdivision failed to recover exact tiled coverage");

    const auto appendRect=[&](double x0,double x1,double y0,double y1,
                              std::vector<Triangle2>& out) {
        const auto tris=rectangleTriangles(x0,x1,y0,y1);
        out.insert(out.end(),tris.begin(),tris.end());
    };
    // A one-millimetre through-height slit occupies only 0.333% of this wall,
    // below the old aggregate area tolerance, but exact local clipping rejects it.
    const auto wallDomainArray=rectangleTriangles(0.0,0.30,0.0,0.20);
    const std::vector<Triangle2> wallDomain(wallDomainArray.begin(),wallDomainArray.end());
    std::vector<Triangle2> wallCover;
    appendRect(0.0,0.1495,0.0,0.20,wallCover);
    appendRect(0.1505,0.30,0.0,0.20,wallCover);
    double wallLocalCoverage=0.0,wallLocalArea=0.0;
    std::uint64_t wallLocalPairs=0u,wallLocalVisits=0u;
    check(singleSetCoverageOnDomain(wallCover,wallDomain,1.0e-14,wallLocalCoverage,
          wallLocalArea,wallLocalPairs,wallLocalVisits)&&
          std::abs(wallLocalCoverage-(1.0-0.001/0.30))<1.0e-10&&
          wallLocalArea*(1.0-wallLocalCoverage)>1.0e-4,
          "exact wall clipping failed to measure the narrow continuous slit");
    const double wallLocalTolerance=std::max(64.0e-14,wallLocalArea*1.0e-10);
    check(wallLocalArea*(1.0-wallLocalCoverage)>wallLocalTolerance,
          "one-millimetre local wall slit passed the accepted-domain opening gate");

    auto mirroredDomain=wallDomain,mirroredCover=wallCover;
    const auto mirrorAndReverse=[](Triangle2& triangle) {
        for (auto& point:triangle.p) point.x=0.30-point.x;
        std::swap(triangle.p[1],triangle.p[2]);
    };
    for (auto& triangle:mirroredDomain) mirrorAndReverse(triangle);
    for (auto& triangle:mirroredCover) mirrorAndReverse(triangle);
    double mirroredCoverage=0.0,mirroredArea=0.0;
    std::uint64_t mirroredPairs=0u,mirroredVisits=0u;
    check(singleSetCoverageOnDomain(mirroredCover,mirroredDomain,1.0e-14,
          mirroredCoverage,mirroredArea,mirroredPairs,mirroredVisits)&&
          std::abs(mirroredCoverage-wallLocalCoverage)<1.0e-10,
          "mirrored/reversed wall opening changed exact coverage");

    // The only permitted end opening follows the two accepted major-flap free
    // edges. Extra gaps outside that corridor remain subject to exact coverage.
    const std::vector<V2> edge0{{0.0,0.49975},{1.0,0.49975}};
    const std::vector<V2> edgeHalfMm{{0.0,0.50025},{1.0,0.50025}};
    const auto seamDomain=deriveButtSeamOutsideDomain(edge0,edgeHalfMm,
        0.0,1.0,0.0,1.0,1.0e-6,1.0e-14);
    std::array<std::vector<Triangle2>,2> seamMajor;
    appendRect(0.0,1.0,0.0,0.49975,seamMajor[0]);
    appendRect(0.0,1.0,0.50025,1.0,seamMajor[1]);
    double seamOutsideCoverage=0.0,seamOutsideArea=0.0;
    std::uint64_t seamPairs=0u,seamVisits=0u;
    check(seamDomain.valid&&std::abs(seamDomain.maximumGapM-0.0005)<1.0e-12&&
          majorUnionCoverageOnDomain(seamMajor,seamDomain.outsideCorridor,1.0e-14,
              seamOutsideCoverage,seamOutsideArea,seamPairs,seamVisits)&&
          seamOutsideCoverage>1.0-1.0e-10,
          "intended 0.5 mm authored butt seam failed the local seam-corridor test");
    auto reverseSeamEdges0=edge0,reverseSeamEdges1=edgeHalfMm;
    auto reverseSeamMajor=seamMajor;
    for (auto& point:reverseSeamEdges0) point.x=1.0-point.x;
    for (auto& point:reverseSeamEdges1) point.x=1.0-point.x;
    for (auto& panel:reverseSeamMajor) for (auto& triangle:panel) {
        for (auto& point:triangle.p) point.x=1.0-point.x;
        std::swap(triangle.p[1],triangle.p[2]);
    }
    const auto reverseSeam=deriveButtSeamOutsideDomain(reverseSeamEdges0,reverseSeamEdges1,
        0.0,1.0,0.0,1.0,1.0e-6,1.0e-14);
    double reverseCoverage=0.0,reverseArea=0.0;
    std::uint64_t reversePairs=0u,reverseVisits=0u;
    check(reverseSeam.valid&&std::abs(reverseSeam.maximumGapM-seamDomain.maximumGapM)<1.0e-12&&
          majorUnionCoverageOnDomain(reverseSeamMajor,reverseSeam.outsideCorridor,1.0e-14,
              reverseCoverage,reverseArea,reversePairs,reverseVisits)&&
          std::abs(reverseCoverage-seamOutsideCoverage)<1.0e-12,
          "mirrored/reversed authored seam changed its gap or outside coverage");
    const std::vector<V2> edgeTwoMm{{0.0,0.502},{1.0,0.502}};
    const auto wideSeam=deriveButtSeamOutsideDomain(edge0,edgeTwoMm,
        0.0,1.0,0.0,1.0,1.0e-6,1.0e-14);
    check(!wideSeam.valid,"2 mm exposed major-flap end strip exceeded the seam limit but was admitted");
    auto extraHole=seamMajor;
    extraHole[0].clear();
    appendRect(0.0,1.0,0.0,0.20,extraHole[0]);
    appendRect(0.0,1.0,0.202,0.49975,extraHole[0]);
    double extraCoverage=0.0,extraArea=0.0;
    std::uint64_t extraPairs=0u,extraVisits=0u;
    check(majorUnionCoverageOnDomain(extraHole,seamDomain.outsideCorridor,1.0e-14,
          extraCoverage,extraArea,extraPairs,extraVisits)&&
          extraArea*(1.0-extraCoverage)>extraArea*1.0e-10,
          "non-seam exposed strip escaped exact coverage outside the seam corridor");

    TriangleGrid scanGrid(seamMajor[0]);
    std::vector<std::uint32_t> scanCandidates;
    std::uint64_t countedGridWork=0u;
    const auto scanRange=scanGrid.range(seamMajor[0].front());
    const std::uint64_t scanCells=static_cast<std::uint64_t>(scanRange[1]-scanRange[0]+1u)*
                                  static_cast<std::uint64_t>(scanRange[3]-scanRange[2]+1u);
    check(scanGrid.valid&&scanGrid.query(seamMajor[0].front(),scanCandidates,&countedGridWork)&&
          scanGrid.buildWork==2u*scanGrid.entries+2u*TriangleGrid::side*TriangleGrid::side&&
          countedGridWork>=scanGrid.buildWork+scanCells,
          "grid posting construction, directory, or query-cell scans were omitted from the bounded work counter");
    TriangleGrid buildCapGrid(seamMajor[0]);
    std::uint64_t cappedBuildVisits=maximumGridWorkUnits-buildCapGrid.buildWork+1u;
    check(buildCapGrid.valid&&
          !buildCapGrid.query(seamMajor[0].front(),scanCandidates,&cappedBuildVisits),
          "spatial-grid posting-build work cap did not fail closed");
    std::vector<Triangle2> sparseCells{
        Triangle2{{V2{0.01,0.01},V2{0.02,0.01},V2{0.01,0.02}}},
        Triangle2{{V2{0.90,0.90},V2{0.91,0.90},V2{0.90,0.91}}}};
    TriangleGrid emptyCellGrid(sparseCells);
    Triangle2 emptyCell{{V2{0.45,0.45},V2{0.46,0.45},V2{0.45,0.46}}};
    const auto emptyRange=emptyCellGrid.range(emptyCell);
    const std::uint64_t emptyCellCount=static_cast<std::uint64_t>(emptyRange[1]-emptyRange[0]+1u)*
                                      static_cast<std::uint64_t>(emptyRange[3]-emptyRange[2]+1u);
    std::uint64_t emptyCellWork=0u;
    std::vector<std::uint32_t> emptyCellCandidates;
    check(emptyCellGrid.valid&&emptyCellGrid.query(emptyCell,emptyCellCandidates,&emptyCellWork)&&
          emptyCellCandidates.empty()&&
          emptyCellWork==emptyCellGrid.buildWork+emptyCellCount,
          "empty cell-directory lookups were not charged even when they contain no triangle postings");
    std::uint64_t scanPostingCount=0u;
    for (std::uint32_t y=scanRange[2];y<=scanRange[3];++y)
        for (std::uint32_t x=scanRange[0];x<=scanRange[1];++x)
            scanPostingCount+=scanGrid.offsets[y*TriangleGrid::side+x+1u]-
                              scanGrid.offsets[y*TriangleGrid::side+x];
    const std::uint64_t scanQueryWork=scanCells+scanPostingCount;
    std::uint64_t cappedCellVisits=maximumGridWorkUnits-scanCells+1u;
    const auto beforeCellCap=cappedCellVisits;
    check(!scanGrid.query(seamMajor[0].front(),scanCandidates,&cappedCellVisits)&&
          cappedCellVisits==beforeCellCap,
          "spatial-grid cell-directory work cap did not fail closed before scanning cells");
    std::uint64_t cappedQueryVisits=maximumGridWorkUnits-scanQueryWork+1u;
    check(!scanGrid.query(seamMajor[0].front(),scanCandidates,&cappedQueryVisits)&&
          cappedQueryVisits==maximumGridWorkUnits,
          "spatial-grid posting scan cap did not fail closed");

    auto steppedMinor=f;
    for (const std::uint32_t panel:{1u,3u}) {
        const std::size_t bottomBase=(static_cast<std::size_t>(panel)*3u+1u)*8u;
        const std::size_t topBase=(static_cast<std::size_t>(panel)*3u+2u)*8u;
        for (std::size_t i=0;i<8u;++i) {
            steppedMinor.nodes[bottomBase+i].positionAndMass.y+=0.01f;
            steppedMinor.nodes[topBase+i].positionAndMass.y-=0.01f;
        }
    }
    const auto steppedResult=measureAcceptedInsideClearance(steppedMinor.blank,
        steppedMinor.nodes,static_cast<std::uint32_t>(steppedMinor.nodes.size()),
        steppedMinor.tets,steppedMinor.faces,{0.30,0.20,0.20});
    check(!steppedResult.gatePassed&&steppedResult.planarGeometryValid&&
          !steppedResult.dimensionsWithinTolerance&&
          std::abs(steppedResult.endMajorPlaneSeparationM[0])<1.0e-6&&
          std::abs(steppedResult.endMajorPlaneSeparationM[1])<1.0e-6&&
          std::abs(steppedResult.measuredInsideM[1]-0.18)<1.0e-5,
          "major-flap target plane concealed a reduced true cavity height from minor-flap thickness steps");

    auto incompletePanels=f.blank;
    incompletePanels.panels.pop_back();
    const auto incompletePanelResult=measureAcceptedInsideClearance(incompletePanels,
        f.nodes,static_cast<std::uint32_t>(f.nodes.size()),f.tets,f.faces,{0.30,0.20,0.20});
    check(!incompletePanelResult.gatePassed,
          "incomplete authored panel map was admitted");

    auto rotated=f;
    for (auto& n:rotated.nodes) {
        const double x=n.positionAndMass.x,y=n.positionAndMass.y,z=n.positionAndMass.z;
        n.positionAndMass.x=static_cast<float>(0.6*x-0.8*z+0.7);
        n.positionAndMass.y=static_cast<float>(y-1.2);
        n.positionAndMass.z=static_cast<float>(0.8*x+0.6*z+2.1);
    }
    auto rotatedResult=measureAcceptedInsideClearance(rotated.blank,rotated.nodes,
        static_cast<std::uint32_t>(rotated.nodes.size()),rotated.tets,rotated.faces,{0.30,0.20,0.20});
    if (!rotatedResult.gatePassed) std::cerr << "rotated fixture rejected: " << rotatedResult.reason << '\n';
    check(rotatedResult.gatePassed,"rigidly rotated/translated accepted box changed clearance result");

    auto coarseFloatCoordinates=f;
    for (auto& node:coarseFloatCoordinates.nodes) {
        node.positionAndMass.x+=4096.0f;
        node.positionAndMass.y+=4096.0f;
        node.positionAndMass.z+=4096.0f;
    }
    const auto coarsePrecisionResult=measureAcceptedInsideClearance(
        coarseFloatCoordinates.blank,coarseFloatCoordinates.nodes,
        static_cast<std::uint32_t>(coarseFloatCoordinates.nodes.size()),
        coarseFloatCoordinates.tets,coarseFloatCoordinates.faces,{0.30,0.20,0.20});
    check(!coarsePrecisionResult.gatePassed&&
          coarsePrecisionResult.reason==
              "accepted FP32 world-coordinate spacing is too coarse for local opening checks"&&
          coarsePrecisionResult.acceptedPositionUlpM>maximumCoveragePositionUlpM,
          "coarse FP32 world coordinates were allowed to hide local openings behind roundoff");

    auto missing=f;
    missing.faces.pop_back();
    auto missingResult=measureAcceptedInsideClearance(missing.blank,missing.nodes,
        static_cast<std::uint32_t>(missing.nodes.size()),missing.tets,missing.faces,{0.30,0.20,0.20});
    check(!missingResult.gatePassed&&!missingResult.topologyComplete,
          "missing compiled exterior face was not rejected");

    auto buckled=f;
    buckled.nodes[0].positionAndMass.z+=0.02f;
    auto buckleResult=measureAcceptedInsideClearance(buckled.blank,buckled.nodes,
        static_cast<std::uint32_t>(buckled.nodes.size()),buckled.tets,buckled.faces,{0.30,0.20,0.20});
    check(!buckleResult.gatePassed,"buckled accepted panel passed planar clearance gate");

    auto overlapGap=makeFixture(true);
    auto gapResult=measureAcceptedInsideClearance(overlapGap.blank,overlapGap.nodes,
        static_cast<std::uint32_t>(overlapGap.nodes.size()),overlapGap.tets,overlapGap.faces,{0.30,0.20,0.20});
    check(!gapResult.gatePassed&&
          (!gapResult.buttSeamGeometryValid||gapResult.maximumButtSeamGapM[0]>maximumButtSeamGapM),
          "uncovered minor flap was admitted as a closed end");

    auto intendedSeam=makeFixture(false,0.0005);
    const auto intendedSeamResult=measureAcceptedInsideClearance(intendedSeam.blank,
        intendedSeam.nodes,static_cast<std::uint32_t>(intendedSeam.nodes.size()),
        intendedSeam.tets,intendedSeam.faces,{0.30,0.20,0.20});
    check(intendedSeamResult.gatePassed&&intendedSeamResult.endOutsideSeamCoverageComplete&&
          intendedSeamResult.maximumButtSeamGapM[0]>0.00049&&
          intendedSeamResult.maximumButtSeamGapM[0]<0.00051,
          "bounded intended butt seam did not pass with exact outside-corridor coverage");
    auto twoMmSeam=makeFixture(false,0.002);
    const auto twoMmSeamResult=measureAcceptedInsideClearance(twoMmSeam.blank,
        twoMmSeam.nodes,static_cast<std::uint32_t>(twoMmSeam.nodes.size()),
        twoMmSeam.tets,twoMmSeam.faces,{0.30,0.20,0.20});
    check(!twoMmSeamResult.gatePassed&&
          twoMmSeamResult.reason=="accepted major-flap free edges do not define a bounded authored butt seam",
          "two-millimetre end gap was admitted by the one-percent area allowance");

    auto outOfFootprint=f;
    for (const std::uint32_t panel:{0u,2u}) for (const std::uint32_t region:{1u,2u}) {
        const std::size_t base=(static_cast<std::size_t>(panel)*3u+region)*8u;
        for (std::size_t i=0;i<8u;++i)
            outOfFootprint.nodes[base+i].positionAndMass.x+=0.002f;
    }
    const auto outOfFootprintResult=measureAcceptedInsideClearance(outOfFootprint.blank,
        outOfFootprint.nodes,static_cast<std::uint32_t>(outOfFootprint.nodes.size()),
        outOfFootprint.tets,outOfFootprint.faces,{0.30,0.20,0.20});
    check(!outOfFootprintResult.gatePassed&&
          outOfFootprintResult.reason ==
              "last-closing flap surface extends outside the wall-derived cavity footprint",
          "last-closing flap outside the wall-derived footprint did not fail at the footprint check");

    auto missingArea=f;
    missingArea.blank.slots.push_back({0u,false,0.320,0.330,0.0,0.20});
    const auto missingAreaResult=measureAcceptedInsideClearance(missingArea.blank,
        missingArea.nodes,static_cast<std::uint32_t>(missingArea.nodes.size()),
        missingArea.tets,missingArea.faces,{0.30,0.20,0.20});
    check(!missingAreaResult.gatePassed&&!missingAreaResult.authoredSurfaceCoverageComplete,
          "accepted surface area inconsistent with authored slot coverage was admitted");

    auto nonfinite=f;
    nonfinite.nodes[0].positionAndMass.x=std::numeric_limits<float>::quiet_NaN();
    auto nanResult=measureAcceptedInsideClearance(nonfinite.blank,nonfinite.nodes,
        static_cast<std::uint32_t>(nonfinite.nodes.size()),nonfinite.tets,nonfinite.faces,{0.30,0.20,0.20});
    check(!nanResult.gatePassed,"nonfinite accepted geometry passed clearance gate");

    auto badSchedule=f.blank;
    std::swap(badSchedule.assembly.actions[0],badSchedule.assembly.actions[2]);
    check(!scheduleHasInnerBeforeOuterFlaps(badSchedule),"wrong flap closure order was accepted");

    const std::array<BoxBlankPoint,3> wall{{{0.05,0.25,0.0},{0.15,0.25,0.0},{0.10,0.35,0.0}}};
    const auto wallTag=classifyFlatFace(f.blank,wall,1.0e-6);
    check(wallTag.classified&&wallTag.region==BoxBlankFaceRegion::wall&&wallTag.panelIndex==0u&&
          wallTag.broadSide==BoxBlankBroadSide::zZero,"authored wall role tag is wrong");
    std::cout << "source159 inside-surface tests: PASS\n";
}
}

int main() {
    try { run(); return 0; }
    catch (const std::exception& e) {
        std::cerr << "source159 inside-surface tests: FAIL: " << e.what() << '\n';
        return 1;
    }
}
