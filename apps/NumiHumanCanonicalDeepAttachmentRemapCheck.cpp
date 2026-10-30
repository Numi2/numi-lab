#include "NumiHumanCanonicalDeepAttachmentRemap.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace metalrobo::human::canonical_deep_attachment_remap;

struct Site {
    std::uint32_t reducedVertex;
    std::uint64_t sourceVertex;
    std::uint32_t donorBody;
};
struct Request {
    std::uint64_t stableAttachmentID;
    std::uint32_t donorBody;
    std::uint32_t stableSourceVertex;
    std::string sourceDomain;
};
struct Float4 { float x=0,y=0,z=0,w=0; };
struct Anchor { std::uint32_t bodyIndex=0,flags=0; Float4 localPoint{}; };
struct Sample {
    std::uint32_t slaveNode=0,bodyIndex=0,flags=0,reserved0=0;
    Float4 bodyLocalPointAndArea{},bodyLocalNormalAndReferenceSeparation{};
    Float4 stiffnessAndNormalStrainPerPressure{};
};

static std::array<Site,14> sites() {
    // Exact immutable rows from kWholeBodySkinDeepAttachmentSites. `sourceVertex`
    // is the canonical stable attachment ID; `reducedVertex` is its old D row.
    // Neither is the derived current source-vertex row used by Request.
    return {{{790u,13555u,20u},{1333u,29464u,23u},{804u,14506u,139u},
             {1719u,37264u,152u},{1790u,39356u,153u},{1557u,33621u,20u},
             {1478u,32578u,23u},{752u,11704u,139u},{1842u,41702u,152u},
             {1831u,40968u,153u},{1591u,34352u,20u},{1104u,21455u,23u},
             {692u,8467u,139u},{1987u,46481u,153u}}};
}
static std::vector<Request> requests(const std::array<Site,14>& s) {
    std::vector<Request> out;
    for(std::uint32_t i=0;i<s.size();++i)
        out.push_back({s[i].sourceVertex,s[i].donorBody,800u+i,"current-D-domain"});
    return out;
}
static SourceLineage lineage(const Site& s,const Request& r,const std::uint32_t i) {
    SourceLineage row;
    row.stableAttachmentID=r.stableAttachmentID;
    row.donorBody=r.donorBody;
    row.derivedSourceVertex=r.stableSourceVertex;
    row.sourceDomain=r.sourceDomain;
    row.surfaceVertexRow=100u+i;
    row.volumeNodeRow=500u+i;
    row.surfaceSourceVertex=r.stableSourceVertex;
    row.surfaceSourceFace=-1;
    row.surfaceSourceDomain=r.sourceDomain;
    row.volumeSourceVertex=r.stableSourceVertex;
    row.volumeSourceFace=-1;
    row.volumeSourceDomain=r.sourceDomain;
    row.surfaceParentTet=1000u+i;
    row.volumeParentTet=2000u+i; // two valid source tets may differ but both share the old support node.
    row.originalParentVolumeNode=3000u+i;
    row.sourceParentTetNodes={row.originalParentVolumeNode,10u,11u,12u};
    row.volumeParentTetNodes={20u,row.originalParentVolumeNode,21u,22u};
    row.surfaceDonor=r.donorBody;
    row.frameOwnerDonor=r.donorBody;
    row.massOwnerDonor=r.donorBody;
    row.sourceParentGeometricDonor=900u+i; // deliberately not overloaded as frame owner.
    row.sourceParentRegion=70u+i;
    row.donorFull86Weight=0.96;
    row.incidentAltitudeM=0.0002;
    (void)s;
    return row;
}
static bool validateAll(const std::array<Site,14>& s,const std::vector<Request>& r,
                        std::vector<const Site*>& bound,std::string& error) {
    if(!authenticateCanonicalRequests(s,r,"current-D-domain",bound,error))return false;
    if(bound.size()!=14u)return false;
    for(std::uint32_t i=0;i<r.size();++i)
        if(!validateSourceLineage(*bound[i],r[i],lineage(*bound[i],r[i],i),
                                  0.85,5.0e-5,error))return false;
    return true;
}
static bool sameResponse(const Sample& a,const Sample& b,const std::array<double,3>& d) {
    const auto x=elasticResponseForDisplacement(a,d);
    const auto y=elasticResponseForDisplacement(b,d);
    return x==y;
}
int main() {
    auto canonical=sites();
    auto mapped=requests(canonical);
    std::vector<const Site*> bound;
    std::string error;
    assert(validateAll(canonical,mapped,bound,error));

    auto bad=mapped;
    bad[2].stableAttachmentID=99999u;
    assert(!authenticateCanonicalRequests(canonical,bad,"current-D-domain",bound,error));
    bad=mapped;bad[2].donorBody+=1u;
    assert(!authenticateCanonicalRequests(canonical,bad,"current-D-domain",bound,error));
    bad=mapped;bad[2].sourceDomain="wrong-domain";
    assert(!authenticateCanonicalRequests(canonical,bad,"current-D-domain",bound,error));
    auto duplicateSites=canonical;
    duplicateSites[2].donorBody=duplicateSites[1].donorBody;
    auto duplicateRows=requests(duplicateSites);
    duplicateRows[2].stableSourceVertex=duplicateRows[1].stableSourceVertex;
    assert(!authenticateCanonicalRequests(duplicateSites,duplicateRows,
                                          "current-D-domain",bound,error));
    bad=mapped;bad[2].stableAttachmentID=bad[1].stableAttachmentID;
    assert(!authenticateCanonicalRequests(canonical,bad,"current-D-domain",bound,error));
    auto missing=std::vector<Request>(mapped.begin(),mapped.end()-1);
    assert(!authenticateCanonicalRequests(canonical,missing,"current-D-domain",bound,error));

    // A current-domain row may numerically equal a different canonical ID; the
    // fields remain separate namespaces, so canonical lookup still binds by ID/body.
    auto namespaceCollision=mapped;
    namespaceCollision[2].stableSourceVertex=static_cast<std::uint32_t>(canonical[3].sourceVertex);
    assert(authenticateCanonicalRequests(canonical,namespaceCollision,
                                         "current-D-domain",bound,error));
    assert(validateSourceLineage(*bound[2],namespaceCollision[2],
            lineage(*bound[2],namespaceCollision[2],2),0.85,5.0e-5,error));

    auto l=lineage(canonical[0],mapped[0],0);
    l.surfaceSourceVertex++;
    assert(!validateSourceLineage(canonical[0],mapped[0],l,0.85,5.0e-5,error));
    l=lineage(canonical[0],mapped[0],0);l.surfaceSourceFace=12;
    assert(!validateSourceLineage(canonical[0],mapped[0],l,0.85,5.0e-5,error));
    l=lineage(canonical[0],mapped[0],0);l.volumeSourceDomain="other";
    assert(!validateSourceLineage(canonical[0],mapped[0],l,0.85,5.0e-5,error));
    l=lineage(canonical[0],mapped[0],0);l.volumeParentTetNodes={1,2,3,4};
    assert(!validateSourceLineage(canonical[0],mapped[0],l,0.85,5.0e-5,error));
    l=lineage(canonical[0],mapped[0],0);l.frameOwnerDonor++;
    assert(!validateSourceLineage(canonical[0],mapped[0],l,0.85,5.0e-5,error));
    l=lineage(canonical[0],mapped[0],0);l.massOwnerDonor++;
    assert(!validateSourceLineage(canonical[0],mapped[0],l,0.85,5.0e-5,error));
    l=lineage(canonical[0],mapped[0],0);l.donorFull86Weight=0.849999;
    assert(!validateSourceLineage(canonical[0],mapped[0],l,0.85,5.0e-5,error));
    l=lineage(canonical[0],mapped[0],0);l.incidentAltitudeM=4.999e-5;
    assert(!validateSourceLineage(canonical[0],mapped[0],l,0.85,5.0e-5,error));
    l=lineage(canonical[0],mapped[0],0);
    assert(validateSourceLineage(canonical[0],mapped[0],l,0.85,5.0e-5,error));

    constexpr std::uint32_t active=1u,elastic=2u,body=23u;
    Anchor oldAnchor{body,active,{1.0f,2.0f,3.0f,0.0f}};
    Anchor newAnchor{};
    assert(rebindHardAnchorToCurrentRest(oldAnchor,body,active,{4.0f,5.0f,6.0f},newAnchor));
    assert(newAnchor.bodyIndex==body&&newAnchor.flags==active);
    assert(newAnchor.localPoint.x==4.0f&&newAnchor.localPoint.y==5.0f&&
           newAnchor.localPoint.z==6.0f&&newAnchor.localPoint.w==0.0f);
    assert(newAnchor.localPoint.x!=oldAnchor.localPoint.x);
    assert(!rebindHardAnchorToCurrentRest(oldAnchor,body+1,active,{4,5,6},newAnchor));
    assert(!rebindHardAnchorToCurrentRest(oldAnchor,body,active+1,{4,5,6},newAnchor));

    Sample oldSample{};
    oldSample.slaveNode=111;oldSample.bodyIndex=body;oldSample.flags=active|elastic;
    oldSample.bodyLocalPointAndArea={1.0f,2.0f,3.0f,2.5e-4f};
    oldSample.bodyLocalNormalAndReferenceSeparation={0.0f,1.0f,0.0f,0.0f};
    oldSample.stiffnessAndNormalStrainPerPressure={3.2e7f,2.0e-8f,0.0f,0.0f};
    Sample currentSample{};
    assert(rebindElasticSampleToCurrentRest(oldSample,777,body,active|elastic,
            {4.0f,5.0f,6.0f},{0.0f,0.0f,1.0f},currentSample));
    assert(currentSample.slaveNode==777&&currentSample.bodyIndex==body);
    assert(currentSample.bodyLocalPointAndArea.x==4.0f&&
           currentSample.bodyLocalPointAndArea.y==5.0f&&
           currentSample.bodyLocalPointAndArea.z==6.0f);
    assert(currentSample.bodyLocalPointAndArea.w==oldSample.bodyLocalPointAndArea.w);
    assert(currentSample.stiffnessAndNormalStrainPerPressure.x==
           oldSample.stiffnessAndNormalStrainPerPressure.x);
    assert(currentSample.stiffnessAndNormalStrainPerPressure.y==
           oldSample.stiffnessAndNormalStrainPerPressure.y);
    assert(currentSample.stiffnessAndNormalStrainPerPressure.z==
           oldSample.stiffnessAndNormalStrainPerPressure.z);
    assert(currentSample.stiffnessAndNormalStrainPerPressure.w==
           oldSample.stiffnessAndNormalStrainPerPressure.w);
    assert(currentSample.bodyLocalNormalAndReferenceSeparation.w==0.0f);
    const auto zero=elasticResponseForDisplacement(currentSample,{0.0,0.0,0.0});
    assert((zero.first==0.0&&zero.second==std::array<double,3>{0.0,0.0,0.0}));
    const std::array<std::array<double,3>,3> displacements{{
        {0.001,0.0,0.0},{0.0,-0.0002,0.0004},{0.001,-0.002,0.003}}};
    for(const auto& d:displacements)
        assert(sameResponse(oldSample,currentSample,d));
    auto invalid=oldSample;invalid.bodyIndex=body+1;
    assert(!rebindElasticSampleToCurrentRest(invalid,777,body,active|elastic,
            {4,5,6},{0,0,1},currentSample));
    invalid=oldSample;invalid.bodyLocalPointAndArea.w=0.0f;
    assert(!rebindElasticSampleToCurrentRest(invalid,777,body,active|elastic,
            {4,5,6},{0,0,1},currentSample));
    invalid=oldSample;
    invalid.stiffnessAndNormalStrainPerPressure.x=
        std::numeric_limits<float>::infinity();
    assert(!rebindElasticSampleToCurrentRest(invalid,777,body,active|elastic,
            {4,5,6},{0,0,1},currentSample));
    invalid=oldSample;
    invalid.stiffnessAndNormalStrainPerPressure.y=
        std::numeric_limits<float>::infinity();
    assert(!rebindElasticSampleToCurrentRest(invalid,777,body,active|elastic,
            {4,5,6},{0,0,1},currentSample));
    assert(!rebindElasticSampleToCurrentRest(oldSample,777,body,active|elastic,
            {4,5,6},{0,0,0},currentSample));

    std::cout<<"canonical-remap-helper=PASS canonical_rows=14 request_rejections=6"
             <<" source_lineage_rejections=8 numeric_namespace_collision=1"
             <<" hard_anchor_reference_rebound=1 elastic_area_stiffness_exact=1"
             <<" zero_prestress=1 matched_force_energy_parity=3"
             <<" native_admission=0 runtime=0 accepted_steps=0\n";
}
