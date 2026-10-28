#include "NumiHumanPhysicalSkinChildRegisteredVolumeImport.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

namespace {
V3 transform(const std::array<double,9>& a,const V3& p,const V3& t) {
    return {a[0]*p[0]+a[1]*p[1]+a[2]*p[2]+t[0],
            a[3]*p[0]+a[4]*p[1]+a[5]*p[2]+t[1],
            a[6]*p[0]+a[7]*p[1]+a[8]*p[2]+t[2]};
}
std::array<double,4> betaFor(std::size_t row) {
    const double f=double(row+1u);
    return {0.1+0.001*f,0.2-0.0005*f,0.3-0.0002*f,
            0.4-0.0003*f};
}
void addPoint(V3& p,const V3& q,double scale) {
    for(unsigned a=0;a<3;++a)p[a]+=scale*q[a];
}
void requireReject(PhysicalSkinChildMassMappingBundle bundle,
        PhysicalSkinRegisteredSurfaceCSVSet registered,
        std::vector<PhysicalSkinRegisteredVolumeNodeRow> rows,
        const std::vector<PhysicalSkinStableDeepAttachmentRequest>& attachments,
        const std::vector<PhysicalSkinRegisteredBodyPose>& bodyPoses,
        const std::string& expected) {
    PhysicalSkinRegisteredVolumeImportSidecars sidecars;
    PhysicalSkinRegisteredVolumeImportReport report;
    std::string error;
    const bool accepted=compilePhysicalSkinRegisteredVolumeImport(
        bundle,registered,rows,attachments,bodyPoses,157u,0.1,"fixture-detail",32u,16u,
        sidecars,report,error);
    assert(!accepted);
    if(error.find(expected)==std::string::npos){std::cerr<<"expected: "<<expected<<" got: "<<error<<std::endl;std::abort();}
}
}

int main() {
    constexpr std::size_t nodeCount=12u;
    const std::array<double,9> affine{1.02,0.01,0.0,0.0,0.99,0.02,0.01,0.0,1.01};
    const V3 translation{0.003,-0.002,0.004};
    const std::array<V3,4> parentReference{{
        V3{0,0,0},V3{1,0,0},V3{0,1,0},V3{0,0,1}}};
    PhysicalSkinChildMassMappingBundle bundle;
    bundle.parent.referenceWorldPoints.assign(parentReference.begin(),parentReference.end());
    for(const auto& p:parentReference)bundle.parent.currentWorldPoints.push_back(transform(affine,p,translation));
    bundle.parent.referenceWorldPoints.insert(bundle.parent.referenceWorldPoints.end(),
        parentReference.begin(),parentReference.end());
    for(const auto& p:parentReference)bundle.parent.currentWorldPoints.push_back(transform(affine,p,translation));
    bundle.parent.tetrahedra={{0,1,2,3},{4,5,6,7}};
    bundle.parent.materialIndices={15u,16u};
    bundle.parent.donorBodyIndices={0u,0u};
    bundle.parent.anatomicalRegionIndices={42u,42u};
    bundle.referenceWorldPoints.resize(nodeCount);
    bundle.childTetrahedra={{0,1,2,9},{3,4,5,9},{3,4,5,10},{6,7,8,10},{6,7,10,11}};
    bundle.childMaterialIndices={15u,15u,16u,16u,16u};
    bundle.childDonorBodyIndices={0u,0u,0u,0u,0u};
    bundle.childAnatomicalRegionIndices={42u,42u,42u,42u,42u};
    bundle.childStableSourceFaceIdentities={98586u,98586u,98586u,98586u,98586u};

    PhysicalSkinRegisteredSurfaceCSVSet registered;
    registered.framesByDonor.resize(157u);
    std::vector<PhysicalSkinRegisteredBodyPose> allBodyPoses(157u);
    for(std::size_t body=0;body<allBodyPoses.size();++body){
        allBodyPoses[body].bodyIndex=static_cast<std::uint32_t>(body);
        allBodyPoses[body].binding=NM_INVALID_INDEX;
        allBodyPoses[body].available=true;
        allBodyPoses[body].currentOrigin={0,0,0};allBodyPoses[body].restOrigin={0,0,0};
        allBodyPoses[body].currentLocalToWorld={1,0,0,0,1,0,0,0,1};
        allBodyPoses[body].restLocalToWorld={1,0,0,0,1,0,0,0,1};
    }
    auto& frame=registered.framesByDonor[0];
    frame.available=true;frame.binding=0u;frame.currentOrigin={0,0,0};frame.restOrigin={0,0,0};
    frame.currentR={1,0,0,0,1,0,0,0,1};frame.restR=frame.currentR;
    allBodyPoses[0].binding=0u;

    std::vector<PhysicalSkinRegisteredVolumeNodeRow> rows(nodeCount);
    const std::array<PhysicalSkinRegisteredVolumeNodeRole,9> roles{{
        PhysicalSkinRegisteredVolumeNodeRole::outerO,PhysicalSkinRegisteredVolumeNodeRole::outerO,
        PhysicalSkinRegisteredVolumeNodeRole::outerO,PhysicalSkinRegisteredVolumeNodeRole::innerI,
        PhysicalSkinRegisteredVolumeNodeRole::innerI,PhysicalSkinRegisteredVolumeNodeRole::innerI,
        PhysicalSkinRegisteredVolumeNodeRole::deepD,PhysicalSkinRegisteredVolumeNodeRole::deepD,
        PhysicalSkinRegisteredVolumeNodeRole::deepD}};
    std::array<std::vector<std::uint32_t>,3> boundaryRows;
    for(std::size_t node=0;node<nodeCount;++node) {
        auto& row=rows[node];
        row.volumeNodeRow=static_cast<std::uint32_t>(node);
        row.role=node<9u?roles[node]:PhysicalSkinRegisteredVolumeNodeRole::interior;
        row.surfaceVertexRow=node<9u?static_cast<std::uint32_t>(node%3u):NM_INVALID_INDEX;
        row.donorBody=node==9u?156u:0u;
        row.donorBinding=allBodyPoses[row.donorBody].binding;
        row.geometricDonorBody=0u;row.anatomicalRegion=42u;
        row.incidentMaterialMask=(node<3u||node==9u)?(1u<<15u):
                                 (node<6u)?((1u<<15u)|(1u<<16u)):(1u<<16u);
        row.sourceMappingKind=node<9u?"vertex":"none";
        row.sourceDomain=node<9u?"fixture-detail":"derived_volume_interior_no_surface_ancestry";
        row.frameMethod=node<9u?"fixture-exact-source":"derived_full_all_bodies_geometric_donor_pose";
        row.sourceVertex=node<9u?static_cast<std::uint32_t>(node):NM_INVALID_INDEX;
        row.parentRegistrationKind=node<9u?"source_parent_tet_reference_registration":
                                           "inferred_current_parent_tet_same_material_donor_region";
        row.parentTet=(node<6u||node==9u)?0u:1u;row.parentTieCount=0u;row.parentBarycentric=betaFor(node);
        const V3 referenceResidual{0.0002,-0.0003,0.0004};
        const auto parentTet=bundle.parent.tetrahedra[row.parentTet];
        V3 target{};
        for(unsigned c=0;c<4;++c)addPoint(target,bundle.parent.referenceWorldPoints[parentTet[c]],row.parentBarycentric[c]);
        target=add(target,referenceResidual);
        bundle.referenceWorldPoints[node]=target;
        row.currentWorldPoint=target;
        V3 currentBase{};
        for(unsigned c=0;c<4;++c)addPoint(currentBase,bundle.parent.currentWorldPoints[parentTet[c]],row.parentBarycentric[c]);
        row.parentCurrentResidualWorldM=sub(target,currentBase);
        row.parentRegistrationDistanceM=norm(row.parentCurrentResidualWorldM);
        row.donorLocalPoint=row.currentWorldPoint;

        if(node<9u) {
            const unsigned layer=node/3u;
            boundaryRows[layer].push_back(static_cast<std::uint32_t>(node));
            auto& surface=registered.surfaces[layer];
            surface.worldPoints.push_back(row.currentWorldPoint);
            surface.donorLocalPoints.push_back(row.donorLocalPoint);
            surface.donorBodyIndices.push_back(row.donorBody);
            surface.worldToDonorLocalLinear.push_back({1,0,0,0,1,0,0,0,1});
            surface.outwardWorldNormals.push_back({0,0,1});
            surface.sourceVertexIndices.push_back(row.sourceVertex);
            surface.sourceEmbeddingFaceIndices.push_back(NM_INVALID_INDEX);
            surface.sourceEmbeddingBarycentric.push_back({});
            surface.sourceEmbeddingResidualWorldM.push_back({});
            PhysicalSkinNodeSourceRow src;
            src.vertex=row.sourceVertex;src.face=-1;src.domain=row.sourceDomain;src.frameMethod=row.frameMethod;
            registered.nodeSource[layer].push_back(src);
            PhysicalSkinParentRegistrationRow parent;
            parent.available=true;parent.tet=row.parentTet;parent.tieCount=row.parentTieCount;
            parent.bary=row.parentBarycentric;parent.residual=row.parentCurrentResidualWorldM;
            parent.distance=row.parentRegistrationDistanceM;
            registered.nodeParent[layer].push_back(parent);
        }
        for(auto& surface:registered.surfaces) {
            if(surface.faces.empty())surface.faces.push_back({0,1,2});
        }
    }
    const std::vector<PhysicalSkinStableDeepAttachmentRequest> attachments{
        {1001u,0u,6u,"fixture-detail"},{1002u,0u,8u,"fixture-detail"}};

    const PhysicalSkinChildMassMappingBundle inputBundle=bundle;
    PhysicalSkinRegisteredVolumeImportSidecars sidecars;
    PhysicalSkinRegisteredVolumeImportReport report;
    std::string error;
    if(!compilePhysicalSkinRegisteredVolumeImport(bundle,registered,rows,attachments,allBodyPoses,157u,0.1,
        "fixture-detail",32u,16u,sidecars,report,error)){std::cerr<<error<<std::endl;return 1;}
    assert(report.volumeNodeCount==nodeCount&&report.registeredParentRows==nodeCount);
    assert((report.mappedBoundaryNodes==std::array<std::size_t,3>{3u,3u,3u}));
    assert(report.oneSourceDomain&&report.canonicalDetailedSourceMapPopulated);
    assert(report.sourceIdentityRows==9u);
    assert(rows[9].donorBody==156u&&rows[9].donorBinding==NM_INVALID_INDEX);
    const std::vector<PhysicalSkinSourceDomainRegistration> typedDomains{
        {"fixture-detail",PhysicalSkinSourceDomainRole::outerSurface,std::string(64,'1'),9u,16u},
        {"fixture-detail",PhysicalSkinSourceDomainRole::innerSurface,std::string(64,'2'),9u,16u},
        {"fixture-detail",PhysicalSkinSourceDomainRole::deepSurface,std::string(64,'3'),9u,16u}};
    auto typedBundle=inputBundle;
    PhysicalSkinRegisteredVolumeImportSidecars typedSidecars;
    PhysicalSkinRegisteredVolumeImportReport typedReport;
    std::string typedError;
    if(!compilePhysicalSkinRegisteredVolumeImport(typedBundle,registered,rows,attachments,allBodyPoses,157u,0.1,
        "",0u,0u,typedSidecars,typedReport,typedError,std::span<const PhysicalSkinSourceDomainRegistration>(typedDomains))){
        std::cerr<<"typed mixed source import failed: "<<typedError<<std::endl;return 3;
    }
    assert(typedReport.typedSourceDomainMapPopulated&&!typedReport.canonicalDetailedSourceMapPopulated);
    assert(!typedReport.oneSourceDomain&&typedReport.sourceDomainCount==3u);
    assert(typedBundle.detailedSourceVertexCount==0u&&typedBundle.detailedSourceFaceCount==0u);
    assert(typedBundle.volumeNodeSourceDomainIndices[0]==0u&&
           typedBundle.volumeNodeSourceDomainIndices[3]==1u&&
           typedBundle.volumeNodeSourceDomainIndices[6]==2u&&
           typedBundle.volumeNodeSourceDomainIndices[9]==NM_INVALID_INDEX);
    assert(typedBundle.outerSkin.sourceDomainIndices==std::vector<std::uint32_t>({0u,0u,0u}));
    assert(typedBundle.independentSkinInner.sourceDomainIndices==std::vector<std::uint32_t>({1u,1u,1u}));
    assert(typedBundle.independentFatDeep.sourceDomainIndices==std::vector<std::uint32_t>({2u,2u,2u}));
    auto typedTopology=typedBundle.topologyView();
    assert(validatePhysicalSkinSourceDomains(typedBundle.outerSkin,typedTopology,typedError));
    const auto rejectsTyped=[&](const std::vector<PhysicalSkinRegisteredVolumeNodeRow>& badRows,
            const std::vector<PhysicalSkinSourceDomainRegistration>& badDomains,
            const std::string& expected){
        auto badBundle=inputBundle;PhysicalSkinRegisteredVolumeImportSidecars badSidecars;
        PhysicalSkinRegisteredVolumeImportReport badReport;std::string badError;
        const bool accepted=compilePhysicalSkinRegisteredVolumeImport(badBundle,registered,badRows,attachments,
            allBodyPoses,157u,0.1,"",0u,0u,badSidecars,badReport,badError,
            std::span<const PhysicalSkinSourceDomainRegistration>(badDomains));
        assert(!accepted);
        if(badError.find(expected)==std::string::npos){
            std::cerr<<"expected typed rejection: "<<expected<<" got: "<<badError<<std::endl;std::abort();
        }
    };
    auto missingRoleDomains=typedDomains;missingRoleDomains.erase(missingRoleDomains.begin()+1);
    rejectsTyped(rows,missingRoleDomains,"not pinned");
    auto duplicateRoleDomains=typedDomains;duplicateRoleDomains.push_back(typedDomains.front());
    rejectsTyped(rows,duplicateRoleDomains,"absent, duplicated, or ambiguous");
    auto wrongHashDomains=typedDomains;wrongHashDomains[0].sourceSHA256[2]='g';
    rejectsTyped(rows,wrongHashDomains,"invalid, unpinned, empty, or duplicate");
    auto shortDomains=typedDomains;shortDomains[2].vertexCount=8u;
    rejectsTyped(rows,shortDomains,"exceeds its pinned namespace");
    auto fabricatedInterior=rows;fabricatedInterior[9].sourceDomain="fixture-detail";
    fabricatedInterior[9].sourceMappingKind="vertex";fabricatedInterior[9].sourceVertex=0u;
    rejectsTyped(fabricatedInterior,typedDomains,"interior node must carry explicit no-surface-ancestry");
    auto mixedCanonicalBundle=inputBundle;PhysicalSkinRegisteredVolumeImportSidecars mixedCanonicalSidecars;
    PhysicalSkinRegisteredVolumeImportReport mixedCanonicalReport;std::string mixedCanonicalError;
    assert(!compilePhysicalSkinRegisteredVolumeImport(mixedCanonicalBundle,registered,rows,attachments,
        allBodyPoses,157u,0.1,"fixture-detail",9u,16u,mixedCanonicalSidecars,mixedCanonicalReport,
        mixedCanonicalError,std::span<const PhysicalSkinSourceDomainRegistration>(typedDomains)));
    assert(mixedCanonicalError.find("cannot be mixed")!=std::string::npos);
    auto wrongRoleBundle=typedBundle;wrongRoleBundle.independentFatDeep.sourceDomainIndices[0]=1u;
    auto wrongRoleTopology=wrongRoleBundle.topologyView();
    assert(!validatePhysicalSkinSourceDomains(wrongRoleBundle.outerSkin,wrongRoleTopology,typedError));
    auto wrongBoundsBundle=typedBundle;wrongBoundsBundle.sourceDomains[2].vertexCount=8u;
    auto wrongBoundsTopology=wrongBoundsBundle.topologyView();
    assert(!validatePhysicalSkinSourceDomains(wrongBoundsBundle.outerSkin,wrongBoundsTopology,typedError));
    auto missingNamespaceBundle=typedBundle;missingNamespaceBundle.volumeNodeSourceDomainIndices.clear();
    auto missingNamespaceTopology=missingNamespaceBundle.topologyView();
    assert(!validatePhysicalSkinSourceDomains(missingNamespaceBundle.outerSkin,missingNamespaceTopology,typedError));
    assert(rows[3].incidentMaterialMask==((1u<<15u)|(1u<<16u)));

    auto loadedTetGenBundle=inputBundle;
    loadedTetGenBundle.currentWorldPoints=loadedTetGenBundle.referenceWorldPoints;
    PhysicalSkinRegisteredVolumeImportSidecars loadedSidecars;
    PhysicalSkinRegisteredVolumeImportReport loadedReport;
    std::string loadedError;
    if(!compilePhysicalSkinRegisteredVolumeImport(loadedTetGenBundle,registered,rows,attachments,
        allBodyPoses,157u,0.1,"fixture-detail",32u,16u,loadedSidecars,loadedReport,loadedError)){
        std::cerr<<"loaded TetGen state failed: "<<loadedError<<std::endl;return 1;
    }
    assert(loadedTetGenBundle.currentWorldPoints==loadedTetGenBundle.referenceWorldPoints);

    auto staleTetGenBundle=inputBundle;
    staleTetGenBundle.currentWorldPoints=staleTetGenBundle.referenceWorldPoints;
    staleTetGenBundle.currentWorldPoints[0][0]+=1.0e-9;
    PhysicalSkinRegisteredVolumeImportSidecars staleSidecars;
    PhysicalSkinRegisteredVolumeImportReport staleReport;
    std::string staleError;
    assert(!compilePhysicalSkinRegisteredVolumeImport(staleTetGenBundle,registered,rows,attachments,
        allBodyPoses,157u,0.1,"fixture-detail",32u,16u,staleSidecars,staleReport,staleError));
    assert(staleError.find("same-target current/reference nodes")!=std::string::npos);
    assert(bundle.boundaryMap.outer==std::vector<std::uint32_t>({0,1,2}));
    assert(bundle.boundaryMap.skinInner==std::vector<std::uint32_t>({3,4,5}));
    assert(bundle.boundaryMap.fatDeep==std::vector<std::uint32_t>({6,7,8}));
    assert(sidecars.deepAttachments.size()==2u);
    assert(sidecars.deepAttachments[0].volumeNodeRow==6u&&sidecars.deepAttachments[1].volumeNodeRow==8u);
    assert(bundle.volumeNodeParentResidualWorldM.size()==nodeCount);
    for(std::size_t node=0;node<nodeCount;++node) {
        const auto beta=rows[node].parentBarycentric;
        V3 expected=rows[node].parentCurrentResidualWorldM;
        const auto tet=bundle.parent.tetrahedra[0];
        for(unsigned c=0;c<4;++c)for(unsigned a=0;a<3;++a)
            expected[a]+=beta[c]*(bundle.parent.currentWorldPoints[tet[c]][a]-
                                  bundle.parent.referenceWorldPoints[tet[c]][a]);
        assert(norm(sub(expected,bundle.volumeNodeParentResidualWorldM[node]))<1.0e-14);
        const V3 transformedCurrentResidual=transform(affine,rows[node].parentCurrentResidualWorldM,{0,0,0});
        assert(norm(sub(transformedCurrentResidual,bundle.volumeNodeParentResidualWorldM[node]))>1.0e-5);
    }

    auto missingBoundary=rows;
    missingBoundary[0].role=PhysicalSkinRegisteredVolumeNodeRole::interior;
    missingBoundary[0].surfaceVertexRow=NM_INVALID_INDEX;
    requireReject(inputBundle,registered,missingBoundary,attachments,allBodyPoses,"does not bijectively cover");

    auto duplicateBoundary=rows;
    duplicateBoundary[1].surfaceVertexRow=0u;
    requireReject(inputBundle,registered,duplicateBoundary,attachments,allBodyPoses,"mapped to multiple");

    auto unsupportedParent=rows;
    unsupportedParent[10].parentRegistrationKind.clear();
    requireReject(inputBundle,registered,unsupportedParent,attachments,allBodyPoses,"unsupported current-parent");

    auto missingSource=rows;
    missingSource[10].sourceDomain.clear();
    requireReject(inputBundle,registered,missingSource,attachments,allBodyPoses,"lacks a typed source domain");

    auto wrongSourceKind=rows;
    wrongSourceKind[10].sourceMappingKind="vertex";
    requireReject(inputBundle,registered,wrongSourceKind,attachments,allBodyPoses,"source absence has an invalid");

    auto nonfiniteSourceResidual=rows;
    nonfiniteSourceResidual[10].sourceResidualWorldM[0]=std::numeric_limits<double>::quiet_NaN();
    requireReject(inputBundle,registered,nonfiniteSourceResidual,attachments,allBodyPoses,"barycentric or residual is nonfinite");

    auto fabricatedSourceBary=rows;
    fabricatedSourceBary[10].sourceBarycentric[0]=1.0;
    requireReject(inputBundle,registered,fabricatedSourceBary,attachments,allBodyPoses,"source absence has an invalid");

    auto wrongMask=rows;
    wrongMask[3].incidentMaterialMask=(1u<<15u);
    requireReject(inputBundle,registered,wrongMask,attachments,allBodyPoses,"incident material mask differs");

    auto wrongSemantic=rows;
    wrongSemantic[10].geometricDonorBody=1u;
    requireReject(inputBundle,registered,wrongSemantic,attachments,allBodyPoses,"not present on an incident child cell");

    auto wrongParentSemantic=inputBundle;
    wrongParentSemantic.parent.donorBodyIndices[0]=1u;
    requireReject(wrongParentSemantic,registered,rows,attachments,allBodyPoses,"cell donor/region differs");

    auto wrongTarget=rows;
    wrongTarget[11].currentWorldPoint[0]+=1.0e-4;
    requireReject(inputBundle,registered,wrongTarget,attachments,allBodyPoses,"not the same declared");

    auto wrongBindingPoses=allBodyPoses;
    wrongBindingPoses[0].binding=1u;
    PhysicalSkinRegisteredVolumeImportSidecars wrongBindingSidecars;
    PhysicalSkinRegisteredVolumeImportReport wrongBindingReport;
    std::string wrongBindingError;
    auto wrongBindingBundle=inputBundle;
    assert(!compilePhysicalSkinRegisteredVolumeImport(wrongBindingBundle,registered,rows,attachments,
        wrongBindingPoses,157u,0.1,"fixture-detail",32u,16u,wrongBindingSidecars,wrongBindingReport,
        wrongBindingError));
    assert(wrongBindingError.find("current/rest frame differs")!=std::string::npos);

    auto wrongRestPoses=allBodyPoses;
    wrongRestPoses[0].restOrigin[0]+=1.0e-6;
    PhysicalSkinRegisteredVolumeImportSidecars wrongRestSidecars;
    PhysicalSkinRegisteredVolumeImportReport wrongRestReport;
    std::string wrongRestError;
    auto wrongRestBundle=inputBundle;
    assert(!compilePhysicalSkinRegisteredVolumeImport(wrongRestBundle,registered,rows,attachments,
        wrongRestPoses,157u,0.1,"fixture-detail",32u,16u,wrongRestSidecars,wrongRestReport,
        wrongRestError));
    assert(wrongRestError.find("current/rest frame differs")!=std::string::npos);

    auto fabricatedBindingPoses=allBodyPoses;
    fabricatedBindingPoses[156].binding=22u;
    PhysicalSkinRegisteredVolumeImportSidecars fabricatedBindingSidecars;
    PhysicalSkinRegisteredVolumeImportReport fabricatedBindingReport;
    std::string fabricatedBindingError;
    auto fabricatedBindingBundle=inputBundle;
    assert(!compilePhysicalSkinRegisteredVolumeImport(fabricatedBindingBundle,registered,rows,attachments,
        fabricatedBindingPoses,157u,0.1,"fixture-detail",32u,16u,fabricatedBindingSidecars,fabricatedBindingReport,
        fabricatedBindingError));
    assert(fabricatedBindingError.find("fabricated registered skin binding")!=std::string::npos);

    auto missingBodyPoses=allBodyPoses;
    missingBodyPoses[156].available=false;
    PhysicalSkinRegisteredVolumeImportSidecars missingFrameSidecars;
    PhysicalSkinRegisteredVolumeImportReport missingFrameReport;
    std::string missingFrameError;
    auto missingFrameBundle=inputBundle;
    assert(!compilePhysicalSkinRegisteredVolumeImport(missingFrameBundle,registered,rows,attachments,
        missingBodyPoses,157u,0.1,"fixture-detail",32u,16u,missingFrameSidecars,missingFrameReport,
        missingFrameError));
    assert(missingFrameError.find("full native all-bodies current/rest pose")!=std::string::npos);

    auto noAnchor=std::vector<PhysicalSkinStableDeepAttachmentRequest>{
        {1003u,0u,31u,"fixture-detail"}};
    std::vector<PhysicalSkinStableDeepAttachmentMapRow> mapped;
    auto deep=registered.surfaces[2];
    std::string mapError;
    assert(!mapPhysicalSkinStableDeepAttachments(noAnchor,deep,registered.nodeSource[2],
        bundle.boundaryMap,mapped,mapError));
    assert(mapError.find("maps to 0 exact D rows")!=std::string::npos);

    auto duplicateSource=registered.surfaces[2];
    auto duplicateSourceRows=registered.nodeSource[2];
    duplicateSource.sourceVertexIndices[1]=duplicateSource.sourceVertexIndices[0];
    duplicateSourceRows[1]=duplicateSourceRows[0];
    assert(!mapPhysicalSkinStableDeepAttachments(
        std::span<const PhysicalSkinStableDeepAttachmentRequest>(attachments.data(),1),
        duplicateSource,duplicateSourceRows,bundle.boundaryMap,mapped,mapError));
    assert(mapError.find("maps to 2 exact D rows")!=std::string::npos);

    auto singular=registered;
    singular.framesByDonor[0].currentR={1,0,0,0,1,0,0,0,1};
    PhysicalSkinChildMassMappingBundle singularBundle=inputBundle;
    singularBundle.currentWorldPoints.clear();
    singularBundle.volumeNodeDonorBodyIndices.clear();
    singularBundle.volumeNodeDonorLocalPoints.clear();
    singularBundle.volumeNodeDonorOriginWorld.clear();
    singularBundle.volumeNodeDonorLocalToWorldLinear.clear();
    singularBundle.volumeNodeParentTetrahedra.clear();
    singularBundle.volumeNodeParentBarycentric.clear();
    singularBundle.volumeNodeParentResidualWorldM.clear();
    singularBundle.boundaryMap={};
    singularBundle.parent.currentWorldPoints[1]=singularBundle.parent.currentWorldPoints[0];
    requireReject(singularBundle,singular,rows,attachments,allBodyPoses,"nonpositive, singular, or inverted");

    std::cout<<"PASS registered volume import: same-target residual conversion on affine parent motion; "
             <<"exact O/I/D bijections; full current/rest 157-body pose table with one registered binding and 156 unbound test rows; "
             <<"incident material masks and parent semantic closure; interior unbound-body pose; "
             <<"stable deep source remap; no physical admission\n";
}
