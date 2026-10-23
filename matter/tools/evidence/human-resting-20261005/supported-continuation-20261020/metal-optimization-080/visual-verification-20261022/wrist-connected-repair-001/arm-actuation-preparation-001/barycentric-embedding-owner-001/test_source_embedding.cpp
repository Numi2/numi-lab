#include "NumiHumanSkinFEMIntegration.wholebody.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <numeric>
#include <limits>
#include <span>
#include <string>
#include <vector>
namespace ct=metalrobo::human::coarse_tissue;
using ct::Face;
using ct::RegisteredSurface;
using ct::V3;
struct TestVertex { unsigned id=0; };
struct TestBinding { std::uint32_t bodyIndex=NM_INVALID_INDEX; };
struct TestSkin {
    std::vector<TestVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<TestBinding> bindings;
    std::vector<float> fullWeights;
};
static int check(bool value,const char* message) {
    if(!value)std::cerr<<"FAIL: "<<message<<"\n";
    return value?0:1;
}
static V3 cross3(V3 a,V3 b) {
    return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}
static double dot3(V3 a,V3 b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
static V3 plus(V3 a,V3 b) { return {a[0]+b[0],a[1]+b[1],a[2]+b[2]}; }
static V3 scale(V3 a,double s) { return {a[0]*s,a[1]*s,a[2]*s}; }
static int testCurrentCandidateFixture(const char* path) {
    std::ifstream input(path);
    if(!input){std::cerr<<"FAIL: cannot open current candidate fixture\n";return 1;}
    std::size_t count=0u,sourceFaceCount=0u;V3 fittedOffset{};double maximumOffsetDeviation=0.0;
    if(!(input>>count>>sourceFaceCount>>fittedOffset[0]>>fittedOffset[1]>>fittedOffset[2]>>maximumOffsetDeviation)||
       count!=167u||sourceFaceCount!=108987u||!std::isfinite(maximumOffsetDeviation)){
        std::cerr<<"FAIL: current candidate fixture header is invalid\n";return 1;
    }
    int failures=0;
    for(std::size_t rowIndex=0;rowIndex<count;++rowIndex){
        std::uint32_t coarseRow=0u,sourceFace=0u;
        std::array<double,3> beta{};V3 expectedPoint{},outer{},residual{};
        if(!(input>>coarseRow>>sourceFace>>beta[0]>>beta[1]>>beta[2]
             >>expectedPoint[0]>>expectedPoint[1]>>expectedPoint[2]
             >>outer[0]>>outer[1]>>outer[2]
             >>residual[0]>>residual[1]>>residual[2])){
            std::cerr<<"FAIL: truncated candidate row "<<rowIndex<<"\n";return 1;
        }
        TestSkin skin;skin.vertices={{0u},{1u},{2u}};skin.indices={0u,1u,2u};
        for(std::uint32_t binding=0;binding<86u;++binding)skin.bindings.push_back({1000u+binding});
        skin.fullWeights.resize(3u*86u);
        std::array<V3,3> points{};
        for(unsigned corner=0;corner<3u;++corner){
            std::uint32_t sourceVertex=0u;
            if(!(input>>sourceVertex>>points[corner][0]>>points[corner][1]>>points[corner][2])){
                std::cerr<<"FAIL: truncated source triangle row "<<rowIndex<<"\n";return 1;
            }
            (void)sourceVertex;
            for(std::size_t binding=0;binding<86u;++binding)
                if(!(input>>skin.fullWeights[corner*86u+binding])){
                    std::cerr<<"FAIL: truncated source weights row "<<rowIndex<<"\n";return 1;
                }
        }
        V3 point{};std::array<double,86> interpolated{};
        std::uint32_t donor=NM_INVALID_INDEX;double confidence=-1.0;std::string error;
        const auto positionAt=[&](std::uint32_t index,const TestVertex&){return points[index];};
        const bool evaluated=ct::evaluateRegisteredSkinSourceEmbedding(skin,0u,beta,positionAt,point,
            std::span<double>(interpolated),donor,confidence,error);
        if(!evaluated){
            std::cerr<<"FAIL: candidate coarse row "<<coarseRow<<" evaluation rejected: "<<error<<"\n";
            ++failures;continue;
        }
        if(std::abs(point[0]-expectedPoint[0])>1e-12||
           std::abs(point[1]-expectedPoint[1])>1e-12||
           std::abs(point[2]-expectedPoint[2])>1e-12){
            std::cerr<<"FAIL: candidate coarse row "<<coarseRow<<" source point mismatch\n";++failures;
        }
        double expectedWeights[86]{};std::size_t dominant=0u;double sum=0.0;
        for(std::size_t binding=0;binding<86u;++binding){
            for(unsigned corner=0;corner<3u;++corner)
                expectedWeights[binding]+=beta[corner]*double(skin.fullWeights[corner*86u+binding]);
            if(std::abs(interpolated[binding]-expectedWeights[binding])>1e-12){
                std::cerr<<"FAIL: candidate coarse row "<<coarseRow<<" body weight mismatch at "<<binding<<"\n";
                ++failures;break;
            }
            sum+=interpolated[binding];
            if(expectedWeights[binding]>expectedWeights[dominant])dominant=binding;
        }
        if(std::abs(sum-1.0)>1e-4||donor!=1000u+dominant||
           std::abs(confidence-expectedWeights[dominant])>1e-12){
            std::cerr<<"FAIL: candidate coarse row "<<coarseRow<<" weight partition/donor mismatch\n";++failures;
        }
        RegisteredSurface ancestry;ancestry.worldPoints={outer};
        ancestry.sourceEmbeddingFaceIndices={sourceFace};
        ancestry.sourceEmbeddingBarycentric={beta};
        ancestry.sourceEmbeddingResidualWorldM={residual};
        if(!ct::validateRegisteredSourceEmbedding(ancestry,sourceFaceCount,error)){
            std::cerr<<"FAIL: candidate coarse row "<<coarseRow<<" ancestry rejected: "<<error<<"\n";++failures;
        }
        (void)fittedOffset;(void)maximumOffsetDeviation;
    }
    if(failures==0)std::cout<<"current_075_missing_palm_rows=PASS count="<<count
        <<" source_faces="<<sourceFaceCount<<" source_geometry_only_candidate_not_admitted\n";
    return failures?1:0;
}

int main(int argc,char** argv) {
    int failures=0;
    TestSkin skin;
    skin.vertices={{0u},{1u},{2u}};
    skin.indices={0u,1u,2u};
    for(std::uint32_t i=0;i<86u;++i)skin.bindings.push_back({100u+i});
    skin.fullWeights.assign(3u*86u,0.0f);
    skin.fullWeights[0u*86u+0u]=1.0f;
    skin.fullWeights[1u*86u+1u]=1.0f;
    skin.fullWeights[2u*86u+0u]=0.25f;
    skin.fullWeights[2u*86u+1u]=0.75f;
    const std::array<V3,3> rest{{{0,0,0},{1,0,0},{0,1,0}}};
    const V3 translation{2,-1,0.5};
    const auto affine=[&](V3 p){return V3{2*p[0]+p[1]+translation[0],-p[0]+3*p[1]+translation[1],0.5*p[0]-2*p[1]+translation[2]};};
    const std::array<V3,3> moved{{affine(rest[0]),affine(rest[1]),affine(rest[2])}};
    const auto positionAt=[&](std::uint32_t i,const TestVertex&){return moved[i];};
    const std::array<double,3> beta{0.2,0.3,0.5};
    V3 point{};std::array<double,86> weights{};
    std::uint32_t donor=NM_INVALID_INDEX;double donorWeight=-1.0;std::string error;
    failures+=check(ct::evaluateRegisteredSkinSourceEmbedding(skin,0u,beta,positionAt,point,
        std::span<double>(weights),donor,donorWeight,error),"valid affine source embedding");
    const V3 basePoint{0.3,0.5,0};
    const V3 expected=affine(basePoint);
    failures+=check(std::abs(point[0]-expected[0])<1e-12&&std::abs(point[1]-expected[1])<1e-12&&
        std::abs(point[2]-expected[2])<1e-12,"affine point reproduction");
    failures+=check(std::abs(weights[0]-0.325)<1e-7&&std::abs(weights[1]-0.675)<1e-7&&
        std::abs(std::accumulate(weights.begin(),weights.end(),0.0)-1.0)<1e-12,
        "full 86-weight interpolation and partition of unity");
    failures+=check(donor==101u&&std::abs(donorWeight-0.675)<1e-7,"dominant interpolated donor and confidence");
    V3 sampled{};error.clear();
    failures+=check(ct::interpolateSourceTrianglePoint(std::array<Face,1>{Face{0u,1u,2u}},rest,0u,beta,
        sampled,error)&&std::abs(sampled[0]-0.3)<1e-12&&std::abs(sampled[1]-0.5)<1e-12,
        "reference point evaluation");
    const V3 residual{1e-4,-2e-4,3e-4};const V3 retained=plus(sampled,residual);
    failures+=check(std::abs(retained[0]-0.3001)<1e-12&&std::abs(retained[1]-0.4998)<1e-12&&
        std::abs(retained[2]-3e-4)<1e-12,"reference-frame residual remains explicit");

    // C^T is checked here as registration-map virtual-work algebra only; this
    // is not a runtime FEM/contact force path or a claim of physical coupling.
    const V3 force{2,-3,4};
    const std::array<V3,3> velocity{{{0.1,0.3,-0.2},{-0.4,0.2,0.5},{0.7,-0.1,0.6}}};
    V3 coarseVelocity{};std::array<V3,3> sourceForces{};
    for(unsigned i=0;i<3u;++i){coarseVelocity=plus(coarseVelocity,scale(velocity[i],beta[i]));sourceForces[i]=scale(force,beta[i]);}
    V3 forceSum{},torqueSum{};double sourceWork=0.0;
    for(unsigned i=0;i<3u;++i){
        forceSum=plus(forceSum,sourceForces[i]);
        torqueSum=plus(torqueSum,cross3(rest[i],sourceForces[i]));
        sourceWork+=dot3(sourceForces[i],velocity[i]);
    }
    const V3 coarseTorque=cross3(sampled,force);
    failures+=check(std::abs(forceSum[0]-force[0])<1e-12&&std::abs(forceSum[1]-force[1])<1e-12&&
        std::abs(forceSum[2]-force[2])<1e-12,"registration-map virtual-work total force identity");
    failures+=check(std::abs(torqueSum[0]-coarseTorque[0])<1e-12&&
        std::abs(torqueSum[1]-coarseTorque[1])<1e-12&&std::abs(torqueSum[2]-coarseTorque[2])<1e-12,
        "registration-map virtual-work moment identity");
    failures+=check(std::abs(sourceWork-dot3(force,coarseVelocity))<1e-12,
        "registration-map virtual-work power identity");

    error.clear();failures+=check(!ct::evaluateRegisteredSkinSourceEmbedding(skin,1u,beta,positionAt,point,
        std::span<double>(weights),donor,donorWeight,error),"out-of-range source face rejected");
    error.clear();const std::array<double,3> badSum{0.2,0.3,0.4};
    failures+=check(!ct::evaluateRegisteredSkinSourceEmbedding(skin,0u,badSum,positionAt,point,
        std::span<double>(weights),donor,donorWeight,error),"non-unit barycentric input rejected");
    error.clear();const std::array<double,3> badNegative{-0.1,0.5,0.6};
    failures+=check(!ct::evaluateRegisteredSkinSourceEmbedding(skin,0u,badNegative,positionAt,point,
        std::span<double>(weights),donor,donorWeight,error),"negative barycentric input rejected");
    error.clear();std::array<double,85> shortWeights{};
    failures+=check(!ct::evaluateRegisteredSkinSourceEmbedding(skin,0u,beta,positionAt,point,
        std::span<double>(shortWeights),donor,donorWeight,error),"insufficient output capacity rejected");
    TestSkin badWeights=skin;badWeights.fullWeights[2u*86u+0u]=0.5f;
    error.clear();failures+=check(!ct::evaluateRegisteredSkinSourceEmbedding(badWeights,0u,beta,positionAt,point,
        std::span<double>(weights),donor,donorWeight,error),"non-normalized source weights rejected");
    TestSkin negativeWeights=skin;negativeWeights.fullWeights[2u*86u+0u]=-0.01f;
    error.clear();failures+=check(!ct::evaluateRegisteredSkinSourceEmbedding(negativeWeights,0u,beta,positionAt,point,
        std::span<double>(weights),donor,donorWeight,error),"negative source weights rejected");

    RegisteredSurface surface;surface.name="embedding test tetrahedron";surface.provider="synthetic";
    surface.sourceMember="test";surface.sourceMeshSHA256=std::string(64u,'a');
    surface.sourcePayloadSHA256=std::string(64u,'b');surface.stableIdentifier=7u;surface.sourceBodyIndex=0u;
    surface.worldPoints={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    surface.donorLocalPoints=surface.worldPoints;surface.faces={{0,2,1},{0,1,3},{0,3,2},{1,2,3}};
    surface.sourceVertexIndices={NM_INVALID_INDEX,NM_INVALID_INDEX,NM_INVALID_INDEX,NM_INVALID_INDEX};
    surface.sourceEmbeddingFaceIndices={0u,0u,0u,0u};
    surface.sourceEmbeddingBarycentric={{{1,0,0}},{{0,1,0}},{{0,0,1}},{{0.2,0.3,0.5}}};
    surface.sourceEmbeddingResidualWorldM={{0,0,0},{0,0,0},{0,0,0},{1e-4,0,0}};
    error.clear();failures+=check(ct::validateRegisteredSourceEmbedding(surface,1u,error),
        "valid registered nodal ancestry with source residual");
    RegisteredSurface patch;error.clear();
    failures+=check(ct::makeConnectedSurfacePatch(surface,0u,4u,patch,error),
        "subset construction with embedded ancestry");
    bool ancestryCopied=patch.worldPoints.size()==surface.worldPoints.size();
    for(std::size_t i=0;i<patch.worldPoints.size()&&ancestryCopied;++i){
        const auto found=std::find(surface.worldPoints.begin(),surface.worldPoints.end(),patch.worldPoints[i]);
        if(found==surface.worldPoints.end()){ancestryCopied=false;break;}
        const auto parent=std::size_t(found-surface.worldPoints.begin());
        ancestryCopied=patch.sourceEmbeddingFaceIndices[i]==surface.sourceEmbeddingFaceIndices[parent]&&
            patch.sourceEmbeddingBarycentric[i]==surface.sourceEmbeddingBarycentric[parent]&&
            patch.sourceEmbeddingResidualWorldM[i]==surface.sourceEmbeddingResidualWorldM[parent];
    }
    failures+=check(ancestryCopied,"source face, barycentric point, and residual survive subset selection");
    RegisteredSurface legacy=surface;legacy.sourceEmbeddingFaceIndices.clear();
    legacy.sourceEmbeddingBarycentric.clear();legacy.sourceEmbeddingResidualWorldM.clear();
    legacy.sourceVertexIndices={0u,1u,2u,3u};error.clear();
    failures+=check(ct::validateRegisteredSourceEmbedding(legacy,1u,error),
        "empty optional map preserves legacy exact source-vertex path");
    RegisteredSurface legacyPatch;error.clear();
    failures+=check(ct::makeConnectedSurfacePatch(legacy,0u,4u,legacyPatch,error),
        "legacy direct-index surface still forms a bounded subset");
    bool legacyParentsCopied=legacyPatch.worldPoints.size()==legacy.worldPoints.size();
    for(std::size_t i=0;i<legacyPatch.worldPoints.size()&&legacyParentsCopied;++i){
        const auto found=std::find(legacy.worldPoints.begin(),legacy.worldPoints.end(),legacyPatch.worldPoints[i]);
        if(found==legacy.worldPoints.end()){legacyParentsCopied=false;break;}
        const auto parent=std::size_t(found-legacy.worldPoints.begin());
        legacyParentsCopied=legacyPatch.sourceVertexIndices[i]==legacy.sourceVertexIndices[parent];
    }
    failures+=check(legacyParentsCopied&&legacyPatch.sourceEmbeddingFaceIndices.empty()&&
        legacyPatch.sourceEmbeddingBarycentric.empty()&&legacyPatch.sourceEmbeddingResidualWorldM.empty(),
        "legacy direct-index subset path remains unchanged and carries no synthetic bary map");
    RegisteredSurface partial=surface;partial.sourceEmbeddingBarycentric.clear();error.clear();
    failures+=check(!ct::validateRegisteredSourceEmbedding(partial,1u,error),
        "partial optional ancestry rejected");
    RegisteredSurface outOfRange=surface;outOfRange.sourceEmbeddingFaceIndices[3]=1u;error.clear();
    failures+=check(!ct::validateRegisteredSourceEmbedding(outOfRange,1u,error),
        "out-of-range source face rejected");
    RegisteredSurface badResidual=surface;badResidual.sourceEmbeddingResidualWorldM[0][0]=std::numeric_limits<double>::quiet_NaN();error.clear();
    failures+=check(!ct::validateRegisteredSourceEmbedding(badResidual,1u,error),
        "nonfinite residual rejected");
    RegisteredSurface shortMap=surface;shortMap.sourceEmbeddingBarycentric.pop_back();error.clear();
    failures+=check(!ct::validateRegisteredSourceEmbedding(shortMap,1u,error),
        "embedding capacity mismatch rejected");
    legacy.sourceEmbeddingResidualWorldM={{0,0,0}};error.clear();
    failures+=check(!ct::validateRegisteredSourceEmbedding(legacy,1u,error),
        "residual without barycentric ancestry rejected");
    failures+=check(argc==2,"current candidate fixture argument supplied");
    if(argc==2)failures+=testCurrentCandidateFixture(argv[1]);
    if(failures){std::cerr<<"source_embedding_cpu_tests=FAIL failures="<<failures<<"\n";return 1;}
    std::cout<<"source_embedding_cpu_tests=PASS affine_point full86_partition unity donor_selection barycentric_subset_residual legacy_direct_index rejection_gates registration_virtual_work_only\n";
    return 0;
}
