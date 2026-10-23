#include "NumiHumanSkinFEMIntegration.wholebody.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace metalrobo::human::coarse_tissue;
namespace {
void check(bool value,const char* message) {
    if(!value) throw std::runtime_error(message);
}
}
int main() {
try {
    // Common rigid placement must work even if the first row is projected.
    const V3 C{1.2,-.4,.7},residual{.003,-.005,.002};
    const std::array<double,9> I{1,0,0,0,1,0,0,0,1};
    const std::array<double,9> currentFrame{0,1,0,-1,0,0,0,0,1};
    RegisteredSkinRigidPlacement expected;
    expected.rotation={0,-1,0,1,0,0,0,0,1};expected.translation={.3,-.2,.9};
    V3 sourceRef{};for(unsigned c=0;c<3;++c)sourceRef[c]=C[c]-residual[c];
    const auto posedSource=expected.point(sourceRef);
    RegisteredSkinRigidPlacement placed;std::string placementError;
    check(deriveRegisteredSkinEmbeddingPlacement(C,I,residual,posedSource,
        currentFrame,placed,placementError),"first projected placement rejected");
    auto corrected=posedSource;const auto rotatedResidual=placed.vector(residual);
    for(unsigned c=0;c<3;++c)corrected[c]+=rotatedResidual[c];
    const auto target=expected.point(C);
    for(unsigned c=0;c<3;++c)check(std::abs(corrected[c]-target[c])<1e-12,
        "projected registration residual was lost or applied twice");
    check(deriveRegisteredSkinEmbeddingPlacement(C,I,V3{},expected.point(C),
        currentFrame,placed,placementError),"direct first row placement rejected");
    auto invalid=residual;invalid[0]=std::numeric_limits<double>::infinity();
    const auto oldTranslation=placed.translation;
    check(!deriveRegisteredSkinEmbeddingPlacement(C,I,invalid,posedSource,
        currentFrame,placed,placementError)&&placed.translation==oldTranslation,
        "nonfinite residual mutated placement");
    const std::string hash(64,'a');
    std::array<std::uint32_t,2> donors{17,18};
    std::array<WholeBodySkinDonorFrameV1,2> frames{};
    std::array<V3,2> points{};
    for(std::size_t i=0;i<2;++i) {
        frames[i].donorBodyIndex=donors[i];
        frames[i].registeredWorldPoint={1.0f,0.0f,0.0f};
        frames[i].registeredWorldNormal={0.0f,0.0f,1.0f};
        frames[i].worldToDonorLocalLinear={1,0,0,0,1,0,0,0,1};
        points[i]={1.0+1e-12,0,0};
    }
    bool candidate=true;std::string error;
    auto valid=[&](std::string_view payload,std::string_view declared) {
        error.clear();
        return validateRegisteredSkinCandidateMetadata(
            payload,declared,2,donors,frames,points,candidate,error);
    };
    check(validateRegisteredSkinCandidateMetadata(
        hash,{},2,{},{},{},candidate,error)&&!candidate,
        "legacy empty metadata must preserve the default owner");
    check(valid(hash,hash)&&candidate,"complete candidate registration rejected");
    check(!valid(std::string(64,'b'),hash)&&!candidate,"wrong payload hash accepted");
    check(!valid(hash,"AA")&&!candidate,"malformed candidate hash accepted");
    check(!valid(hash,{})&&!candidate,"metadata without a declaration accepted");
    check(!validateRegisteredSkinCandidateMetadata(
        hash,hash,2,std::span(donors).first(1),frames,points,candidate,error),
        "partial donors accepted");
    check(!validateRegisteredSkinCandidateMetadata(
        hash,hash,2,donors,std::span(frames).first(1),points,candidate,error),
        "partial frames accepted");
    check(!validateRegisteredSkinCandidateMetadata(
        hash,hash,2,donors,frames,std::span(points).first(1),candidate,error),
        "partial reference points accepted");
    frames[1].donorBodyIndex=17;
    check(!valid(hash,hash),"contradictory donor/frame identities accepted");
    frames[1].donorBodyIndex=18;
    points[1][0]=1.001;
    check(!valid(hash,hash),"reference point inconsistent with Float32 frame accepted");
    points[1][0]=1.0;
    frames[1].worldToDonorLocalLinear[0]=std::numeric_limits<float>::quiet_NaN();
    check(!valid(hash,hash),"nonfinite transform accepted");
    frames[1].worldToDonorLocalLinear[0]=1;
    frames[1].donorLocalPoint[0]=std::numeric_limits<float>::infinity();
    check(!valid(hash,hash),"nonfinite local point accepted");
    frames[1].donorLocalPoint[0]=0;
    check(valid(hash,hash)&&candidate,"valid registration did not recover after rejection");
    std::cout<<"candidate_skin_registration=PASS legacy_default=unchanged "
               "hash_owner_reference_consistency=required nonfinite_partial_metadata=REJECTED\n";
    return 0;
} catch(const std::exception& e) {
    std::cerr<<e.what()<<'\n';return 1;
}
}
