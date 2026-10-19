#include "NumiHumanAcceptedFEMSkinTransfer.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
using T=numi::human::AcceptedFEMSkinTransfer;

static bool oneHot(const T::GPUTransferRow& row) {
    int ones=0;
    for(float w:row.weight) {
        if(std::abs(w-1.0f)<1e-6f) ++ones;
        else if(std::abs(w)>1e-6f) return false;
    }
    return ones==1;
}
int main() {
    // One triangle whose vertices have distinct dominant donors. It must be
    // eligible in each represented donor's tree, without creating a tree for
    // any absent donor.
    std::vector<T::Point> coarse{{0,0,0},{1,0,0},{0,1,0}};
    std::vector<std::uint32_t> coarseSource{10,11,12}, coarseDonor{1,2,3};
    std::vector<T::Face> faces{{0,1,2}};
    std::vector<T::Point> detail{{1,0,0},{.2,.2,.1}};
    std::vector<std::uint32_t> detailSource{11,99}, detailDonor{2,9};
    std::vector<std::uint8_t> used{1,1};
    std::vector<std::uint32_t> representative{std::numeric_limits<std::uint32_t>::max(),0};
    T transfer;
    T::BuildReport report;
    std::string error;
    assert(!transfer.build(detail,detailSource,detailDonor,used,coarse,coarseSource,
                           coarseDonor,faces,&report,&error));
    assert(error.find("source ancestry map")!=std::string::npos);
    assert(transfer.build(detail,detailSource,detailDonor,used,coarse,coarseSource,
                          coarseDonor,faces,&report,&error,
                          std::numeric_limits<double>::infinity(),&representative));
    assert(report.exactAncestryRows==1 && report.projectedRows==1);
    assert(report.ancestryFallbackRows==1 && report.donorTreeCount==3);
    assert(oneHot(transfer.rows()[0]));
    const auto& exact=transfer.rows()[0];
    double exactX=0,exactY=0,exactZ=0,sum=0;
    for(int k=0;k<3;++k) {
        const auto v=exact.coarseVertex[k]; const double w=exact.weight[k]; sum+=w;
        exactX+=w*coarse[v][0]; exactY+=w*coarse[v][1]; exactZ+=w*coarse[v][2];
    }
    assert(std::abs(sum-1)<1e-6 && std::abs(exactX-1)<1e-6 &&
           std::abs(exactY)<1e-6 && std::abs(exactZ)<1e-6);
    const auto& projected=transfer.rows()[1];
    assert(projected.donorBodyIndex==9);
    assert(std::abs(projected.weight[0]+projected.weight[1]+projected.weight[2]-1)<1e-6f);
    for(auto v:projected.coarseVertex) assert(v<3); // only the representative's one-ring
    assert(std::abs(projected.restDistance-.1f)<1e-6f);
    assert(report.maximumAncestryFallbackDistance<=std::sqrt(.2*.2+.2*.2+.1*.1)+1e-9);
    // No active row may silently use an absent ancestry assignment.
    representative[1]=std::numeric_limits<std::uint32_t>::max();
    assert(!transfer.build(detail,detailSource,detailDonor,used,coarse,coarseSource,
                           coarseDonor,faces,&report,&error,
                           std::numeric_limits<double>::infinity(),&representative));
    assert(error.find("no source-connected")!=std::string::npos);
    // The same mixed triangle is a valid donor-BVH projection for a donor that
    // is actually represented at a corner.
    std::vector<T::Point> detailOne{{.25,.25,.1}};
    std::vector<std::uint32_t> srcOne{99}, donorOne{3};
    std::vector<std::uint8_t> usedOne{1};
    assert(transfer.build(detailOne,srcOne,donorOne,usedOne,coarse,coarseSource,
                          coarseDonor,faces,&report,&error));
    assert(report.projectedRows==1 && report.ancestryFallbackRows==0);

    // With an ancestry map, even a donor that exists elsewhere must stay in
    // the assigned representative's incident patch rather than jumping to a
    // closer but disconnected face carrying the same donor label.
    std::vector<T::Point> splitCoarse{
        {0,0,0},{1,0,0},{0,1,0},
        {-.031,-.030,.001},{-.029,-.030,.001},{-.030,-.029,.001}};
    std::vector<std::uint32_t> splitSource{20,21,22,23,24,25};
    std::vector<std::uint32_t> splitDonor{1,2,3,1,1,1};
    std::vector<T::Face> splitFaces{{0,1,2},{3,4,5}};
    std::vector<T::Point> splitDetail{{-.030,-.030,.010}};
    std::vector<std::uint32_t> splitDetailSource{99}, splitDetailDonor{1};
    std::vector<std::uint8_t> splitUsed{1};
    std::vector<std::uint32_t> splitRepresentative{0};
    assert(transfer.build(splitDetail,splitDetailSource,splitDetailDonor,splitUsed,
                          splitCoarse,splitSource,splitDonor,splitFaces,&report,&error));
    assert(report.ancestryFallbackRows==0);
    for(auto v:transfer.rows()[0].coarseVertex) assert(v>=3); // legacy donor BVH may select the nearby disconnected patch
    assert(transfer.build(splitDetail,splitDetailSource,splitDetailDonor,splitUsed,
                          splitCoarse,splitSource,splitDonor,splitFaces,&report,&error,
                          std::numeric_limits<double>::infinity(),&splitRepresentative));
    assert(report.ancestryFallbackRows==1);
    for(auto v:transfer.rows()[0].coarseVertex) assert(v<3); // ancestry confines it to the assigned one-ring
    std::cout<<"PASS mixed-donor face membership, exact one-hot ancestry, source-connected one-ring mapping, fail-closed missing assignment\n";
}
