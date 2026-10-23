#include "NumiHumanAcceptedFEMSkinTransfer.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Transfer=numi::human::AcceptedFEMSkinTransfer;
using Point=Transfer::Point;
using Face=Transfer::Face;
static void need(bool v,const char* what){if(!v)throw std::runtime_error(what);}
static bool sameRow(const Transfer::GPUTransferRow&a,const Transfer::GPUTransferRow&b){
    for(unsigned i=0;i<3;++i)
        if(a.coarseVertex[i]!=b.coarseVertex[i]||a.weight[i]!=b.weight[i])return false;
    return a.donorBodyIndex==b.donorBodyIndex&&a.restDistance==b.restDistance;
}
static Transfer build(unsigned rings,bool stable,
    const std::vector<Point>& detail,const std::vector<std::uint32_t>& detailSource,
    const std::vector<std::uint32_t>& detailDonor,const std::vector<std::uint8_t>& used,
    const std::vector<Point>& coarse,const std::vector<std::uint32_t>& coarseSource,
    const std::vector<std::uint32_t>& coarseDonor,const std::vector<Face>& faces,
    const std::vector<std::uint32_t>& representatives,const std::vector<std::uint32_t>& stableIds){
    Transfer t;Transfer::BuildReport r;std::string error;
    const auto* identities=stable?&stableIds:nullptr;
    need(t.build(detail,detailSource,detailDonor,used,coarse,coarseSource,coarseDonor,
        faces,&r,&error,std::numeric_limits<double>::infinity(),&representatives,
        nullptr,nullptr,nullptr,rings,identities),error.c_str());
    return t;
}
int main(){try{
    // Face 0 is the representative's seed. Face 1 and duplicate face 3 are
    // edge-connected; face 2 is a geometrically closer disconnected island.
    // The seed representative deliberately belongs to donor 2, while the
    // detail rows belong to donor 1: legacy seed support must remain available.
    const std::vector<Point> coarse{
        Point{0,0,0},Point{1,0,0},Point{0,1,0},Point{1,1,0},
        Point{.94,.94,0},Point{1.06,.94,0},Point{.94,1.06,0}
    };
    const std::vector<std::uint32_t> coarseSource{0,1,2,3,4,5,6};
    const std::vector<std::uint32_t> coarseDonor{2,1,1,1,1,1,1};
    const std::vector<Face> faces{Face{0,1,2},Face{1,3,2},Face{4,5,6},Face{1,3,2}};
    const std::vector<std::uint32_t> stableIds{100,50,0,40};
    const std::vector<Point> detail{Point{.95,.95,.05},coarse[1]};
    const std::vector<std::uint32_t> detailSource{999,1};
    const std::vector<std::uint32_t> detailDonor{1,1};
    const std::vector<std::uint8_t> used{1,1};
    const std::vector<std::uint32_t> representatives{0,1};

    auto legacyDefault=build(1,false,detail,detailSource,detailDonor,used,
        coarse,coarseSource,coarseDonor,faces,representatives,stableIds);
    auto explicitRing1=build(1,true,detail,detailSource,detailDonor,used,
        coarse,coarseSource,coarseDonor,faces,representatives,stableIds);
    auto ring2=build(2,true,detail,detailSource,detailDonor,used,
        coarse,coarseSource,coarseDonor,faces,representatives,stableIds);
    auto ring3=build(3,true,detail,detailSource,detailDonor,used,
        coarse,coarseSource,coarseDonor,faces,representatives,stableIds);
    need(legacyDefault.rows().size()==2&&explicitRing1.rows().size()==2,"unexpected row count");
    for(size_t i=0;i<2;++i)need(sameRow(legacyDefault.rows()[i],explicitRing1.rows()[i]),"ring-1 changed legacy map");
    const auto& r1=legacyDefault.rows()[0];
    const auto& r2=ring2.rows()[0];
    const auto& r3=ring3.rows()[0];
    need(legacyDefault.selectedFaceIndex(0)==0,"legacy seed face changed");
    need(ring2.selectedFaceIndex(0)==3,"stable-ID tie or same-component selection failed");
    need(ring3.selectedFaceIndex(0)==3,"ring-3 tie selection changed");
    need(ring2.selectedFaceIndex(0)!=2,"search crossed to disconnected closer island");
    need(std::isfinite(r1.restDistance)&&std::isfinite(r2.restDistance)&&std::isfinite(r3.restDistance),"nonfinite projection distance");
    need(r2.restDistance<=r1.restDistance+1.0e-12&&r3.restDistance<=r2.restDistance+1.0e-12,"expanded candidate worsened projection");
    need(r2.restDistance<r1.restDistance*.2,"one shared-edge expansion did not reach the nearby connected face");
    need(sameRow(legacyDefault.rows()[1],ring2.rows()[1])&&sameRow(legacyDefault.rows()[1],ring3.rows()[1]),"exact source row changed");
    for(const auto* t:{&legacyDefault,&ring2,&ring3})for(const auto& row:t->rows()){
        double sum=0;
        for(unsigned k=0;k<3;++k){need(std::isfinite(row.weight[k])&&row.weight[k]>=0,"invalid transfer weight");sum+=row.weight[k];}
        need(std::abs(sum-1.0)<2.0e-7,"transfer weights do not partition unity");
    }
    Transfer invalid;Transfer::BuildReport report;std::string error;
    need(!invalid.build(detail,detailSource,detailDonor,used,coarse,coarseSource,coarseDonor,
        faces,&report,&error,std::numeric_limits<double>::infinity(),&representatives,
        nullptr,nullptr,nullptr,4,&stableIds),"ring bound above 3 was accepted");
    std::cout<<"PASS legacy_default_parity exact_source_stability same_component donor_mismatch_seed stable_tie projection_monotonic partition_unity invalid_bound_reject\n";
    return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}}
