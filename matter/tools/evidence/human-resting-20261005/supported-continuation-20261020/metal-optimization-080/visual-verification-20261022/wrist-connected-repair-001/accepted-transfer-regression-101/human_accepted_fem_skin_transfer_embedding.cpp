#include "NumiHumanAcceptedFEMSkinTransfer.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
using T=numi::human::AcceptedFEMSkinTransfer;
int main() {
    const auto invalid=std::numeric_limits<std::uint32_t>::max();
    std::vector<T::Point> detail{{0,0,0},{1,0,0},{0,1,0}};
    std::vector<std::uint32_t> ids{0,1,2},donors(3,0),synthetic(3,invalid);
    std::vector<std::uint8_t> used(3,1);
    std::vector<T::Face> topology{{0,1,2}},sourceFaces(3,T::Face{0,1,2});
    std::vector<T::Point> beta{{.5,.25,.25},{.25,.5,.25},{.25,.25,.5}};
    std::vector<T::Point> coarse{{.25,.25,.01},{.5,.25,.01},{.25,.5,.01}};
    T transfer;T::BuildReport report;std::string error;
    auto build=[&](const auto* faces,const auto* bary) {
        return transfer.build(detail,ids,donors,used,coarse,synthetic,donors,
            topology,&report,&error,std::numeric_limits<double>::infinity(),nullptr,faces,bary);
    };
    assert(build(&sourceFaces,&beta));
    std::vector<T::Pose> rest(1),current(1);
    std::vector<T::Point> baseline;
    assert(transfer.makeCurrentCoarseBaseline(detail,detail,rest,current,baseline,&error));
    assert(baseline==coarse);
    // Each current triangle corner follows its own accepted articulated pose.
    // Interpolate all three current points before carrying the remesh offset.
    std::vector<T::Point> deformed{{1,2,3},{1,4,3},{-2,2,3}};
    current[0].rotation={std::sqrt(.5),0,0,std::sqrt(.5)}; // +90 deg about X
    assert(transfer.makeCurrentCoarseBaseline(detail,deformed,rest,current,baseline,&error));
    for(std::size_t i=0;i<3;++i) {
        T::Point expected{};
        for(int k=0;k<3;++k)for(int j=0;j<3;++j)expected[j]+=beta[i][k]*deformed[k][j];
        expected[1]-=.01;
        for(int j=0;j<3;++j)assert(std::abs(baseline[i][j]-expected[j])<1e-13);
    }
    std::vector<std::array<float,4>> displacement;
    assert(transfer.makeCoarseDisplacements(baseline,baseline,displacement,&error));
    for(const auto& d:displacement)for(float v:d)assert(v==0);
    auto shortSource=detail;shortSource.pop_back();
    assert(!transfer.makeCurrentCoarseBaseline(shortSource,deformed,rest,current,baseline,&error));
    assert(baseline.empty());
    auto badBeta=beta;badBeta[0][0]=.6;
    assert(!build(&sourceFaces,&badBeta));assert(transfer.rows().empty());
    badBeta=beta;badBeta[0][0]=std::numeric_limits<double>::quiet_NaN();
    assert(!build(&sourceFaces,&badBeta));
    badBeta=beta;badBeta.pop_back();assert(!build(&sourceFaces,&badBeta));
    auto badFaces=sourceFaces;badFaces[0][0]=3;assert(!build(&badFaces,&beta));
    const std::vector<T::Point>* absent=nullptr;
    assert(!build(&sourceFaces,absent));
    // A direct source identity cannot be silently relabelled as a blend.
    assert(!transfer.build(detail,ids,donors,used,coarse,ids,donors,topology,
        &report,&error,std::numeric_limits<double>::infinity(),nullptr,&sourceFaces,&beta));
    // Legacy and explicit one-hot registration produce identical transfer rows
    // and baselines; rebuilding legacy state clears prior embedding metadata.
    T legacy;
    assert(legacy.build(detail,ids,donors,used,detail,ids,donors,topology,&report,&error));
    std::vector<T::Point> oneHot{{1,0,0},{0,1,0},{0,0,1}};
    assert(transfer.build(detail,ids,donors,used,detail,ids,donors,topology,
        &report,&error,std::numeric_limits<double>::infinity(),nullptr,&sourceFaces,&oneHot));
    assert(std::memcmp(legacy.rows().data(),transfer.rows().data(),
        legacy.rows().size()*sizeof(T::GPUTransferRow))==0);
    std::vector<T::Point> a,b;
    assert(legacy.makeCurrentCoarseBaseline(detail,deformed,rest,current,a,&error));
    assert(transfer.makeCurrentCoarseBaseline(detail,deformed,rest,current,b,&error));assert(a==b);
    assert(transfer.build(detail,ids,donors,used,detail,ids,donors,topology,&report,&error));
    assert(transfer.makeCurrentCoarseBaseline(detail,deformed,rest,current,b,&error));assert(a==b);
    std::cout<<"PASS embedded coarse baseline, accepted source interpolation, donor rotation, zero displacement, legacy parity and invalid ancestry rejection\n";
}
