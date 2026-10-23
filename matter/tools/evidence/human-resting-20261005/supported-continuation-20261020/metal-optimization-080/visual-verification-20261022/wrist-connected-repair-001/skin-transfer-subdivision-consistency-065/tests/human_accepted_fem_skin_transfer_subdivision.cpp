#include "NumiHumanAcceptedFEMSkinTransfer.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
using T=numi::human::AcceptedFEMSkinTransfer;
using P=T::Point;
static P avg(P a,P b){return {(a[0]+b[0])*.5,(a[1]+b[1])*.5,(a[2]+b[2])*.5};}
static double error(const std::vector<P>& a,const std::vector<P>& b){double e=0;assert(a.size()==b.size());for(std::size_t i=0;i<a.size();++i)for(unsigned k=0;k<3;++k)e=std::max(e,std::abs(a[i][k]-b[i][k]));return e;}
int main(){
    const auto invalid=std::numeric_limits<std::uint32_t>::max();
    const std::vector<P> detail{{0,0,0},{1,0,0},{0,1,0},{.25,.25,0}};
    const std::vector<unsigned> ids{0,1,2,3},detailDonor{0,1,0,0};
    const std::vector<std::uint8_t> used(4,1);
    const std::vector<P> oldRest{{0,0,.02},{1,0,.02},{0,1,.02}};
    const std::vector<unsigned> oldIDs{0,1,2},oldDonor{0,1,0};
    const std::vector<T::Face> oldFaces{{0,1,2}};
    auto newRest=oldRest;newRest.push_back(avg(oldRest[0],oldRest[1]));
    const std::vector<unsigned> newIDs{0,1,2,invalid},newDonor{0,1,0,0};
    const std::vector<T::Face> newFaces{{0,3,2},{3,1,2}},sourceFaces(4,T::Face{0,1,2});
    const std::vector<P> sourceBary{{1,0,0},{0,1,0},{0,0,1},{.5,.5,0}};
    std::vector<T::BaselineProlongationRow> prolong(4);
    for(unsigned i=0;i<3;++i)prolong[i]={{i,i,i},{1,0,0}};
    prolong[3]={{0,1,0},{.5,.5,0}};
    T old,unfixed,fixed;T::BuildReport report;std::string why;
    assert(old.build(detail,ids,detailDonor,used,oldRest,oldIDs,oldDonor,oldFaces,&report,&why));
    auto build=[&](T& t,const std::vector<T::BaselineProlongationRow>* rows){return t.build(detail,ids,detailDonor,used,newRest,newIDs,newDonor,newFaces,&report,&why,std::numeric_limits<double>::infinity(),nullptr,&sourceFaces,&sourceBary,rows);};
    assert(build(unfixed,nullptr));assert(build(fixed,&prolong));
    std::vector<T::Pose> rest(2),pose(2);
    pose[1].rotation={0,std::sin(.39269908169872414),0,std::cos(.39269908169872414)};
    const std::vector<P> skeletal{{0,0,0},{.866,0,.5},{0,1,0},{.2165,.25,.125}};
    std::vector<P> bo,bu,bf;
    assert(old.makeCurrentCoarseBaseline(detail,skeletal,rest,pose,bo,&why));
    assert(unfixed.makeCurrentCoarseBaseline(detail,skeletal,rest,pose,bu,&why));
    assert(fixed.makeCurrentCoarseBaseline(detail,skeletal,rest,pose,bf,&why));
    assert(bf[3]==avg(bo[0],bo[1]));
    for(unsigned i=0;i<3;++i)assert(bf[i]==bo[i]);
    std::vector<P> accepted{{.02,.01,.05},{.99,.02,.07},{.03,.98,.06}},an=accepted;an.push_back(avg(accepted[0],accepted[1]));
    std::vector<std::array<float,4>> do_,du,df;
    assert(old.makeCoarseDisplacements(accepted,bo,do_,&why));
    assert(unfixed.makeCoarseDisplacements(an,bu,du,&why));
    assert(fixed.makeCoarseDisplacements(an,bf,df,&why));
    std::vector<P> xo,xu,xf;
    assert(old.applyReference(skeletal,do_,xo,&why));assert(unfixed.applyReference(skeletal,du,xu,&why));assert(fixed.applyReference(skeletal,df,xf,&why));
    const double before=error(xo,xu),after=error(xo,xf);
    assert(before>.003);assert(after<2e-8);
    // Identity/zero-FEM behavior is unchanged by the optional metadata.
    assert(fixed.makeCoarseDisplacements(bf,bf,df,&why));assert(fixed.applyReference(skeletal,df,xf,&why));assert(xf==skeletal);
    auto bad=prolong;bad[3].parent[0]=4;assert(!build(fixed,&bad));
    bad=prolong;bad[0]={{1,1,1},{1,0,0}};assert(!build(fixed,&bad));
    bad=prolong;bad[3].weight[0]=.6;assert(!build(fixed,&bad));
    bad=prolong;bad[3]={{3,0,0},{.5,.5,0}};assert(!build(fixed,&bad));
    bad=prolong;bad[3]={{0,2,0},{.5,.5,0}};assert(!build(fixed,&bad));
    bad=prolong;bad.pop_back();assert(!build(fixed,&bad));
    assert(build(fixed,&prolong));
    assert(build(fixed,nullptr)); // Legacy rebuild clears prior metadata.
    assert(fixed.makeCurrentCoarseBaseline(detail,skeletal,rest,pose,bf,&why));assert(bf==bu);
    std::cout<<"PASS subdivision_consistency mixed_donor_parent_edge old_error_m="<<before<<" corrected_error_m="<<after<<" invalid_ancestry_rejected=1 legacy_rebuild_exact=1\n";
}
