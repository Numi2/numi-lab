#pragma once
// Preparation-only bridge into the existing connected-skin-transfer-013 owner.
// Reverse projections are labeled construction inference, never measured source ancestry.
#include "../connected-skin-mechanical-transfer-013/ConnectedSkinMechanicalTransfer013.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>
namespace metalrobo::human::connected_skin_transfer015 {
using Point=std::array<double,3>; using Face=std::array<std::uint32_t,3>;
using TransferRow=numi::human::connected_skin_transfer013::TransferRow;
using MechanicalInput=numi::human::connected_skin_transfer013::Input;
using MechanicalOutput=numi::human::connected_skin_transfer013::Output;
struct AcceptedSnapshot { bool accepted=false; std::uint64_t sequence=0; std::string snapshotSha256,poseSha256; std::span<const Point> acceptedOuterWorldM,currentMechanicalBaselineWorldM; };
struct ProjectionMap { std::vector<std::uint32_t> ownerFaceToSourceFace; std::vector<Face> ownerInputFaces; double maximumProjectionDistanceM=0,rmsProjectionDistanceM=0; };
struct Context {
 std::span<const Point> sourceRestWorldM; std::span<const Face> detailFaces;
 std::span<const std::uint32_t> detailSourceVertexIds,detailDonorBody; std::span<const Point> detailRestWorldM;
 std::span<const std::uint8_t> detailTransferActive,detailMayMove,detailProtected;
 std::span<const std::uint32_t> physicalDonorBody; std::span<const std::uint8_t> physicalMayMove,physicalProtected;
 std::span<const double> physicalNodeMassKg; std::string physicalMeshSha256,sourcePayloadSha256;
 double maximumInferredProjectionM=0,reconstructionToleranceM=1e-8,displacementToleranceM=1e-12;
};
struct Report { std::size_t explicitSourceEmbeddingRows=0,inferredConstructionEmbeddingRows=0,activeAcceptedRows=0,inactiveAcceptedRows=0; double maximumInferredProjectionM=0,rmsInferredProjectionM=0; std::string sourceEmbeddingProvenance; numi::human::connected_skin_transfer013::Report mechanical; };
namespace detail {
inline bool finite(const Point&p){return std::isfinite(p[0])&&std::isfinite(p[1])&&std::isfinite(p[2]);}
inline Point sub(const Point&a,const Point&b){return{a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
inline Point add(const Point&a,const Point&b){return{a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
inline void addScaled(Point&p,const Point&q,double w){for(unsigned a=0;a<3;++a)p[a]+=w*q[a];}
inline double norm(const Point&p){return std::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);}
inline bool sha(const std::string&s){if(s.size()!=64)return false;return std::all_of(s.begin(),s.end(),[](unsigned char c){return std::isxdigit(c)!=0;});}
inline bool bary(const Point&b){double s=0;for(double w:b){if(!std::isfinite(w)||w< -1e-9||w>1+1e-9)return false;s+=w;}return std::abs(s-1)<=1e-5;}
inline bool contains(const Face&f,std::uint32_t id,unsigned&k){for(k=0;k<3;++k)if(f[k]==id)return true;return false;}
}

// Infer O-row donor identities from the exact interpolated 86-lane source weights,
// using the caller's existing registered donor resolver. This initial owner pass
// finds geometric support across all registered source faces; the final projection
// below is rebuilt with the resulting donor IDs, so the support is then donor-filtered
// by AcceptedFEMSkinTransfer itself.
template<class Owner,class Candidate,class DonorResolver>
inline bool inferPhysicalDonorBodies(Owner&owner,const Candidate&c,std::span<const Point>source,
 std::span<const double>sourceWeights86,std::span<const std::uint32_t>bindingBodyIndices,
 double maxDistanceM,DonorResolver&&resolveDonor,std::vector<std::uint32_t>&physicalDonor,
 ProjectionMap&map,std::string&error){
 auto fail=[&](const std::string&s){error=s;physicalDonor.clear();map={};return false;};
 constexpr std::size_t kBindings=86u;const auto ns=source.size(),np=c.outerWorld.size();
 if(!ns||!np||c.sourceVertexCount!=ns||c.sourceFaces.empty()||
    sourceWeights86.size()!=ns*kBindings||bindingBodyIndices.size()!=kBindings||
    !std::isfinite(maxDistanceM)||maxDistanceM<=0.0)return fail("donor inference inputs or finite projection bound incomplete");
 std::vector<typename Owner::Point>target(np),coarse(ns);std::vector<typename Owner::Face>faces;faces.reserve(c.sourceFaces.size());
 for(std::size_t i=0;i<np;++i){if(!detail::finite(c.outerWorld[i]))return fail("nonfinite O point");target[i]=c.outerWorld[i];}
 for(std::size_t i=0;i<ns;++i){if(!detail::finite(source[i]))return fail("nonfinite source point");coarse[i]=source[i];}
 for(const auto&f:c.sourceFaces){for(auto v:f)if(v>=ns)return fail("source face index outside domain");faces.push_back({f[0],f[1],f[2]});}
 std::vector<std::uint32_t>targetSource(np,std::numeric_limits<std::uint32_t>::max()),
  provisionalTargetDonor(np,0u),sourceIdentity(ns),
  provisionalSourceDonor(ns,0u),stable;
 for(std::size_t i=0;i<ns;++i)sourceIdentity[i]=static_cast<std::uint32_t>(i);
 std::vector<std::uint8_t>used(np,1u);
 if constexpr(requires{c.sourceFaceStableIds;}) {
  if(c.sourceFaceStableIds.size()!=0u){if(c.sourceFaceStableIds.size()!=c.sourceFaces.size())return fail("stable source face ID count differs from topology");for(auto x:c.sourceFaceStableIds){if(x<0||static_cast<std::uint64_t>(x)>std::numeric_limits<std::uint32_t>::max())return fail("stable source face ID is outside owner ABI");stable.push_back(static_cast<std::uint32_t>(x));}}
 }
 typename Owner::BuildReport build{};std::string why;
 if(!owner.build(target,targetSource,provisionalTargetDonor,used,coarse,sourceIdentity,provisionalSourceDonor,
    faces,&build,&why,maxDistanceM,nullptr,nullptr,nullptr,nullptr,1u,stable.empty()?nullptr:&stable))
    return fail("registered source donor-probe projection failed: "+why);
 if(owner.rows().size()!=np||build.projectedRows!=np)return fail("donor-probe owner did not cover every O row");
 physicalDonor.resize(np);std::vector<std::uint32_t>ownerFace(np);
 for(std::size_t i=0;i<np;++i){const auto fi=owner.selectedFaceIndex(i);if(fi>=c.sourceFaces.size())return fail("donor-probe face row outside source topology");
  const auto&f=c.sourceFaces[fi];const auto&r=owner.rows()[i];double sum=0.0;std::array<double,kBindings>w{};
  for(unsigned k=0;k<3;++k){if(r.coarseVertex[k]!=f[k]||!std::isfinite(r.weight[k])||r.weight[k]<-1e-7f)return fail("donor-probe triangle or barycentric row differs from selected source face");sum+=r.weight[k];}
  if(!(sum>0.0)||std::abs(sum-1.0)>1e-4)return fail("donor-probe projection weights do not normalize");
  for(unsigned k=0;k<3;++k){const auto v=f[k];for(std::size_t lane=0;lane<kBindings;++lane){const double q=sourceWeights86[std::size_t(v)*kBindings+lane];if(!std::isfinite(q)||q<0.0)return fail("invalid source 86-lane weight");w[lane]+=(double(r.weight[k])/sum)*q;}}
  double wsum=0;for(double q:w){if(!std::isfinite(q)||q<0.0)return fail("invalid interpolated donor weight");wsum+=q;}if(!std::isfinite(wsum)||std::abs(wsum-1.0)>1e-4)return fail("interpolated donor weights do not normalize");
  std::uint32_t donor=std::numeric_limits<std::uint32_t>::max();if(!resolveDonor(std::span<const double,kBindings>(w),c.outerWorld[i],donor,why))return fail("registered donor resolver failed at O row "+std::to_string(i)+": "+why);
  if(donor==std::numeric_limits<std::uint32_t>::max())return fail("registered donor resolver returned invalid body at O row "+std::to_string(i));
  physicalDonor[i]=donor;ownerFace[i]=fi;
 }
 map.ownerFaceToSourceFace.resize(c.sourceFaces.size());for(std::size_t i=0;i<c.sourceFaces.size();++i)map.ownerFaceToSourceFace[i]=static_cast<std::uint32_t>(i);
 map.ownerInputFaces.assign(c.sourceFaces.begin(),c.sourceFaces.end());map.maximumProjectionDistanceM=build.maximumProjectionDistance;map.rmsProjectionDistanceM=build.rmsProjectionDistance;
 error.clear();return true;
}

// Build the actual reverse support with the candidate's dominant physical donor
// IDs. Mixed-donor source triangles remain eligible exactly as in the existing
// AcceptedFEMSkinTransfer owner; they are contributors, not competing ownership.
template<class Owner,class Candidate>
inline bool buildReverseProjectionOwner(Owner&owner,const Candidate&c,std::span<const Point>source,
 std::span<const std::uint32_t>sourceDonor,std::span<const std::uint32_t>physicalDonor,double maxDistanceM,
 std::span<const std::uint32_t>stableFaceIds,ProjectionMap&map,std::string&error){
 auto fail=[&](const std::string&s){error=s;map={};return false;};const auto ns=source.size(),np=c.outerWorld.size();
 if(!ns||!np||c.sourceVertexCount!=ns||c.sourceFaces.empty()||sourceDonor.size()!=ns||physicalDonor.size()!=np||
    (!stableFaceIds.empty()&&stableFaceIds.size()!=c.sourceFaces.size())||!std::isfinite(maxDistanceM)||maxDistanceM<=0)return fail("reverse projection inputs or finite distance bound incomplete");
 std::vector<typename Owner::Point>target(np),coarse(ns);std::vector<typename Owner::Face>faces;faces.reserve(c.sourceFaces.size());
 for(std::size_t i=0;i<np;++i){if(!detail::finite(c.outerWorld[i]))return fail("nonfinite O point");target[i]=c.outerWorld[i];}
 for(std::size_t i=0;i<ns;++i){if(!detail::finite(source[i]))return fail("nonfinite source point");coarse[i]=source[i];}
 for(const auto&f:c.sourceFaces){for(auto v:f)if(v>=ns)return fail("source face index outside domain");faces.push_back({f[0],f[1],f[2]});}
 std::vector<std::uint32_t>physicalSource(np,std::numeric_limits<std::uint32_t>::max()),sourceIds(ns),coarseDonor(sourceDonor.begin(),sourceDonor.end());
 for(std::size_t i=0;i<ns;++i)sourceIds[i]=static_cast<std::uint32_t>(i);std::vector<std::uint8_t>used(np,1u);
 std::vector<std::uint32_t>pd(physicalDonor.begin(),physicalDonor.end()),sf(stableFaceIds.begin(),stableFaceIds.end());
 typename Owner::BuildReport br{};std::string why;
 if(!owner.build(target,physicalSource,pd,used,coarse,sourceIds,coarseDonor,faces,&br,&why,maxDistanceM,nullptr,nullptr,nullptr,nullptr,1u,stableFaceIds.empty()?nullptr:&sf))return fail("accepted-owner reverse projection failed: "+why);
 if(owner.rows().size()!=np||br.projectedRows!=np)return fail("reverse projection did not cover every O row");
 map.ownerFaceToSourceFace.resize(c.sourceFaces.size());for(std::size_t i=0;i<c.sourceFaces.size();++i)map.ownerFaceToSourceFace[i]=static_cast<std::uint32_t>(i);
 map.ownerInputFaces.assign(c.sourceFaces.begin(),c.sourceFaces.end());map.maximumProjectionDistanceM=br.maximumProjectionDistance;map.rmsProjectionDistanceM=br.rmsProjectionDistance;error.clear();return true;
}

template<class Candidate,class ReverseOwner>
inline bool buildPhysicalSourceEmbedding(const Candidate&c,std::span<const Point>source,const ReverseOwner*reverse,const ProjectionMap*projection,double toleranceM,
 std::vector<std::uint32_t>&faceRows,std::vector<Point>&bary,std::vector<Point>&residual,Report&report,std::string&error){
 auto fail=[&](const std::string&s){error=s;faceRows.clear();bary.clear();residual.clear();return false;};const auto n=c.outerWorld.size();
 if(!n||source.size()!=c.sourceVertexCount||c.sourceFaces.empty()||c.sourceVertexIds.size()!=n||c.sourceFaceIds.size()!=n||c.sourceBarycentric.size()!=n||c.vertexKinds.size()!=n||c.sourceResidualWorldM.size()!=n||!std::isfinite(toleranceM)||toleranceM<=0)return fail("candidate source mapping incomplete");
 if(bool(reverse)!=bool(projection))return fail("reverse owner and projection map must be paired");faceRows.resize(n);bary.resize(n);residual.resize(n);std::size_t inferred=0;double maxD=0,sumD2=0;
 for(std::size_t i=0;i<n;++i){std::uint32_t fr=UINT32_MAX;Point b{};bool explicitRow=false;const auto kind=c.vertexKinds[i];const auto raw=c.sourceVertexIds[i];
  if(kind==0){if(raw<0||std::uint64_t(raw)>=source.size())return fail("retained O row missing source ID row="+std::to_string(i));unsigned corner=0;for(std::size_t f=0;f<c.sourceFaces.size();++f)if(detail::contains(c.sourceFaces[f],std::uint32_t(raw),corner)){fr=std::uint32_t(f);b[corner]=1;explicitRow=true;break;}if(!explicitRow)return fail("retained source vertex not incident on source face row="+std::to_string(i));residual[i]=c.sourceResidualWorldM[i];}
  else if(c.sourceFaceIds[i]<c.sourceFaces.size()){fr=c.sourceFaceIds[i];b=c.sourceBarycentric[i];residual[i]=c.sourceResidualWorldM[i];explicitRow=true;}
  if(!explicitRow){if(!reverse||!projection||i>=reverse->rows().size())return fail("unmapped O row lacks same-donor reverse projection row="+std::to_string(i));const auto ff=reverse->selectedFaceIndex(i);if(ff>=projection->ownerFaceToSourceFace.size())return fail("reverse face row out of range");fr=projection->ownerFaceToSourceFace[ff];if(fr>=c.sourceFaces.size())return fail("reverse source face row out of range");const auto face=c.sourceFaces[fr];const auto&r=reverse->rows()[i];double weightSum=0.0;for(unsigned k=0;k<3;++k){bool found=false;for(unsigned j=0;j<3;++j)if(face[k]==r.coarseVertex[j]){b[k]=double(r.weight[j]);found=true;break;}if(!found)return fail("reverse projection triangle differs from source face");weightSum+=b[k];}if(!(weightSum>0.0)||!std::isfinite(weightSum))return fail("reverse projection barycentric sum invalid");for(double& w:b)w/=weightSum;b[2]=1.0-b[0]-b[1];Point q{};for(unsigned k=0;k<3;++k)detail::addScaled(q,source[face[k]],b[k]);residual[i]=detail::sub(c.outerWorld[i],q);const double d=r.restDistance;if(!std::isfinite(d)||d<0)return fail("reverse projection distance invalid");maxD=std::max(maxD,d);sumD2+=d*d;++inferred;}
  if(fr>=c.sourceFaces.size()||!detail::bary(b)||!detail::finite(residual[i]))return fail("invalid source face/bary/residual row="+std::to_string(i));const auto face=c.sourceFaces[fr];Point q{};for(unsigned k=0;k<3;++k){if(face[k]>=source.size())return fail("source face vertex out of range");detail::addScaled(q,source[face[k]],b[k]);}q=detail::add(q,residual[i]);const double e=detail::norm(detail::sub(q,c.outerWorld[i]));if(!std::isfinite(e)||e>toleranceM)return fail("source residual closure failed row="+std::to_string(i)+" error_m="+std::to_string(e));faceRows[i]=fr;bary[i]=b;
 }
 report.inferredConstructionEmbeddingRows=inferred;report.explicitSourceEmbeddingRows=n-inferred;report.maximumInferredProjectionM=maxD;report.rmsInferredProjectionM=inferred?std::sqrt(sumD2/double(inferred)):0;report.sourceEmbeddingProvenance=inferred?"candidate ancestry where supplied; missing rows use same-donor AcceptedFEMSkinTransfer projection; INFERRED_CONSTRUCTION_EMBEDDING residual, not measured source ancestry":"candidate-supplied face/bary/residual ancestry";error.clear();return true;
}

template<class OwnerRow>
inline bool copyAcceptedOwnerRowsChecked(std::span<const OwnerRow>src,std::span<const std::uint8_t>active,std::span<const std::uint32_t>donor,
 std::vector<TransferRow>&out,std::vector<float>&distance,Report&report,std::string&error){
 auto fail=[&](const std::string&s){error=s;out.clear();distance.clear();return false;};if(src.size()!=active.size()||src.size()!=donor.size())return fail("accepted row/mask/donor lengths differ");out.resize(src.size());distance.resize(src.size());std::size_t na=0;
 for(std::size_t i=0;i<src.size();++i){const auto&r=src[i];if(r.donorBodyIndex!=donor[i])return fail("accepted donor mismatch row="+std::to_string(i));if(!std::isfinite(r.restDistance)||r.restDistance<0)return fail("invalid restDistance row="+std::to_string(i));TransferRow x{};x.active=active[i]!=0;double sum=0;
  for(unsigned k=0;k<3;++k){x.physicalVertex[k]=r.coarseVertex[k];x.weight[k]=r.weight[k];if(!std::isfinite(r.weight[k])||r.weight[k]<-1e-7f)return fail("invalid accepted weight row="+std::to_string(i));sum+=r.weight[k];}
  if(x.active){if(std::abs(sum-1)>1e-4)return fail("active accepted row not normalized row="+std::to_string(i));++na;}else if(r.weight[0]!=0||r.weight[1]!=0||r.weight[2]!=0||r.coarseVertex[0]!=0||r.coarseVertex[1]!=0||r.coarseVertex[2]!=0)return fail("inactive row carries mapping row="+std::to_string(i));out[i]=x;distance[i]=r.restDistance;}
 report.activeAcceptedRows=na;report.inactiveAcceptedRows=src.size()-na;error.clear();return true;
}

template<class Candidate,class ForwardOwner,class ReverseOwner>
inline bool transferAcceptedSnapshot(const Candidate&candidate,const Context&c,const ForwardOwner&owner,const AcceptedSnapshot&state,
 ReverseOwner*reverse,const ProjectionMap*projection,MechanicalOutput&out,Report&report,std::string&error){
 auto fail=[&](const std::string&s){error=s;out={};return false;};if(!state.accepted||state.sequence==0||!detail::sha(state.snapshotSha256)||!detail::sha(state.poseSha256))return fail("identified accepted FEM snapshot and pose digest required");
 const auto np=candidate.outerWorld.size(),nd=c.detailRestWorldM.size();if(state.acceptedOuterWorldM.size()!=np||state.currentMechanicalBaselineWorldM.size()!=np)return fail("accepted/baseline O row count mismatch");
 if(!detail::sha(c.physicalMeshSha256)||!detail::sha(c.sourcePayloadSha256)||!detail::sha(std::string(candidate.sourceMeshSHA256))||std::string(candidate.sourcePayloadSHA256)!=c.sourcePayloadSha256||c.sourceRestWorldM.size()!=candidate.sourceVertexCount)return fail("source mesh/payload and physical pins or source points missing");
 if(reverse&&(!std::isfinite(c.maximumInferredProjectionM)||c.maximumInferredProjectionM<=0.0||!projection||projection->maximumProjectionDistanceM>c.maximumInferredProjectionM))return fail("reverse projection is missing or exceeds the explicit inference distance gate");
 if(!nd||c.detailSourceVertexIds.size()!=nd||c.detailDonorBody.size()!=nd||c.detailTransferActive.size()!=nd||c.detailMayMove.size()!=nd||c.detailProtected.size()!=nd||c.detailFaces.empty()||c.physicalDonorBody.size()!=np||c.physicalMayMove.size()!=np||c.physicalProtected.size()!=np||c.physicalNodeMassKg.size()!=np||owner.rows().size()!=nd)return fail("detail/O rows, masks, map, or mass incomplete");
 std::vector<std::uint32_t>face;std::vector<Point>bary,residual;if(!buildPhysicalSourceEmbedding(candidate,c.sourceRestWorldM,reverse,projection,c.reconstructionToleranceM,face,bary,residual,report,error))return fail(error);
 std::vector<TransferRow>rows;std::vector<float>restDistance;if(!copyAcceptedOwnerRowsChecked(std::span<const typename ForwardOwner::GPUTransferRow>(owner.rows()),c.detailTransferActive,c.detailDonorBody,rows,restDistance,report,error))return fail(error);
 std::vector<typename ForwardOwner::Point>a(state.acceptedOuterWorldM.begin(),state.acceptedOuterWorldM.end()),b(state.currentMechanicalBaselineWorldM.begin(),state.currentMechanicalBaselineWorldM.end());std::vector<std::array<float,4>>df;std::string why;
 if(!owner.makeCoarseDisplacements(a,b,df,&why))return fail("accepted-owner delta extraction failed: "+why);if(df.size()!=np)return fail("accepted-owner delta row count mismatch");std::vector<Point>delta(np);for(std::size_t i=0;i<np;++i){delta[i]={df[i][0],df[i][1],df[i][2]};if(!detail::finite(delta[i]))return fail("nonfinite accepted displacement");}
 MechanicalInput in;in.sourceRestWorldM=c.sourceRestWorldM;in.sourceFaces=candidate.sourceFaces;in.detailSourceVertexIds=c.detailSourceVertexIds;in.detailRestWorldM=c.detailRestWorldM;in.detailFaces=c.detailFaces;in.detailMayMove=c.detailMayMove;in.detailProtected=c.detailProtected;in.physicalOuterRestWorldM=candidate.outerWorld;in.physicalOuterFaces=candidate.faces;in.physicalSourceFaceRows=face;in.physicalSourceBarycentric=bary;in.physicalSourceResidualWorldM=residual;in.physicalMayMove=c.physicalMayMove;in.physicalProtected=c.physicalProtected;in.detailFromPhysical=rows;in.physicalDeltaWorldM=delta;in.physicalNodeMassKg=c.physicalNodeMassKg;in.pins={std::string(candidate.sourceMeshSHA256),c.physicalMeshSha256,state.poseSha256};in.reconstructionToleranceM=c.reconstructionToleranceM;in.displacementToleranceM=c.displacementToleranceM;
 if(!numi::human::connected_skin_transfer013::transfer(in,out,error))return false;report.mechanical=out.report;error.clear();return true;
}
} // namespace metalrobo::human::connected_skin_transfer015
