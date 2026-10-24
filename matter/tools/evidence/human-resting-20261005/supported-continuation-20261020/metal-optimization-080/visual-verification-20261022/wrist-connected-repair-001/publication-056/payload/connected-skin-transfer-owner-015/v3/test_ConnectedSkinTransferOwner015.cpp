#include "ConnectedSkinTransferOwner015.hpp"
#include "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/wholebody-attachment-source-rebind-033-final091-regenerated-deltas-001/source/apps/NumiHumanCoarseTissuePreparation.hpp"
#include "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/wholebody-attachment-source-rebind-033-final091-regenerated-deltas-001/source/apps/NumiHumanAcceptedFEMSkinTransfer.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
using namespace metalrobo::human::connected_skin_transfer015;
using Owner=numi::human::AcceptedFEMSkinTransfer;
using C=metalrobo::human::coarse_tissue::RegisteredSkinFatCandidateView;
struct Fixture {
 std::vector<Point> source{{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
 std::vector<Face> faces{{0,2,1},{0,1,3},{0,3,2},{1,2,3}};
 std::vector<Point> outer=source,detail=source,residual=std::vector<Point>(4),beta=std::vector<Point>(4);
 std::vector<std::int64_t> ids{0,1,2,3};std::vector<std::uint32_t> faceIds=std::vector<std::uint32_t>(4,UINT32_MAX);
 std::vector<std::uint8_t> kinds=std::vector<std::uint8_t>(4,0),active=std::vector<std::uint8_t>(4,1),mayMove=std::vector<std::uint8_t>(4,1),protect=std::vector<std::uint8_t>(4,0);
 std::vector<std::uint32_t> donors=std::vector<std::uint32_t>(4,7),sourceDonor=std::vector<std::uint32_t>(4,7);
 std::vector<double> mass{1,1,1,1};std::vector<float> sourceWeights=std::vector<float>(4*86,0);
 std::string sourceSha=std::string(64,'a'),payloadSha=std::string(64,'b'),physicalSha=std::string(64,'c'),poseSha=std::string(64,'d');
 C view(){C c;c.name="synthetic accepted transfer fixture";c.provider="test";c.sourceMember="fixture";c.sourceMeshSHA256=sourceSha;c.sourcePayloadSHA256=payloadSha;c.stableIdentifier=1;c.sourceVertexCount=4;c.outerWorld=outer;c.faces=faces;c.sourceFaces=faces;c.sourceVertexIds=ids;c.sourceFaceIds=faceIds;c.sourceBarycentric=beta;c.vertexKinds=kinds;c.sourceResidualWorldM=residual;c.sourceWeights86=sourceWeights;return c;}
};
static void testAcceptedSnapshotAndReverseProjection(){
 Fixture f;for(std::size_t i=0;i<4;++i)f.sourceWeights[i*86+7]=1.0f;for(auto&p:f.outer)p[0]+=1.0e-4;std::fill(f.kinds.begin(),f.kinds.end(),1);std::fill(f.ids.begin(),f.ids.end(),-1);
 std::vector<std::uint32_t> ids{0,1,2,3};std::vector<std::uint8_t> used(4,1);Owner forward,reverse;Owner::BuildReport br{};std::string error;
 assert(forward.build(f.detail,ids,f.donors,used,f.outer,ids,f.donors,f.faces,&br,&error,0.01));ProjectionMap map;assert(buildReverseProjectionOwner(reverse,f.view(),f.source,f.sourceDonor,f.donors,0.01,{},map,error));
 Context x;x.sourceRestWorldM=f.source;x.detailFaces=f.faces;x.detailSourceVertexIds=ids;x.detailRestWorldM=f.detail;x.detailDonorBody=f.donors;x.detailTransferActive=f.active;x.detailMayMove=f.mayMove;x.detailProtected=f.protect;x.physicalDonorBody=f.donors;x.physicalMayMove=f.mayMove;x.physicalProtected=f.protect;x.physicalNodeMassKg=f.mass;x.physicalMeshSha256=f.physicalSha;x.maximumInferredProjectionM=0.01;x.sourcePayloadSha256=f.payloadSha;
 std::vector<Point> accepted=f.outer,baseline=f.outer;for(auto&p:accepted)p[1]+=0.002;AcceptedSnapshot state{true,1,std::string(64,'e'),f.poseSha,accepted,baseline};MechanicalOutput out;Report report;
 if(!transferAcceptedSnapshot(f.view(),x,forward,state,&reverse,&map,out,report,error)){std::cerr<<"transfer error: "<<error<<"\n";std::abort();}assert(report.inferredConstructionEmbeddingRows==4&&report.explicitSourceEmbeddingRows==0);assert(report.sourceEmbeddingProvenance.find("INFERRED_CONSTRUCTION_EMBEDDING")!=std::string::npos);assert(out.report.physicalMassRowsBitwiseUnchanged&&out.physicalNodeMassKg==f.mass);
 for(const auto&d:out.detailDeltaWorldM)assert(std::abs(d[1]-0.002)<2e-7);state.accepted=false;assert(!transferAcceptedSnapshot(f.view(),x,forward,state,&reverse,&map,out,report,error));assert(error.find("accepted FEM snapshot")!=std::string::npos);
}
static void testOwnerRowsAndDonorFilter(){
 Fixture f;for(std::size_t i=0;i<4;++i)f.sourceWeights[i*86+7]=1.0f;std::vector<std::uint32_t> ids{0,1,2,3};std::vector<std::uint8_t> used(4,1);Owner owner;Owner::BuildReport br{};std::string error;assert(owner.build(f.detail,ids,f.donors,used,f.outer,ids,f.donors,f.faces,&br,&error,0.1));
 std::vector<TransferRow> rows;std::vector<float> dist;Report report;assert(copyAcceptedOwnerRowsChecked(std::span<const Owner::GPUTransferRow>(owner.rows()),f.active,f.donors,rows,dist,report,error));auto wrong=f.donors;wrong[0]=8;assert(!copyAcceptedOwnerRowsChecked(std::span<const Owner::GPUTransferRow>(owner.rows()),f.active,wrong,rows,dist,report,error));assert(error.find("donor mismatch")!=std::string::npos);
 auto mixed=f.sourceDonor;mixed[2]=9;mixed[3]=9;Owner reverse;ProjectionMap map;assert(buildReverseProjectionOwner(reverse,f.view(),f.source,mixed,f.donors,0.1,{},map,error));assert(reverse.rows().size()==4);for(std::size_t i=0;i<4;++i){assert(reverse.rows()[i].donorBodyIndex==7);const auto face=reverse.selectedFaceIndex(i);assert(face<f.faces.size());bool hasOwner=false,hasContributor=false;for(auto v:f.faces[face]){hasOwner|=mixed[v]==7;hasContributor|=mixed[v]==9;}assert(hasOwner&&hasContributor);}
}

static void testPhysicalDonorProbeUsesUniqueSourceIDs(){
 Fixture f;for(std::size_t i=0;i<4;++i)f.sourceWeights[i*86+7]=1.0f;
 std::vector<double> w(f.sourceWeights.begin(),f.sourceWeights.end());std::vector<std::uint32_t> body(86);for(std::size_t i=0;i<body.size();++i)body[i]=static_cast<std::uint32_t>(i);body[7]=7;
 auto resolver=[&](std::span<const double,86> row,const Point&,std::uint32_t& out,std::string& why){std::size_t best=0;for(std::size_t b=1;b<86;++b)if(row[b]>row[best])best=b;out=body[best];why.clear();return true;};
 Owner owner;std::vector<std::uint32_t> donor;ProjectionMap map;std::string error;
 assert(inferPhysicalDonorBodies(owner,f.view(),f.source,w,body,0.01,resolver,donor,map,error));
 assert(donor==std::vector<std::uint32_t>(4,7));assert(owner.rows().size()==4);assert(map.ownerFaceToSourceFace.size()==f.faces.size());
 for(std::size_t i=0;i<4;++i){assert(owner.selectedFaceIndex(i)<f.faces.size());assert(owner.rows()[i].restDistance<1e-12f);}
}

static void testExplicitAncestryClosure(){
 Fixture f;for(std::size_t i=0;i<4;++i){f.sourceWeights[i*86+7]=1.0f;f.kinds[i]=0;f.ids[i]=std::int64_t(i);}Report report;std::string error;std::vector<std::uint32_t>face;std::vector<Point>bary,res;
 assert(buildPhysicalSourceEmbedding(f.view(),f.source,static_cast<const Owner*>(nullptr),static_cast<const ProjectionMap*>(nullptr),1e-8,face,bary,res,report,error));assert(report.explicitSourceEmbeddingRows==4&&report.inferredConstructionEmbeddingRows==0);f.residual[0][0]=1e-4;assert(!buildPhysicalSourceEmbedding(f.view(),f.source,static_cast<const Owner*>(nullptr),static_cast<const ProjectionMap*>(nullptr),1e-8,face,bary,res,report,error));assert(error.find("residual closure")!=std::string::npos);
}
static void testSourceConnectedRepresentativeFallbackIsLocalAndKeepsDonor(){
 Fixture f;std::vector<Point> detail{{0.1,0.1,0.1}};std::vector<std::uint32_t> detailSource{std::numeric_limits<std::uint32_t>::max()},detailDonor{10},coarseSource(4,std::numeric_limits<std::uint32_t>::max()),coarseDonor(4,4),representative{0};std::vector<std::uint8_t> used{1};Owner owner;Owner::BuildReport report{};std::string error;
 assert(owner.build(detail,detailSource,detailDonor,used,f.outer,coarseSource,coarseDonor,f.faces,&report,&error,1.0,&representative,nullptr,nullptr,nullptr,1u,nullptr));
 assert(report.ancestryFallbackRows==1&&report.projectedRows==1&&owner.rows().size()==1);assert(owner.rows()[0].donorBodyIndex==10);const auto fi=owner.selectedFaceIndex(0);assert(fi<f.faces.size());bool incident=false;for(auto v:f.faces[fi])incident|=v==representative[0];assert(incident);double sum=0;for(float q:owner.rows()[0].weight){assert(std::isfinite(q)&&q>=0);sum+=q;}assert(std::abs(sum-1.0)<1e-4);double parent2=0;for(unsigned a=0;a<3;++a){double d=detail[0][a]-f.outer[0][a];parent2+=d*d;}assert(double(owner.rows()[0].restDistance)<=std::sqrt(parent2)+1e-9);assert(owner.rows()[0].restDistance<=1.0f);
 Owner rejected;auto invalid=representative;invalid[0]=4;assert(!rejected.build(detail,detailSource,detailDonor,used,f.outer,coarseSource,coarseDonor,f.faces,&report,&error,1.0,&invalid,nullptr,nullptr,nullptr,1u,nullptr));assert(error.find("out of range")!=std::string::npos);
 assert(!rejected.build(detail,detailSource,detailDonor,used,f.outer,coarseSource,coarseDonor,f.faces,&report,&error,0.001,&representative,nullptr,nullptr,nullptr,1u,nullptr));assert(error.find("configured distance")!=std::string::npos);
}

int main(){testAcceptedSnapshotAndReverseProjection();testOwnerRowsAndDonorFilter();testPhysicalDonorProbeUsesUniqueSourceIDs();testExplicitAncestryClosure();testSourceConnectedRepresentativeFallbackIsLocalAndKeepsDonor();std::cout<<"PASS 015 accepted snapshot, donor-checked owner rows, same-body inferred embedding, explicit residual closure and existing 013 transfer and source-connected one-ring fallback\n";}
