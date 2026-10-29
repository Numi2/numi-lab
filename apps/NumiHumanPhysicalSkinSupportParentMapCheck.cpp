#include "NumiHumanPhysicalSkinSupportParentMap.hpp"
#include "NumiHumanPhysicalSkinChildRegisteredVolumeCSV.hpp"
#include <iostream>
#include <limits>
using namespace metalrobo::human::coarse_tissue;
int main(int argc,char** argv) {
 if(argc==6) {
  std::size_t nodes=0;double bound=0;
  if(!physical_skin_child_mass_detail::parseNumber(std::string(argv[4]),nodes)||
     !physical_skin_child_mass_detail::parseNumber(std::string(argv[5]),bound))return 40;
  PhysicalSkinChildMassMappingBundle bundle;std::string error;
  if(!loadCompiledSkinFatParentMassPacketCSV(argv[1],bundle.parent,error)) {
   std::cerr<<error;return 41;
  }
  std::ifstream input(argv[2]);
  std::vector<PhysicalSkinRegisteredVolumeNodeRow> rows;PhysicalSkinCombinedNodeMapCSVReport report;
  if(!input||!readPhysicalSkinCombinedNodeMapCSV(input,nodes,157u,rows,report,error)) {
   std::cerr<<error;return 42;
  }
  for(const auto& row:rows) {
   if(row.parentTet>=bundle.parent.tetrahedra.size())return 43;
   bundle.currentWorldPoints.push_back(row.currentWorldPoint);
   bundle.referenceWorldPoints.push_back(row.currentWorldPoint);
   bundle.volumeNodeParentTetrahedra.push_back(row.parentTet);
   bundle.volumeNodeParentBarycentric.push_back(row.parentBarycentric);
   V3 residual=row.currentWorldPoint;
   for(unsigned k=0;k<4;++k)for(unsigned axis=0;axis<3;++axis)
    residual[axis]-=row.parentBarycentric[k]*bundle.parent.referenceWorldPoints[
       bundle.parent.tetrahedra[row.parentTet][k]][axis];
   bundle.volumeNodeParentResidualWorldM.push_back(residual);
  }
  std::ifstream elements(argv[3]);std::size_t count=0;unsigned corners=0,attributes=0;
  if(!(elements>>count>>corners>>attributes)||corners!=4||attributes!=1)return 44;
  for(std::size_t i=0;i<count;++i) {
   std::size_t tag=0;std::array<std::uint32_t,4> tet{};
   if(!(elements>>tag>>tet[0]>>tet[1]>>tet[2]>>tet[3])||tag!=i+1)return 45;
   unsigned material=0;if(!(elements>>material)||(material!=15&&material!=16))return 46;
   for(auto& node:tet){if(node==0)return 46;--node;}
   bundle.childTetrahedra.push_back(tet);
  }
  std::vector<std::array<float,3>> actual;
  for(const auto& p:bundle.parent.currentWorldPoints)
   actual.push_back({float(p[0]),float(p[1]),float(p[2])});
  PhysicalSkinSupportParentMap out;
  if(!provePhysicalSkinRetainedSupportNodes(bundle,actual,bundle.parent.tetrahedra,bound,out,error)) {
   std::cerr<<error;return 47;
  }
  const auto inherited=std::count_if(out.childNodeToParent.begin(),out.childNodeToParent.end(),
      [](auto node){return node!=NM_INVALID_INDEX;});
  std::cout<<"actual_retained_support_nodes=PASS retained="<<inherited
      <<" added_unassigned="<<(nodes-inherited)
      <<" maximum_registration_distance_m="<<out.maximumNodeDisplacementM
      <<" topology_proof=NOT_CLAIMED physical_admission=false\n";return 0;
 }
 if(argc!=1)return 48;
 PhysicalSkinChildMassMappingBundle base;
 base.parent.currentWorldPoints={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
 base.parent.referenceWorldPoints=base.parent.currentWorldPoints;
 base.parent.tetrahedra={{0,1,2,3}};
 base.childTetrahedra=base.parent.tetrahedra;
 base.parent.materialIndices=base.childMaterialIndices={15};
 base.parent.donorBodyIndices=base.childDonorBodyIndices={7};
 base.parent.anatomicalRegionIndices=base.childAnatomicalRegionIndices={9};
 base.parent.stableSourceFaceIdentities=base.childStableSourceFaceIdentities={123};
 base.volumeNodeParentTetrahedra={0,0,0,0};
 base.volumeNodeParentBarycentric={{{1,0,0,0}},{{0,1,0,0}},{{0,0,1,0}},{{0,0,0,1}}};
 base.volumeNodeParentResidualWorldM.assign(4,V3{.0001,0,0});
 base.currentWorldPoints=base.referenceWorldPoints=base.parent.currentWorldPoints;
 for(auto& x:base.currentWorldPoints)x[0]+=.0001;
 base.referenceWorldPoints=base.currentWorldPoints;
 base.spatialMassTransportResult.childParentOffsets={0,1};
 base.spatialMassTransportResult.childParentRows={0};
 std::vector<std::array<float,3>> actual{{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
 std::string error;PhysicalSkinSupportParentMap out;
 if(!provePhysicalSkinSupportParentMap(base,actual,base.parent.tetrahedra,.0004,out,error)){
  std::cerr<<error;return 1;
 }
 const auto saved=out.childNodeToParent;const auto savedD=out.maximumNodeDisplacementM;
 unsigned rejected=0;
 auto reject=[&](const auto& bad,double bound=.0004){
  if(provePhysicalSkinSupportParentMap(bad,actual,base.parent.tetrahedra,bound,out,error)||
     out.childNodeToParent!=saved||out.maximumNodeDisplacementM!=savedD)return false;
  ++rejected;return true;
 };
 auto bad=base;bad.volumeNodeParentBarycentric[0]={.5,.5,0,0};if(!reject(bad))return 2;
 bad=base;bad.volumeNodeParentBarycentric[0]={0,1,0,0};if(!reject(bad))return 3;
 bad=base;bad.volumeNodeParentTetrahedra[0]=1;if(!reject(bad))return 4;
 bad=base;bad.volumeNodeParentResidualWorldM[0][0]=0;if(!reject(bad))return 5;
 bad=base;bad.childMaterialIndices[0]=16;if(!reject(bad))return 6;
 bad=base;bad.childDonorBodyIndices[0]=8;if(!reject(bad))return 7;
 bad=base;bad.childAnatomicalRegionIndices[0]=8;if(!reject(bad))return 8;
 bad=base;bad.childStableSourceFaceIdentities[0]=8;if(!reject(bad))return 9;
 bad=base;std::swap(bad.childTetrahedra[0][0],bad.childTetrahedra[0][1]);if(!reject(bad))return 10;
 bad=base;bad.childTetrahedra[0][0]=4;if(!reject(bad))return 11;
 bad=base;bad.spatialMassTransportResult.childParentRows[0]=1;if(!reject(bad))return 12;
 bad=base;bad.spatialMassTransportResult.childParentOffsets[1]=2;if(!reject(bad))return 13;
 if(!reject(base,.00001))return 14;
 bad=base;bad.parent.currentWorldPoints[0][0]=.0000001;if(!reject(bad))return 15;
 bad=base;bad.referenceWorldPoints[0][0]=std::numeric_limits<double>::quiet_NaN();if(!reject(bad))return 16;
 bad=base;bad.parent.referenceWorldPoints[0][0]=std::numeric_limits<double>::quiet_NaN();if(!reject(bad))return 17;
 bad=base;bad.childStableSourceFaceIdentities.clear();if(!reject(bad))return 18;
 auto reordered=base;
 for(auto* points:{&reordered.currentWorldPoints,&reordered.referenceWorldPoints,&reordered.volumeNodeParentResidualWorldM})
  std::swap((*points)[0],(*points)[1]);
 std::swap(reordered.volumeNodeParentBarycentric[0],reordered.volumeNodeParentBarycentric[1]);
 reordered.childTetrahedra[0]={1,0,2,3};
 PhysicalSkinSupportParentMap perm;
 if(!provePhysicalSkinSupportParentMap(reordered,actual,base.parent.tetrahedra,.0004,perm,error)||
    perm.childNodeToParent!=std::vector<std::uint32_t>{1,0,2,3})return 19;
 auto attachment=base;
 attachment.currentWorldPoints=attachment.parent.currentWorldPoints;
 attachment.referenceWorldPoints=attachment.parent.referenceWorldPoints;
 if(!provePhysicalSkinInheritedAttachmentNode(attachment,actual,0,0))return 20;
 attachment.currentWorldPoints[0][0]=.00000001;
 if(provePhysicalSkinInheritedAttachmentNode(attachment,actual,0,0))return 21;
 attachment.currentWorldPoints=attachment.parent.currentWorldPoints;
 attachment.referenceWorldPoints[0][0]=.00000001;
 if(provePhysicalSkinInheritedAttachmentNode(attachment,actual,0,0))return 22;
 attachment.referenceWorldPoints=attachment.parent.referenceWorldPoints;
 if(provePhysicalSkinInheritedAttachmentNode(attachment,actual,0,1))return 23;
 attachment.volumeNodeParentBarycentric[0]={.5,.5,0,0};
 if(provePhysicalSkinInheritedAttachmentNode(attachment,actual,0,0))return 24;
 attachment=base;attachment.currentWorldPoints=attachment.parent.currentWorldPoints;
 attachment.referenceWorldPoints=attachment.parent.referenceWorldPoints;
 attachment.parent.referenceWorldPoints[0][0]=std::numeric_limits<double>::quiet_NaN();
 if(provePhysicalSkinInheritedAttachmentNode(attachment,actual,0,0))return 25;
 auto refinement=base;
 refinement.currentWorldPoints.push_back({.2501,.25,.25});
 refinement.referenceWorldPoints=refinement.currentWorldPoints;
 refinement.volumeNodeParentTetrahedra.push_back(0);
 refinement.volumeNodeParentBarycentric.push_back({.25,.25,.25,.25});
 refinement.volumeNodeParentResidualWorldM.push_back({.0001,0,0});
 refinement.childTetrahedra={{4,1,2,3},{4,0,3,2},{4,0,1,3},{4,0,2,1}};
 PhysicalSkinSupportParentMap retained;
 if(!provePhysicalSkinRetainedSupportNodes(refinement,actual,base.parent.tetrahedra,.0004,retained,error)||
    retained.childNodeToParent!=std::vector<std::uint32_t>{0,1,2,3,NM_INVALID_INDEX}) {
  std::cerr<<error;return 26;
 }
 const auto kept=retained.childNodeToParent;const auto keptD=retained.maximumNodeDisplacementM;
 unsigned refinementRejects=0;
 const auto rejectRefinement=[&](const auto& bad){
  if(provePhysicalSkinRetainedSupportNodes(bad,actual,base.parent.tetrahedra,.0004,retained,error)||
     retained.childNodeToParent!=kept||retained.maximumNodeDisplacementM!=keptD)return false;
  ++refinementRejects;return true;
 };
 bad=refinement;bad.volumeNodeParentBarycentric[0]={0,1,0,0};if(!rejectRefinement(bad))return 27;
 bad=refinement;bad.volumeNodeParentBarycentric[4]={-.1,.35,.35,.4};if(!rejectRefinement(bad))return 28;
 bad=refinement;bad.volumeNodeParentBarycentric[4]={.2,.2,.2,.2};if(!rejectRefinement(bad))return 29;
 bad=refinement;bad.volumeNodeParentResidualWorldM[4][0]=0;if(!rejectRefinement(bad))return 30;
 bad=refinement;bad.parent.currentWorldPoints[0][0]=1;if(!rejectRefinement(bad))return 31;
 bad=refinement;bad.childTetrahedra[0][0]=5;if(!rejectRefinement(bad))return 32;
 bad=refinement;std::swap(bad.childTetrahedra[0][0],bad.childTetrahedra[0][1]);if(!rejectRefinement(bad))return 33;
 bad=refinement;bad.currentWorldPoints[0][0]=.001;if(!rejectRefinement(bad))return 34;
 bad=refinement;bad.volumeNodeParentTetrahedra[4]=1;if(!rejectRefinement(bad))return 35;
 bad=refinement;bad.parent.referenceWorldPoints[0][0]=std::numeric_limits<double>::quiet_NaN();if(!rejectRefinement(bad))return 36;
 if(!rejectRefinement(base))return 37;
 std::cout<<"retained_support_refinement_nodes=PASS inserted_nodes_unassigned=PASS invalid_cases="
          <<refinementRejects<<" failure_preserves_state=PASS new_supports_created=0\n";
 std::cout<<"support_parent_topology=PASS bounded_geometry_change=PASS reordered_nodes=PASS invalid_cases="
          <<rejected<<" failure_preserves_state=PASS unchanged_attachment_node=PASS moved_attachment_rejected=PASS contact_forces_created=0 physical_admission=false\n";
}
