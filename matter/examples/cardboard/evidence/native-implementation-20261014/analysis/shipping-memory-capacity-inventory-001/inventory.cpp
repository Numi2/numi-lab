#include "cardboard_shipping_memory.hpp"
#include <iostream>
using namespace numi::cardboard;
int main() {
 const auto s=screenShippingPrebuildMemory(1.025,0.0079);
 std::cout<<"{\"kind\":\"conservative_prebuild_inventory\",\"cases\":[";
 bool first=true;
 for(unsigned cap:{0u,65536u,131072u,262144u}) for(bool sparse:{false,true}) for(bool alias:{false,true}) {
  ShippingPrebuildArenaOptions o; o.rigidProxyCount=12; o.rigidBodyCount=12;
  o.cohesiveFaceUpperBound=s.nominalTetrahedronCount/5; o.compactImmutableSurfaceMap=true;
  o.requireFullContactResidualDescent=true; o.deformableSelfContactEnabled=true;
  o.deformableContactCapacity=cap; o.fullContactArmijoTiledCohesive=true;
  if(sparse) configureShippingPrebuildSparsePrivateContact(o,"65536",nullptr,true,true);
  // Surface alias is a runtime-only option. Report prebuild here without inventing its support.
  if(alias) continue;
  auto b=estimateShippingPrebuildMemoryBudget(s,o);
  if(!first)std::cout<<",";first=false;
  std::cout<<"{\"deformable_capacity\":"<<cap<<",\"sparse_rigid\":"<<sparse
   <<",\"peak\":"<<b.estimatedPeakDomainBytes<<",\"metal\":"<<b.estimatedMetalPeakBytes
   <<",\"noncontact\":"<<b.measuredNoncontactRuntimeEnvelopeBytes
   <<",\"persistent_contact\":"<<b.persistentContactBytes
   <<",\"rigid_runtime\":"<<b.rigidContactRuntimeBytes
   <<",\"cohesive_runtime\":"<<b.cohesiveRuntimeBytes
   <<",\"private_rigid\":"<<b.requiredRigidContactArmijoBytes
   <<",\"private_deformable\":"<<b.requiredFullContactArmijoBytes
   <<",\"private_cohesive\":"<<b.requiredCohesiveArmijoBytes
   <<",\"host_mesh\":"<<b.hostMeshAndCompileBytes
   <<",\"host_checkpoints\":"<<b.hostCheckpointBytes
   <<",\"snapshot\":"<<b.snapshotReadbackBytes<<",\"upload\":"<<b.uploadStagingBytes
   <<",\"admitted\":"<<b.withinLimit<<"}";
 }
 std::cout<<"]}\n";
}
