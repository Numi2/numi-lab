#pragma once
#import <Metal/Metal.h>
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/VisualPresentation.hpp"
#include "numi/matter/human_respiration.h"
#include <cstddef>
#include <cmath>
#include <stdexcept>
// Presentation of the existing accepted blood inventory. No physical state.
class NumiHumanRestingBloodVisual final {
public:
 struct Layout { mr_uint4 ranges{}; };
 static constexpr unsigned capacity=NM_HUMAN_EXTERNAL_BLOOD_PARCEL_CAPACITY;
 static_assert(sizeof(NMHumanExternalBloodParcelGPU)==4*sizeof(mr_float4));
 static_assert(offsetof(NMHumanExternalBloodParcelGPU,positionAndVolume)==0);
 static_assert(offsetof(NMHumanExternalBloodParcelGPU,velocityAndMass)==16);
 static_assert(offsetof(NMHumanExternalBloodParcelGPU,gasAndFlags)==32);
 static_assert(offsetof(NMHumanExternalBloodParcelGPU,intrinsicAngularMomentum)==48);
 static Layout appendGeometry(metalrobo::VisualAssetPackV2& pack) {
  Layout l; l.ranges={unsigned(pack.vertices.size()),unsigned(pack.primitives.size()),unsigned(pack.instances.size()),capacity};
  const unsigned mat=unsigned(pack.materials.size());
  MRVisualMaterialGPUV2 material{};
  material.baseColorAndOpacity={0.32f,0.006f,0.009f,1};
  material.surface={0.20f,0,1,1};
  material.coatingAndAlphaCutoff={0.5f,0.12f,0.35f,0.5f};
  material.textureIndices0={MR_INVALID_INDEX,MR_INVALID_INDEX,MR_INVALID_INDEX,MR_INVALID_INDEX};
  material.textureIndices1=material.textureIndices0;
  material.flags={MR_VISUAL_ALPHA_OPAQUE,0,0,53000};
  material.reserved=material.textureIndices0; pack.materials.push_back(material);
  constexpr float axes[6][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
  constexpr unsigned faces[24]={0,2,4,2,1,4,1,3,4,3,0,4,2,0,5,1,2,5,3,1,5,0,3,5};
  for(unsigned p=0;p<capacity;++p) {
   const unsigned first=unsigned(pack.vertices.size()),index=unsigned(pack.indices.size());
   for(const auto& a:axes) {
    MRVisualVertexGPUV2 v{};v.position={a[0]*.001f,a[1]*.001f,a[2]*.001f,1};
    v.normalAndTangentSign={a[0],a[1],a[2],1};
    const bool referenceY=std::abs(a[1])<0.9f;
    float tx=referenceY?a[2]:0.0f;
    float ty=referenceY?0.0f:-a[2];
    float tz=referenceY?-a[0]:a[1];
    const float tangentLength=std::sqrt(tx*tx+ty*ty+tz*tz);
    tx/=tangentLength;ty/=tangentLength;tz/=tangentLength;
    v.tangent={tx,ty,tz,0};
    v.color={1,1,1,1};pack.vertices.push_back(v);
   }
   for(const auto f:faces) pack.indices.push_back(first+f);
   MRVisualPrimitiveGPUV2 primitive{};
   primitive.geometry={index,24u,mat,l.ranges.z+p};
   primitive.identity={52002u,l.ranges.z+p,MR_INVALID_INDEX,53000u+p};
   primitive.boundsMinimum={-.001f,-.001f,-.001f,1};
   primitive.boundsMaximum={.001f,.001f,.001f,1};
   MRVisualInstanceGPUV2 instance{};instance.translationAndScale={0,0,0,1};instance.orientation={0,0,0,1};
   instance.binding={0,MR_INVALID_INDEX,MR_VISUAL_BINDING_WORLD,0};
   instance.identity=primitive.identity;instance.geometry={l.ranges.y+p,1,0,0};
   pack.primitives.push_back(primitive);pack.instances.push_back(instance);
  }
  pack.preprocessingProvenance+="/accepted_gpu_external_blood_inventory_visual";
  return l;
 }
 NumiHumanRestingBloodVisual(id<MTLDevice> dev,id<MTLLibrary> library,Layout layout):device_(dev),layout_(layout) {
  NSError* error=nil;auto f=[library newFunctionWithName:@"nm_human_resting_blood_visual"];
  pipeline_=f?[dev newComputePipelineStateWithFunction:f error:&error]:nil;
  if(!pipeline_)throw std::runtime_error(error?error.localizedDescription.UTF8String:"missing blood visual kernel");
 }
 bool encode(const metalrobo::MetalHybridMeshDeformationLease& lease,id<MTLBuffer> state,bool visibilityOnly)const noexcept {
  constexpr std::size_t offset=offsetof(NMHumanRespirationState,externalBlood)+offsetof(NMHumanExternalBloodInventoryGPU,parcels);
  if(!state||!lease.encoder||!lease.encoder->valid()||!lease.meshVertices||!lease.meshPrimitives||!lease.meshInstances||
     state.device.registryID!=device_.registryID||state.length<offset+capacity*sizeof(NMHumanExternalBloodParcelGPU)||
     lease.meshVertexCount<layout_.ranges.x+6*capacity||lease.meshPrimitiveCount<layout_.ranges.y+capacity||
     lease.meshInstanceCount<layout_.ranges.z+capacity)return false;
  const auto& e=*lease.encoder;const unsigned phase=visibilityOnly?1u:0u;
  e.setPipeline(e.context,(__bridge void*)pipeline_);
  e.setBytes(e.context,&layout_.ranges,sizeof(layout_.ranges),0);
  e.setBuffer(e.context,(__bridge void*)state,offset,1);
  e.setBuffer(e.context,lease.meshVertices,0,2);
  e.setBuffer(e.context,lease.meshPrimitives,0,3);
  e.setBuffer(e.context,lease.meshInstances,0,4);
  e.setBytes(e.context,&phase,sizeof(phase),5);
  e.dispatchThreads(e.context,capacity,64);
  return true;
 }
private:
 __strong id<MTLDevice> device_=nil;
 __strong id<MTLComputePipelineState> pipeline_=nil;
 Layout layout_{};
};
