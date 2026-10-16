#include "metal/common.metalinc"
#include "metal/fem.metalinc"
#include "base_cgls.metalinc"
inline void fgmresCompensatedAdd(thread float& sum,thread float& correction,const float value){
 const float adjusted=value-correction;const float next=sum+adjusted;
 correction=(next-sum)-adjusted;sum=next;
}
#include "base_cgls_runtime.metalinc"
using namespace metal;
namespace numi_matter_metal {
kernel void nm_cgls_arena_8192_identity_action(constant uint& count [[buffer(0)]],
 device const float4* input [[buffer(1)]],device float4* output [[buffer(2)]],
 uint i [[thread_position_in_grid]]) {
 if(i>=count)return;output[i]=i<count/2u?float4(input[i].xyz,0.0f):float4(0.0f);
}
}
