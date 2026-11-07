// Test-only wrapper includes the actual shared helper, without changing the
// production metallib or treating this as an end-to-end world qualification.
#include "../../../src/metal/MetalWorldContact.metal"

kernel void numi_wave32_qualification(
    device const float* input [[buffer(0)]],
    device const MREvaluatedConstraintIRConeGPU* cones [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    const uint base=16u*id;
    float matrix[3][3];
    for(uint i=0;i<3;++i)for(uint j=0;j<3;++j)matrix[i][j]=input[base+3u*i+j];
    float inv[3][3],normal=0.f;
    if(!invert3x3(matrix,inv,&normal)) {
        output[4u*id]=INFINITY;return;
    }
    MRWave32PreconditionerGPU p={};
    p.row0.xyz=float3(inv[0][0],inv[0][1],inv[0][2]);
    p.row1.xyz=float3(inv[1][0],inv[1][1],inv[1][2]);
    p.row2.xyz=float3(inv[2][0],inv[2][1],inv[2][2]);
    float3 impulse=0.f;
    const float3 initialRhs=float3(input[base+9],input[base+10],input[base+11]);
    const uint iterations=uint(input[base+12]);
    const bool duplicateRows=input[base+13]>0.f;
    for(uint iter=0;iter<iterations;++iter) {
        float3 response=float3(0.f);
        for(uint i=0;i<3;++i)for(uint j=0;j<3;++j)response[i]+=matrix[i][j]*impulse[j];
        if(duplicateRows)response.x+=matrix[0][0]*impulse.x;
        impulse=normalFirstContactCandidate(impulse,initialRhs-response,p,normal,cones[id]);
    }
    output[4u*id]=impulse.x;
    output[4u*id+1u]=impulse.y;
    output[4u*id+2u]=impulse.z;
    output[4u*id+3u]=normal;
}
