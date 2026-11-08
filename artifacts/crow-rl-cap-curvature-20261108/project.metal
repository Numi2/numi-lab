#include "MetalUnifiedQualitySolver.metal"
kernel void cap_projection_probe(device const MRUnifiedQualityBlockGPU* b [[buffer(0)]],
 device const float* w [[buffer(1)]],device float* output [[buffer(2)]],
 constant uint& count [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i>=count)return;
 float v[6],p[6],d[36];for(uint k=0;k<6;++k){v[k]=w[i*6+k];p[k]=0;}
 for(uint k=0;k<36;++k)d[k]=0;
 float obj=0,feas=0,comp=0;
 bool ok=projectConeBlock(b[i],v,p,d,obj,feas,comp,1.0e-10f);
 for(uint k=0;k<48;++k)output[i*48+k]=0;
 for(uint k=0;k<6;++k)output[i*48+k]=p[k];
 for(uint k=0;k<36;++k)output[i*48+6+k]=d[k];
 output[i*48+42]=obj;output[i*48+43]=feas;output[i*48+44]=comp;output[i*48+45]=ok?1:0;
}
