#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "shared.h"
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>
static void vec(const std::array<double,3>& v){std::cout<<"["<<v[0]<<","<<v[1]<<","<<v[2]<<"]";}
int main(int argc,char**argv){@autoreleasepool{
 if(argc!=2)return 2;unsigned n,ignored;if(!(std::cin>>n>>ignored)||n<3||n>ROD_N)return 3;
 std::vector<std::array<double,3>> q(n),v(n);std::vector<double> t(n-1),w(n-1);std::array<std::array<double,3>,2> target;
 for(auto& a:q)for(auto& z:a)std::cin>>z;for(auto& a:v)for(auto& z:a)std::cin>>z;
 for(auto& z:t)std::cin>>z;for(auto& z:w)std::cin>>z;for(auto& a:target)for(auto& z:a)std::cin>>z;if(!std::cin)return 4;
 constexpr double R=2e-5,RHO=1100.,EA=2.3818718058345718,L=.11758755785232791,H=.001;
 double ell=L/(n-1),EI=EA*R*R/4,GJ=EI/1.35;RodInput in{};in.n=n;
 for(unsigned i=0;i<n-1;i++)for(unsigned j=0;j<3;j++){double d=q[i+1][j]-q[i][j];float hi=float(d);in.edgeHigh[3*i+j]=hi;in.edgeLow[3*i+j]=float(d-double(hi));}
 for(unsigned i=0;i<n;i++)for(unsigned j=0;j<3;j++)in.vel[3*i+j]=float(v[i][j]);
 double twistCenter=0;for(double x:t)twistCenter+=x;twistCenter/=double(n-1);
 for(unsigned i=0;i<n-1;i++){in.twist[i]=float(t[i]-twistCenter);in.omega[i]=float(w[i]);in.inertia[i]=float(RHO*std::numbers::pi*std::pow(R,4)/2*ell);}
 for(unsigned i=0;i<n;i++)in.mass[i]=float(RHO*std::numbers::pi*R*R*ell*((i==0||i==n-1)?.5:1));
 for(unsigned k=0;k<2;k++)for(unsigned j=0;j<3;j++){double x=target[k][j]-q[k? n-1:0][j];in.targetDelta[3*k+j]=float(x);in.targetDeltaLow[3*k+j]=float(x-double(in.targetDelta[3*k+j]));}
 in.ellHigh=float(ell);in.ellLow=float(ell-double(in.ellHigh));in.ea=float(EA);in.ei=float(EI);in.gj=float(GJ);in.h=float(H);
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice();NSError* err=nil;
 id<MTLLibrary> lib=[dev newLibraryWithFile:[NSString stringWithUTF8String:argv[1]] error:&err];
 if(!lib){std::cerr<<"library: "<<[[err description] UTF8String]<<"\n";return 5;}
 id<MTLFunction> fn=[lib newFunctionWithName:@"rod_newton"];id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:fn error:&err];
 if(!p){std::cerr<<"pipeline: "<<[[err description] UTF8String]<<"\n";return 6;}
 id<MTLBuffer> ib=[dev newBufferWithBytes:&in length:sizeof(in) options:MTLResourceStorageModeShared];
 RodOutput init{};id<MTLBuffer> ob=[dev newBufferWithBytes:&init length:sizeof(init) options:MTLResourceStorageModeShared];
 id<MTLCommandQueue> queue=[dev newCommandQueue];id<MTLCommandBuffer> cmd=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
 [enc setComputePipelineState:p];[enc setBuffer:ib offset:0 atIndex:0];[enc setBuffer:ob offset:0 atIndex:1];[enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];
 if(cmd.status!=MTLCommandBufferStatusCompleted){std::cerr<<"command: "<<[[cmd.error description] UTF8String]<<"\n";return 7;}
 RodOutput out=*static_cast<RodOutput*>(ob.contents);bool ok=out.status==0;
 std::cout<<std::setprecision(17)<<"{\"status\":"<<(ok?0:14)<<",\"gpu\":"<<(ok?0:4)<<",\"published\":"<<(ok?"true":"false")<<",\"states\":"<<(ok?1:0)<<",\"reactions\":"<<(ok?2:0)<<",\"candidate_status\":"<<out.status<<",\"iterations\":"<<out.iterations<<",\"backtracks\":"<<out.backtracks<<",\"internal_linear\":"<<out.maxLinear<<",\"internal_angular\":"<<out.maxAngular;
 if(ok){
  auto vectors=[&](const char* name,const auto& values){std::cout<<",\""<<name<<"\":[";bool first=true;for(const auto& x:values){if(!first)std::cout<<",";first=false;vec(x);}std::cout<<"]";};
  auto scalars=[&](const char* name,const auto& values){std::cout<<",\""<<name<<"\":[";bool first=true;for(double x:values){if(!first)std::cout<<",";first=false;std::cout<<x;}std::cout<<"]";};
  auto qo=q,vo=v;auto to=t,wo=w;for(unsigned i=0;i<n;i++)for(unsigned j=0;j<3;j++){double d=double(out.displacement[3*i+j])+double(out.displacementLow[3*i+j]);qo[i][j]+=d;vo[i][j]=d/H;}
  for(unsigned i=0;i<n-1;i++){to[i]=twistCenter+double(out.twistDelta[i]);wo[i]=(to[i]-t[i])/H;}
  std::array<std::array<double,3>,2> f{};for(unsigned i=0;i<2;i++)for(unsigned j=0;j<3;j++)f[i][j]=out.reaction[3*i+j];
  vectors("q",qo);vectors("v",vo);scalars("theta",to);scalars("omega",wo);vectors("reaction",f);
 }
 std::cout<<"}\n";return 0;
}}
