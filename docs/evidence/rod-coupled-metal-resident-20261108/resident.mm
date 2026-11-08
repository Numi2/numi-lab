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
static void hex(const void* p,size_t n){const auto*b=static_cast<const unsigned char*>(p);static const char* a="0123456789abcdef";std::cout<<'"';for(size_t i=0;i<n;i++)std::cout<<a[b[i]>>4]<<a[b[i]&15];std::cout<<'"';}
int main(int argc,char**argv){@autoreleasepool{
 if(argc!=3)return 2;bool moving=std::string(argv[2])=="moving";unsigned n,ignored;if(!(std::cin>>n>>ignored)||n<3||n>ROD_N)return 3;
 std::vector<std::array<double,3>> q(n),v(n);std::vector<double> t(n-1),w(n-1);std::array<std::array<double,3>,2> target;
 for(auto& a:q)for(auto& z:a)std::cin>>z;for(auto& a:v)for(auto& z:a)std::cin>>z;for(auto& z:t)std::cin>>z;for(auto& z:w)std::cin>>z;for(auto& a:target)for(auto& z:a)std::cin>>z;if(!std::cin)return 4;
 constexpr double R=2e-5,RHO=1100.,EA=2.3818718058345718,L=.11758755785232791,H=.001;
 double ell=L/(n-1),EI=EA*R*R/4,GJ=EI/1.35;ResidentState initial{};RodInput& in=initial.input;in.n=n;
 for(unsigned i=0;i<n-1;i++)for(unsigned j=0;j<3;j++){double d=q[i+1][j]-q[i][j];float hi=float(d);in.edgeHigh[3*i+j]=hi;in.edgeLow[3*i+j]=float(d-double(hi));}
 for(unsigned i=0;i<n;i++)for(unsigned j=0;j<3;j++)in.vel[3*i+j]=float(v[i][j]);
 double center=0;for(double x:t)center+=x;center/=double(n-1);
 for(unsigned i=0;i<n-1;i++){in.twist[i]=float(t[i]-center);in.omega[i]=float(w[i]);in.inertia[i]=float(RHO*std::numbers::pi*std::pow(R,4)/2*ell);}
 for(unsigned i=0;i<n;i++)in.mass[i]=float(RHO*std::numbers::pi*R*R*ell*((i==0||i==n-1)?.5:1));
 in.ellHigh=float(ell);in.ellLow=float(ell-double(in.ellHigh));in.ea=float(EA);in.ei=float(EI);in.gj=float(GJ);in.h=float(H);
 StepControl base{};if(moving){double l=0;for(unsigned j=0;j<3;j++)l+=std::pow(q[1][j]-q[0][j],2);l=std::sqrt(l);for(unsigned j=0;j<3;j++){double d=(q[1][j]-q[0][j])/l*1e-6;base.targetHigh[3+j]=float(d);base.targetLow[3+j]=float(d-double(base.targetHigh[3+j]));}}
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice();NSError* err=nil;id<MTLLibrary> lib=[dev newLibraryWithFile:[NSString stringWithUTF8String:argv[1]] error:&err];
 if(!lib){std::cerr<<"library: "<<[[err description] UTF8String]<<"\n";return 5;}
 auto pipeline=[&](NSString* name)->id<MTLComputePipelineState>{id<MTLFunction> f=[lib newFunctionWithName:name];return [dev newComputePipelineStateWithFunction:f error:&err];};
 id<MTLComputePipelineState> prepare=pipeline(@"resident_prepare"), solve=pipeline(@"rod_newton"), commit=pipeline(@"resident_commit");if(!prepare||!solve||!commit){std::cerr<<"pipeline: "<<[[err description] UTF8String]<<"\n";return 6;}
 auto buf=[&](const void* p,size_t size)->id<MTLBuffer>{return p?[dev newBufferWithBytes:p length:size options:MTLResourceStorageModeShared]:[dev newBufferWithLength:size options:MTLResourceStorageModeShared];};
 id<MTLBuffer> state[2]={buf(&initial,sizeof(initial)),buf(&initial,sizeof(initial))};id<MTLBuffer> scratch[2]={buf(nullptr,sizeof(RodInput)),buf(nullptr,sizeof(RodInput))};id<MTLBuffer> out[2]={buf(nullptr,sizeof(RodOutput)),buf(nullptr,sizeof(RodOutput))};id<MTLBuffer> records[2]={buf(nullptr,10*sizeof(ResidentRecord)),buf(nullptr,11*sizeof(ResidentRecord))};
 id<MTLCommandQueue> queue=[dev newCommandQueue];id<MTLCommandBuffer> cmd=[queue commandBuffer];
 for(unsigned arm=0;arm<2;arm++)for(unsigned i=0;i<(arm?11u:10u);i++){
  StepControl c=base;c.inject=(arm==1&&i==4)?1:0;
  id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];[e setComputePipelineState:prepare];[e setBuffer:state[arm] offset:0 atIndex:0];[e setBuffer:scratch[arm] offset:0 atIndex:1];[e setBuffer:out[arm] offset:0 atIndex:2];[e setBytes:&c length:sizeof(c) atIndex:3];[e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[e endEncoding];
  e=[cmd computeCommandEncoder];[e setComputePipelineState:solve];[e setBuffer:scratch[arm] offset:0 atIndex:0];[e setBuffer:out[arm] offset:0 atIndex:1];[e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[e endEncoding];
  e=[cmd computeCommandEncoder];[e setComputePipelineState:commit];[e setBuffer:state[arm] offset:0 atIndex:0];[e setBuffer:out[arm] offset:0 atIndex:1];[e setBuffer:records[arm] offset:0 atIndex:2];[e setBytes:&i length:sizeof(i) atIndex:3];[e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[e endEncoding];
 }
 [cmd commit];[cmd waitUntilCompleted];if(cmd.status!=MTLCommandBufferStatusCompleted){std::cerr<<"command: "<<[[cmd.error description] UTF8String]<<"\n";return 7;}
 std::cout<<std::setprecision(17)<<"{\"device\":\""<<[[dev name] UTF8String]<<"\",\"moving\":"<<(moving?"true":"false");
 for(unsigned arm=0;arm<2;arm++){
  std::cout<<",\""<<(arm?"fault":"clean")<<"\":[";auto* rows=static_cast<const ResidentRecord*>(records[arm].contents);unsigned count=arm?11:10;
  for(unsigned i=0;i<count;i++){
   if(i)std::cout<<",";const auto&r=rows[i];const auto&s=r.after;std::cout<<"{\"status\":"<<r.output.status<<",\"iterations\":"<<r.output.iterations<<",\"accepted_steps\":"<<s.acceptedSteps<<",\"state_bytes\":";hex(&s,sizeof(s));
   if(arm==1&&i==4){std::cout<<",\"before_bytes\":";hex(&r.before,sizeof(r.before));}
   if(r.output.status==0){
    auto qo=q,vo=v;auto to=t,wo=w;for(unsigned k=0;k<n;k++)for(unsigned j=0;j<3;j++){unsigned x=3*k+j;qo[k][j]+=double(s.sumHigh[x])+double(s.sumLow[x]);vo[k][j]=double(s.input.vel[x]);}
    for(unsigned k=0;k<n-1;k++){to[k]=center+double(s.input.twist[k]);wo[k]=double(s.input.omega[k]);}
    std::array<std::array<double,3>,2> f{};for(unsigned k=0;k<2;k++)for(unsigned j=0;j<3;j++)f[k][j]=s.lastReaction[3*k+j];
    auto vectors=[&](const char* name,const auto& values){std::cout<<",\""<<name<<"\":[";bool first=true;for(const auto& x:values){if(!first)std::cout<<",";first=false;vec(x);}std::cout<<"]";};
    auto scalars=[&](const char* name,const auto& values){std::cout<<",\""<<name<<"\":[";bool first=true;for(double x:values){if(!first)std::cout<<",";first=false;std::cout<<x;}std::cout<<"]";};
    vectors("q",qo);vectors("v",vo);scalars("theta",to);scalars("omega",wo);vectors("reaction",f);
   }
   std::cout<<"}";
  }
  std::cout<<"]";
 }
 std::cout<<"}\n";return 0;
}}
