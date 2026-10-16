#include "metalrobo/compensated_geometry_gpu.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using S=MRCompensatedScalar;
using V=MRCompensatedPositionGPU;
constexpr float kGeometryFloor=1.0e-12f;
V vec(float x,float y,float z){return {{x,y,z,0.0f},{0.0f,0.0f,0.0f,0.0f}};}
V add(V a,V b){return mrCompensatedVectorAdd(a,b);}
V neg(V a){return mrCompensatedVectorNegate(a);}
V sub(V a,V b){return add(a,neg(b));}
V scale(V a,S b){return mrCompensatedVectorScale(a,b);}
S ssub(S a,S b){return mrCompensatedAdd(a,mrCompensatedNegate(b));}
int sign(S a){if(a.high<0)return -1;if(a.high>0)return 1;if(a.low<0)return -1;if(a.low>0)return 1;return 0;}
bool le(S a,S b){return sign(ssub(a,b))<=0;}
bool ge(S a,S b){return sign(ssub(a,b))>=0;}
S mul(S a,S b){return mrCompensatedMultiply(a,b);}
S divs(S a,S b){return mrCompensatedDivide(a,b);}
S denom(S a){S f{kGeometryFloor,0.0f};return le(a,f)?f:a;}
S oneMinus(S a){return ssub({1.0f,0.0f},a);}
S dotv(V a,V b){return mrCompensatedVectorDot(a,b);}
struct Closest { V q; S bary[3]; const char* branch; };
Closest closest(V p,V a,V b,V c){
 const auto ab=sub(b,a),ac=sub(c,a),ap=sub(p,a);
 const auto d1=dotv(ab,ap),d2=dotv(ac,ap);
 if(le(d1,{0,0})&&le(d2,{0,0}))return {a,{{1,0},{0,0},{0,0}},"vertex_a"};
 const auto bp=sub(p,b);const auto d3=dotv(ab,bp),d4=dotv(ac,bp);
 if(ge(d3,{0,0})&&le(d4,d3))return {b,{{0,0},{1,0},{0,0}},"vertex_b"};
 const auto vc=ssub(mul(d1,d4),mul(d3,d2));
 if(le(vc,{0,0})&&ge(d1,{0,0})&&le(d3,{0,0})){
  const auto v=divs(d1,denom(ssub(d1,d3)));
  return {add(a,scale(ab,v)),{oneMinus(v),v,{0,0}},"edge_ab"};
 }
 const auto cp=sub(p,c);const auto d5=dotv(ab,cp),d6=dotv(ac,cp);
 if(ge(d6,{0,0})&&le(d5,d6))return {c,{{0,0},{0,0},{1,0}},"vertex_c"};
 const auto vb=ssub(mul(d5,d2),mul(d1,d6));
 if(le(vb,{0,0})&&ge(d2,{0,0})&&le(d6,{0,0})){
  const auto w=divs(d2,denom(ssub(d2,d6)));
  return {add(a,scale(ac,w)),{oneMinus(w),{0,0},w},"edge_ac"};
 }
 const auto va=ssub(mul(d3,d6),mul(d5,d4));
 const auto d4d3=ssub(d4,d3),d5d6=ssub(d5,d6);
 if(le(va,{0,0})&&ge(d4d3,{0,0})&&ge(d5d6,{0,0})){
  const auto w=divs(d4d3,denom(mrCompensatedAdd(d4d3,d5d6)));
  return {add(b,scale(sub(c,b),w)),{{0,0},oneMinus(w),w},"edge_bc"};
 }
 const auto d=denom(mrCompensatedAdd(mrCompensatedAdd(va,vb),vc));
 const auto v=divs(vb,d),w=divs(vc,d);
 const auto q=add(add(a,scale(ab,v)),scale(ac,w));
 return {q,{ssub(ssub({1,0},v),w),v,w},"face"};
}
float word(const std::uint8_t* p){float x;std::memcpy(&x,p,4);return x;}
V readv(const std::uint8_t* p){return vec(word(p),word(p+4),word(p+8));}
void pair(S x){std::cout<<"["<<x.high<<","<<x.low<<"]";}
void printv(V x){std::cout<<"[["<<x.high.x<<","<<x.low.x<<"],["<<x.high.y<<","<<x.low.y<<"],["<<x.high.z<<","<<x.low.z<<"]]";}
void eval(const char* label,unsigned row,const std::uint8_t* r,unsigned po,unsigned to,unsigned nativeOffset){
 const auto p=readv(r+po),a=readv(r+to),b=readv(r+to+16),c=readv(r+to+32);
 const auto z=closest(p,a,b,c);const auto delta=sub(p,z.q);
 const auto distance=mrCompensatedSqrt(dotv(delta,delta));
 std::cout<<"{\"row\":"<<row<<",\"endpoint\":\""<<label<<"\",\"branch\":\""<<z.branch
  <<"\",\"native_separation\":"<<word(r+nativeOffset)<<",\"distance_words\":";
 pair(distance);std::cout<<",\"closest_words\":";printv(z.q);std::cout<<",\"barycentric_words\":[";
 pair(z.bary[0]);std::cout<<",";pair(z.bary[1]);std::cout<<",";pair(z.bary[2]);std::cout<<"]}\n";
}
int main(int argc,char**argv){
 if(argc!=2)throw std::runtime_error("usage: closest_vt_dd <native500-output.bin>");
 std::ifstream input(argv[1],std::ios::binary);if(!input)throw std::runtime_error("cannot open output");
 std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),{});
 constexpr std::size_t record=14u*16u;if(bytes.size()!=24u*record)throw std::runtime_error("output extent mismatch");
 std::cout<<std::setprecision(17);
 for(unsigned row=0;row<8;++row){const auto*r=bytes.data()+row*record;
  eval("current",row,r,0u,16u,196u);
  eval("contact_limiter",row,r,64u,80u,204u);
  eval("fem_applied",row,r,128u,144u,212u);
 }
}
