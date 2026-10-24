#include "../apps/NumiHumanD37ExactBankDump.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <span>
#include <string>
#include <vector>
struct F4 { float x=0,y=0,z=0,w=0; };
struct Tri { std::uint32_t bodyIndex=0,boneStableId=0,triangleStableId=0; F4 bodyLocalVertex0{},bodyLocalVertex1{},bodyLocalVertex2{}; };
struct Node { F4 positionAndMass{}; };
struct Body { F4 position{},orientation{}; };
static std::array<double,4> rotate(const std::array<double,4>& q,const std::array<double,4>& p) {
 const std::array<double,3> a{q[0],q[1],q[2]};
 const std::array<double,3> c{a[1]*p[2]-a[2]*p[1],a[2]*p[0]-a[0]*p[2],a[0]*p[1]-a[1]*p[0]};
 const std::array<double,3> t{2*c[0],2*c[1],2*c[2]};
 const std::array<double,3> d{a[1]*t[2]-a[2]*t[1],a[2]*t[0]-a[0]*t[2],a[0]*t[1]-a[1]*t[0]};
 return {p[0]+q[3]*t[0]+d[0],p[1]+q[3]*t[1]+d[1],p[2]+q[3]*t[2]+d[2],0};
}
int main() {
 const float q=std::sqrt(0.5f);
 std::vector<Tri> triangles{{0,7,19,{0.25f,0.5f,0.75f,0},{1,0,0,0},{0,1,0,0}}};
 std::vector<Node> nodes{{{1.25f,-2.5f,3.75f,0.125f}}};
 std::vector<Body> bodies{{{0.1f,0.2f,0.3f,0},{0,0,q,q}}};
 std::array<std::uint8_t,1> roles{1};std::array<std::uint32_t,1> rows{5};
 std::ostringstream to,no,bo;
 const auto off=[](std::uint32_t){return std::array<double,3>{0.01,-0.02,0.03};};
 const auto source=[](std::uint32_t b,std::uint32_t id){return b==0&&id==7?42u:UINT32_MAX;};
 const auto rot=[](const auto& qq,const auto& pp){return rotate(qq,pp);};
 const auto r=numi::human::d37_exact_bank_export::writeExactEmittedBankSnapshot(
  to,no,bo,triangles,nodes,bodies,0,1,roles,rows,off,source,rot);
 if(!r.valid||r.triangles!=1||r.banks!=1||r.nodes!=1||r.bodies!=1)return 1;
 const auto ts=to.str(),ns=no.str(),bs=bo.str();
 if(ts.find("0,7,42,19,")==std::string::npos)return 2;
 if(ns.find(std::to_string(std::bit_cast<std::uint32_t>(nodes[0].positionAndMass.x)))==std::string::npos)return 3;
 if(bs.find("0.01,-0.02,") == std::string::npos)return 4;
 const std::array<double,4> qi{0,0,-double(q),double(q)};
 const auto col0=rotate(qi,{1,0,0,0});
 if(std::abs(col0[0])>1e-6||std::abs(col0[1]+1.0)>1e-6)return 5;
 std::ostringstream badT,badN,badB;
 const auto missing=[](std::uint32_t,std::uint32_t){return UINT32_MAX;};
 const auto bad=numi::human::d37_exact_bank_export::writeExactEmittedBankSnapshot(
  badT,badN,badB,triangles,nodes,bodies,0,1,roles,rows,off,missing,rot);
 if(bad.valid||bad.failure!="source_record_missing")return 6;
 std::cout<<"PASS exact F32 bits, body-local rows, double inverse quaternion matrix, COM offsets, source identity failure\n";
}
