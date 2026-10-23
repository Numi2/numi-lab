#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#pragma pack(push,1)
struct Header { char magic[8]; uint32_t abi,boneCount,vertexCount,indexCount,reserved; uint8_t sourceSha[32]; };
struct Bone { uint32_t body,firstVertex,vertexCount,firstIndex,indexCount,stable; float tx,ty,tz,qx,qy,qz,qw,scale; };
struct BoundBone { Bone b; uint32_t sourceRecord; };
struct Vertex { float x,y,z,nx,ny,nz; };
#pragma pack(pop)
static_assert(sizeof(Header)==60 && sizeof(Bone)==56 && sizeof(BoundBone)==60 && sizeof(Vertex)==24);
struct P { float x,y,z; };
P add(P a,P b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
P mul(P a,float s){return {a.x*s,a.y*s,a.z*s};}
P rotate(float qx,float qy,float qz,float qw,P p) {
 const P axis{qx,qy,qz};
 const P tc{2.0f*(axis.y*p.z-axis.z*p.y),2.0f*(axis.z*p.x-axis.x*p.z),2.0f*(axis.x*p.y-axis.y*p.x)};
 const P corr{qw*tc.x+axis.y*tc.z-axis.z*tc.y,qw*tc.y+axis.z*tc.x-axis.x*tc.z,qw*tc.z+axis.x*tc.y-axis.y*tc.x};
 return {p.x+corr.x,p.y+corr.y,p.z+corr.z};
}
P bodyPoint(Bone b,Vertex v){return add(P{b.tx,b.ty,b.tz},mul(rotate(b.qx,b.qy,b.qz,b.qw,P{v.x,v.y,v.z}),b.scale));}
int main(int argc,char**argv){
 if(argc!=4){std::cerr<<"usage: extract_small_face_geometry <nhbones> <excluded.csv> <output.csv>\n";return 2;}
 std::ifstream in(argv[2]);std::string line;
 if(!std::getline(in,line)){std::cerr<<"empty exclusion table\n";return 2;}
 std::map<uint64_t,std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>> wanted;
 while(std::getline(in,line)){std::stringstream ss(line);std::string field;std::array<std::string,11> f{};for(auto&x:f){if(!std::getline(ss,x,','))break;}
  const uint32_t body=std::stoul(f[0]),stable=std::stoul(f[1]),ord=std::stoul(f[2]),record=std::stoul(f[3]);const uint64_t global=std::stoull(f[4]);
  if(!wanted.emplace(global,std::make_tuple(body,stable,record,ord)).second){std::cerr<<"duplicate input key\n";return 2;}
 }
 std::ifstream f(argv[1],std::ios::binary);Header h{};f.read(reinterpret_cast<char*>(&h),sizeof(h));
 if(!f||std::string(h.magic,8)!="NHBONES1"||h.abi!=3){std::cerr<<"bad header\n";return 2;}
 std::vector<BoundBone> rec(h.boneCount);f.read(reinterpret_cast<char*>(rec.data()),rec.size()*sizeof(BoundBone));
 std::vector<Vertex> vs(h.vertexCount);f.read(reinterpret_cast<char*>(vs.data()),vs.size()*sizeof(Vertex));
 std::vector<uint32_t> ix(h.indexCount);f.read(reinterpret_cast<char*>(ix.data()),ix.size()*sizeof(uint32_t));
 if(!f){std::cerr<<"truncated\n";return 2;}
 std::ofstream out(argv[3]);if(!out){std::cerr<<"cannot write\n";return 2;}
 out<<"body_index,bone_stable_id,source_record,triangle_ordinal,global_source_face,area_m2,max_edge_m,quality,a0x,a0y,a0z,a1x,a1y,a1z,a2x,a2y,a2z,nx,ny,nz\n"<<std::setprecision(17);
 uint64_t global=0,found=0;
 for(auto rr:rec){Bone b=rr.b;double inv=1.0/std::sqrt(double(b.qx)*b.qx+double(b.qy)*b.qy+double(b.qz)*b.qz+double(b.qw)*b.qw);
 b.qx=float(double(b.qx)*inv);b.qy=float(double(b.qy)*inv);b.qz=float(double(b.qz)*inv);b.qw=float(double(b.qw)*inv);
  for(uint32_t ord=0;ord<b.indexCount/3;++ord,++global){
   auto it=wanted.find(global);if(it==wanted.end())continue;
   auto [body,stable,expectedRecord,expectedOrdinal]=it->second;if(body!=b.body||expectedRecord!=rr.sourceRecord||expectedOrdinal!=ord){std::cerr<<"source identity mismatch\n";return 2;}
   uint32_t ib=b.firstIndex+3*ord;P p0=bodyPoint(b,vs[ix[ib]]),p1=bodyPoint(b,vs[ix[ib+1]]),p2=bodyPoint(b,vs[ix[ib+2]]);
   double x1=double(p1.x)-p0.x,y1=double(p1.y)-p0.y,z1=double(p1.z)-p0.z,x2=double(p2.x)-p0.x,y2=double(p2.y)-p0.y,z2=double(p2.z)-p0.z;
   double nx=y1*z2-z1*y2,ny=z1*x2-x1*z2,nz=x1*y2-y1*x2,area=.5*std::sqrt(nx*nx+ny*ny+nz*nz);
   double e01=x1*x1+y1*y1+z1*z1,e02=x2*x2+y2*y2+z2*z2;
   double dx=double(p2.x)-p1.x,dy=double(p2.y)-p1.y,dz=double(p2.z)-p1.z;
   double maxEdge=std::sqrt(std::max({e01,e02,dx*dx+dy*dy+dz*dz}));
   double quality=(2.0*area)/(maxEdge*maxEdge);
   double nl=2.0*area;nx/=nl;ny/=nl;nz/=nl;
   out<<body<<","<<stable<<","<<rr.sourceRecord<<","<<ord<<","<<global<<","<<area<<","<<maxEdge<<","<<quality<<","
      <<p0.x<<","<<p0.y<<","<<p0.z<<","<<p1.x<<","<<p1.y<<","<<p1.z<<","
      <<p2.x<<","<<p2.y<<","<<p2.z<<","<<nx<<","<<ny<<","<<nz<<"\n";++found;
  }
 }
 if(found!=wanted.size()){std::cerr<<"missing requested faces "<<found<<"/"<<wanted.size()<<"\n";return 2;}
 std::cout<<"PASS emitted "<<found<<" exact float32-transformed triangles\n";
}
