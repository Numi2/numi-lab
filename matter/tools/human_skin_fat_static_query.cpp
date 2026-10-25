#include "numi/matter/fem_body_triangle_contact_sample.hpp"
#include "NumiHumanD37InitialSkinBoneWitness.hpp"
#include "numi/matter/fem_body_contact_static_bvh.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace d37=numi::human::d37_initial_skin_bone_witness;
namespace w=numi::human::initial_skin_bone_contact_witness;
namespace bvh=numi::matter::fem_body_contact_bvh;
struct QNode { nm_float4 positionAndMass{}; };
struct QQuat { float x=0,y=0,z=0,w=1; };
struct QBody { nm_float4 position{}; QQuat orientation{}; };
struct QTree { std::uint32_t body,firstNode,nodeCount,firstTriangle,triangleCount; };
static std::vector<std::string> split(const std::string& s) {
 std::vector<std::string> r;std::size_t p=0;
 for(;;){auto q=s.find(',',p);r.emplace_back(s.substr(p,q==std::string::npos?q:q-p));if(q==std::string::npos)break;p=q+1;}return r;
}
static std::uint32_t U(const std::string&s){auto x=std::stoull(s);if(x>UINT32_MAX)throw std::runtime_error("uint32 overflow");return std::uint32_t(x);}
static float F(const std::string&s){float x=std::stof(s);if(!std::isfinite(x))throw std::runtime_error("nonfinite f32");return x;}
static double D(const std::string&s){double x=std::stod(s);if(!std::isfinite(x))throw std::runtime_error("nonfinite f64");return x;}
static std::vector<std::string> read(const std::string&p){std::ifstream f(p);if(!f)throw std::runtime_error("open input failed: "+p);std::vector<std::string> v;std::string s;while(std::getline(f,s))if(!s.empty())v.push_back(s);if(v.size()<2)throw std::runtime_error("empty CSV: "+p);return v;}
static void pairsOut(std::ostream&o,const std::string&s){
 o<<"query_row,bank_body,bank_stable_id,classification,inside_rays,outside_rays,ambiguous,nearest_bone,nearest_triangle,nearest_distance_m,signed_face_distance_m,query_local_x,query_local_y,query_local_z,closest_x,closest_y,closest_z\n";
 std::istringstream in(s);std::string line;std::getline(in,line);while(std::getline(in,line)){auto f=split(line);if(f.size()!=21)throw std::runtime_error("pair row width");o<<f[0];for(size_t i=5;i<f.size();++i)o<<','<<f[i];o<<'\n';}
}
static void nodesOut(std::ostream&o,const std::string&s){
 o<<"query_row,total_bone_banks,aabb_proven_outside_banks,parity_outside_banks,inside_banks,ambiguous_banks,invalid_banks,parity_query_banks\n";
 std::istringstream in(s);std::string line;std::getline(in,line);while(std::getline(in,line)){auto f=split(line);if(f.size()!=12)throw std::runtime_error("summary row width");o<<f[0];for(size_t i=5;i<f.size();++i)o<<','<<f[i];o<<'\n';}
}
int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("usage: human_skin_fat_static_query triangles bodies points pairs_out nodes_out expected_banks");
 auto tl=read(argv[1]),bl=read(argv[2]),pl=read(argv[3]);
 const std::string th="asset_triangle_ordinal,body_index,bone_stable_id,source_record_index,triangle_stable_id,v0x_m,v0y_m,v0z_m,v1x_m,v1y_m,v1z_m,v2x_m,v2y_m,v2z_m,v0x_bits_u32,v0y_bits_u32,v0z_bits_u32,v1x_bits_u32,v1y_bits_u32,v1z_bits_u32,v2x_bits_u32,v2y_bits_u32,v2z_bits_u32";
 if(tl[0]!=th||bl[0].find("body_index,position_x_m,position_y_m,position_z_m,orientation_qx,orientation_qy,orientation_qz,orientation_qw,")!=0||pl[0]!="query_row,x_m,y_m,z_m,x_bits_u32,y_bits_u32,z_bits_u32")throw std::runtime_error("input CSV header mismatch");
 std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> tris;tris.reserve(tl.size()-1);
 for(size_t i=1;i<tl.size();++i){auto f=split(tl[i]);if(f.size()!=23||U(f[0])!=i-1)throw std::runtime_error("triangle row order/width");
  NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU t{};t.bodyIndex=U(f[1]);t.boneStableId=U(f[2]);t.triangleStableId=U(f[4]);
  std::array<float,9> v{};for(size_t k=0;k<9;++k){v[k]=F(f[5+k]);if(std::bit_cast<uint32_t>(v[k])!=U(f[14+k]))throw std::runtime_error("triangle f32 bits mismatch");}
  t.bodyLocalVertex0={v[0],v[1],v[2],0};t.bodyLocalVertex1={v[3],v[4],v[5],0};t.bodyLocalVertex2={v[6],v[7],v[8],0};
  auto n=w::cross(w::sub(w::read3(t.bodyLocalVertex1),w::read3(t.bodyLocalVertex0)),w::sub(w::read3(t.bodyLocalVertex2),w::read3(t.bodyLocalVertex0)));double len=w::norm(n);if(!(len>0))throw std::runtime_error("degenerate triangle");
  n=w::mul(n,1.0/len);t.authoredOutwardNormal={float(n.x),float(n.y),float(n.z),0};tris.push_back(t);
 }
 uint32_t maxBody=0;for(size_t i=1;i<bl.size();++i){auto f=split(bl[i]);if(f.size()!=33)throw std::runtime_error("body row width");maxBody=std::max(maxBody,U(f[0]));}
 std::vector<QBody>bodies(size_t(maxBody)+1);std::vector<std::array<double,3>>offs(bodies.size());std::vector<uint8_t>seen(bodies.size());
 for(size_t i=1;i<bl.size();++i){auto f=split(bl[i]);uint32_t id=U(f[0]);if(id>=bodies.size()||seen[id]++)throw std::runtime_error("body id duplicate");bodies[id].position={F(f[1]),F(f[2]),F(f[3]),0};bodies[id].orientation={F(f[4]),F(f[5]),F(f[6]),F(f[7])};offs[id]={D(f[17]),D(f[18]),D(f[19])};}
 if(std::find(seen.begin(),seen.end(),0)!=seen.end())throw std::runtime_error("body ID range incomplete");
 auto built=bvh::buildStaticCompleteBodyBVH(tris,tris.size(),true);if(!built.succeeded())throw std::runtime_error("closed per-bone BVH build failed code="+std::to_string(unsigned(built.failure)));
 std::vector<QTree>trees;for(auto b:built.bodies)trees.push_back({b.bodyIndex,b.firstNode,b.nodeCount,b.firstTriangle,b.triangleCount});
 std::vector<QNode>nodes;std::vector<uint8_t>roles;std::vector<uint32_t>rows;
 for(size_t i=1;i<pl.size();++i){auto f=split(pl[i]);if((f.size()!=4&&f.size()!=7)||U(f[0])!=i-1)throw std::runtime_error("point row order/width");float x=F(f[1]),y=F(f[2]),z=F(f[3]);if(f.size()==7&&(std::bit_cast<uint32_t>(x)!=U(f[4])||std::bit_cast<uint32_t>(y)!=U(f[5])||std::bit_cast<uint32_t>(z)!=U(f[6])))throw std::runtime_error("point f32 bits mismatch");nodes.push_back({nm_float4{x,y,z,0}});roles.push_back(0);rows.push_back(UINT32_MAX);}
 if(nodes.empty()||nodes.size()>UINT32_MAX)throw std::runtime_error("query point count invalid");
 auto off=[&](uint32_t id){if(id>=offs.size())throw std::runtime_error("body offset missing");return offs[id];};
 std::ostringstream rawP,rawN;auto a=d37::writeAllBankContainment(rawP,nodes,bodies,trees,built.triangles,built.nodes,roles,rows,0,uint32_t(nodes.size()),off,&rawN);
 if(!a.valid||a.banks!=std::stoull(argv[6]))throw std::runtime_error("all-bank audit/inventory failed: "+a.failure);
 std::ofstream po(argv[4],std::ios::binary|std::ios::out|std::ios::trunc),no(argv[5],std::ios::binary|std::ios::out|std::ios::trunc);if(!po||!no)throw std::runtime_error("output open failed");
 pairsOut(po,rawP.str());nodesOut(no,rawN.str());po.close();no.close();if(!po||!no)throw std::runtime_error("output write failed");
 std::cout<<"query_rows="<<a.nodes<<" banks="<<a.banks<<" aabb_outside="<<a.aabbProvenOutsidePairs<<" parity_outside="<<a.parityOutsidePairs<<" inside="<<a.insidePairs<<" ambiguous="<<a.ambiguousPairs<<" invalid="<<a.invalidPairs<<" triangles="<<built.triangles.size()<<" bvh_nodes="<<built.nodes.size()<<"\n";
 return 0;
}catch(const std::exception&e){std::cerr<<"generic query failed: "<<e.what()<<"\n";return 1;}}

