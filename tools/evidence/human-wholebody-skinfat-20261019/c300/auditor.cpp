#include "NumiHumanCoarseTissuePreparation.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
using namespace metalrobo::human::coarse_tissue;
struct Tet { std::array<std::uint32_t,4> id{}; std::uint32_t band=0,face=0,split=0; int rawSign=0,expected=0; bool swapped=false; double rawV=0,posV=0; };
struct Use { unsigned count=0; Face oriented{}; std::uint32_t tet=0; };
static int sgn(double v){return (v>0)-(v<0);}
static int parity(Face f){int p=1;for(int i=0;i<3;i++)for(int j=i+1;j<3;j++)if(f[i]>f[j])p=-p;return p;}
static bool readCandidate(const std::string& p,std::vector<std::uint32_t>& src,std::vector<Face>& f,std::vector<std::uint32_t>& sf){
 std::ifstream in(p);unsigned n=0,nf=0;if(!(in>>n>>nf)||n<4||nf<4)return false;src.resize(n);f.resize(nf);sf.resize(nf);
 for(auto& x:src)if(!(in>>x))return false;for(auto& x:f)for(auto& y:x)if(!(in>>y))return false;for(auto& x:sf)if(!(in>>x))return false;std::string x;return !(in>>x);
}
static bool readLayers(const std::string& p,const std::vector<std::uint32_t>& src,std::vector<V3>& o,std::vector<V3>& s,std::vector<V3>& f,std::vector<V3>& d){
 std::ifstream in(p);std::string line;if(!std::getline(in,line))return false;o.resize(src.size());s.resize(src.size());f.resize(src.size());d.resize(src.size());
 for(std::size_t i=0;i<src.size();i++){if(!std::getline(in,line))return false;std::replace(line.begin(),line.end(),',',' ');std::istringstream r(line);std::uint32_t id=0;double a[20]{};if(!(r>>id)||id!=src[i])return false;for(double& x:a)if(!(r>>x)||!std::isfinite(x))return false;o[i]={a[2],a[3],a[4]};s[i]={a[5],a[6],a[7]};f[i]={a[8],a[9],a[10]};d[i]={a[17],a[18],a[19]};for(V3* q:{&o[i],&s[i],&f[i]})for(double& x:*q)x=static_cast<double>(static_cast<float>(x));}return true;
}
static bool readDonors(const std::string& p,const std::vector<std::uint32_t>& src,std::vector<std::uint32_t>& d){
 std::ifstream in(p);std::string line;if(!std::getline(in,line))return false;std::vector<std::uint32_t> all;
 while(std::getline(in,line)){auto c=line.find(',');if(c==std::string::npos)return false;try{auto id=static_cast<std::uint32_t>(std::stoul(line.substr(0,c)));auto e=line.find(',',c+1);auto donor=static_cast<std::uint32_t>(std::stoul(line.substr(c+1,e-c-1)));if(id!=all.size())return false;all.push_back(donor);}catch(...){return false;}}
 d.resize(src.size());for(std::size_t i=0;i<src.size();i++){if(src[i]>=all.size()||all[src[i]]==NM_INVALID_INDEX)return false;d[i]=all[src[i]];}return true;
}
static bool reverseFace(Face a,Face b){return (a[0]==b[0]&&a[1]==b[2]&&a[2]==b[1])||(a[0]==b[1]&&a[1]==b[0]&&a[2]==b[2])||(a[0]==b[2]&&a[1]==b[1]&&a[2]==b[0]);}
int main(int argc,char** argv){
 if(argc!=8){std::cerr<<"usage: compact_boundary_audit candidate.txt layers.csv registered.csv outdir candidate_sha layers_sha registered_sha\n";return 2;}
 const std::string out=argv[4];std::vector<std::uint32_t> src,sf,donor;std::vector<Face> raw;
 if(!readCandidate(argv[1],src,raw,sf)){std::cerr<<"candidate parse failed\n";return 3;}
 std::vector<V3> outer,skin,fat,dir;if(!readLayers(argv[2],src,outer,skin,fat,dir)){std::cerr<<"CSV parse/order failed\n";return 4;}
 if(!readDonors(argv[3],src,donor)){std::cerr<<"registered donor mapping failed\n";return 5;}
 const std::uint32_t n=static_cast<std::uint32_t>(src.size()),nf=static_cast<std::uint32_t>(raw.size());
 std::vector<V3> p;p.reserve(3*n);p.insert(p.end(),outer.begin(),outer.end());p.insert(p.end(),skin.begin(),skin.end());p.insert(p.end(),fat.begin(),fat.end());
 const std::uint32_t tf[4][3]={{1,2,3},{0,3,2},{0,1,3},{0,2,1}};
 std::vector<Tet> tets;tets.reserve(6ull*nf);std::uint64_t orientBad=0,zero=0,weak=0;double minV=std::numeric_limits<double>::infinity(),sumV=0;
 std::ofstream orient(out+"/raw-face-expected-orientation.csv");
 orient<<"input_face,source_face,raw_a,raw_b,raw_c,sorted_a,sorted_b,sorted_c,raw_to_sorted_parity,raw_normal_dot_outward,band,split,expected_signed_sign,float32_signed_volume,canonical_swap,positive_volume\n";
 std::ofstream tc(out+"/positive-tetrahedra.csv");tc<<"tet,band,input_face,source_face,split,n0,n1,n2,n3,raw_sign,expected_sign,canonical_swap,positive_volume_m3\n";
 const std::array<int,3> pattern{-1,1,-1};
 for(std::uint32_t fi=0;fi<nf;fi++){
  const Face r=raw[fi];Face f=r;std::sort(f.begin(),f.end());if(f[0]>=n||f[1]>=n||f[2]>=n||f[0]==f[1]||f[1]==f[2]||f[2]==f[0])return 6;
  const int par=parity(r);const V3 normal=cross(sub(outer[r[1]],outer[r[0]]),sub(outer[r[2]],outer[r[0]]));const V3 dn=add(add(dir[r[0]],dir[r[1]]),dir[r[2]]);const double rawDot=dot(normal,dn);
  for(std::uint32_t band=0;band<2;band++)for(std::uint32_t sp=0;sp<3;sp++){
   const std::uint32_t base=band? n:0,up=band?2*n:n,a=f[0]+base,b=f[1]+base,c=f[2]+base;
   std::array<std::uint32_t,4> ids{};if(sp==0)ids={a,b,c,up+f[2]};else if(sp==1)ids={a,b,up+f[1],up+f[2]};else ids={a,up+f[0],up+f[1],up+f[2]};
   const double v=tetVolume(p[ids[0]],p[ids[1]],p[ids[2]],p[ids[3]]);const int ex=pattern[sp]*par,got=sgn(v);bool sw=false;double pv=std::abs(v);
   if(!std::isfinite(v)||got==0){zero++;orientBad++;}
   if(got!=ex)orientBad++;
   if(v<0){std::swap(ids[0],ids[1]);sw=true;}
   if(!std::isfinite(pv)||pv<=0)pv=0;if(pv<=1e-15)weak++;minV=std::min(minV,pv);sumV+=pv;
   const auto ti=static_cast<std::uint32_t>(tets.size());tets.push_back({ids,band,fi,sp,got,ex,sw,v,pv});
   orient<<fi<<","<<sf[fi]<<","<<r[0]<<","<<r[1]<<","<<r[2]<<","<<f[0]<<","<<f[1]<<","<<f[2]<<","<<par<<","<<std::setprecision(17)<<rawDot<<","<<band<<","<<sp<<","<<ex<<","<<v<<","<<sw<<","<<pv<<"\n";
   tc<<ti<<","<<band<<","<<fi<<","<<sf[fi]<<","<<sp<<","<<ids[0]<<","<<ids[1]<<","<<ids[2]<<","<<ids[3]<<","<<got<<","<<ex<<","<<sw<<","<<pv<<"\n";
  }
 }
 std::map<Face,Use> uses;bool internalOrientation=true;
 for(std::uint32_t ti=0;ti<tets.size();ti++)for(const auto& f:tf){Face q{tets[ti].id[f[0]],tets[ti].id[f[1]],tets[ti].id[f[2]]},key=q;std::sort(key.begin(),key.end());auto& u=uses[key];if(!u.count){u.oriented=q;u.tet=ti;}else if(u.count==1&&!reverseFace(u.oriented,q))internalOrientation=false;u.count++;}
 std::vector<Face> boundary;std::vector<std::uint32_t> boundaryParent;std::uint64_t nonmanifoldFaces=0;
 std::ofstream bc(out+"/compact-exposed-boundary.csv");bc<<"boundary_face,n0,n1,n2,owner_tet,owner_band,input_face,source_face,owner_split\n";
 for(const auto& [key,u]:uses){if(u.count==1){const auto& t=tets[u.tet];bc<<boundary.size()<<","<<u.oriented[0]<<","<<u.oriented[1]<<","<<u.oriented[2]<<","<<u.tet<<","<<t.band<<","<<t.face<<","<<sf[t.face]<<","<<t.split<<"\n";boundary.push_back(u.oriented);boundaryParent.push_back(sf[t.face]);}else if(u.count!=2)nonmanifoldFaces++;}
 std::map<std::pair<std::uint32_t,std::uint32_t>,std::pair<unsigned,int>> edges;
 for(Face f:boundary)for(unsigned e=0;e<3;e++){auto a=f[e],b=f[(e+1)%3];auto k=std::minmax(a,b);auto& v=edges[{k.first,k.second}];v.first++;v.second+=(a<b)?1:-1;}
 std::uint64_t badEdgeCount=0,badEdgeOrientation=0;std::ofstream ec(out+"/boundary-edge-errors.csv");ec<<"edge_a,edge_b,count,orientation_sum\n";
 for(auto [e,u]:edges)if(u.first!=2||u.second!=0){ec<<e.first<<","<<e.second<<","<<u.first<<","<<u.second<<"\n";if(u.first!=2)badEdgeCount++;if(u.second!=0)badEdgeOrientation++;}
 std::map<std::pair<std::uint32_t,std::uint32_t>,unsigned> sourceEdges;for(auto f:raw)for(int e=0;e<3;e++){auto k=std::minmax(f[e],f[(e+1)%3]);sourceEdges[k]++;}
 std::uint64_t open=0,nonman=0;for(auto [e,c]:sourceEdges){(void)e;if(c==1)open++;if(c>2)nonman++;}
 const std::uint64_t expectBoundary=2ull*nf+4ull*open;
 RegisteredSurface s;s.name="compact exposed boundary of candidate skin-fat volume";s.provider="C296 auditClosedSurface";s.sourceMember="candidate-024-safe + layer-coordinate CSV";s.sourceMeshSHA256=argv[5];s.sourcePayloadSHA256=argv[6];s.aggregateIdentity=true;s.sourceBodyIndex=NM_INVALID_INDEX;
 std::vector<std::uint32_t> compactMap(p.size(),NM_INVALID_INDEX),oldNode;
 for(const auto& f:boundary)for(auto id:f)if(compactMap[id]==NM_INVALID_INDEX){compactMap[id]=static_cast<std::uint32_t>(oldNode.size());oldNode.push_back(id);}
 s.worldPoints.reserve(oldNode.size());s.donorLocalPoints.reserve(oldNode.size());s.sourceVertexIndices.reserve(oldNode.size());s.donorBodyIndices.reserve(oldNode.size());
 for(auto id:oldNode){s.worldPoints.push_back(p[id]);s.donorLocalPoints.push_back(p[id]);const auto local=id%n;s.sourceVertexIndices.push_back(src[local]);s.donorBodyIndices.push_back(donor[local]);}
 s.faces.reserve(boundary.size());for(const auto& f:boundary)s.faces.push_back({compactMap[f[0]],compactMap[f[1]],compactMap[f[2]]});s.sourceFaceIndices=boundaryParent;
 std::ofstream vm(out+"/compact-boundary-vertex-map.csv");vm<<"compact_vertex,volume_node,layer,source_vertex,donor_body,x,y,z\n";
 for(std::uint32_t i=0;i<oldNode.size();i++){const auto id=oldNode[i],layer=id/n,local=id%n;const auto q=p[id];vm<<i<<","<<id<<","<<layer<<","<<src[local]<<","<<donor[local]<<","<<std::setprecision(17)<<q[0]<<","<<q[1]<<","<<q[2]<<"\n";}
 GeometryAudit g{};std::string err;std::vector<std::array<std::uint32_t,2>> crossings;const bool helper=auditClosedSurface(s,g,err,false,&crossings);
 std::ofstream xc(out+"/boundary-self-crossings.csv");xc<<"face_a,face_b,source_face_a,source_face_b\n";for(auto x:crossings)xc<<x[0]<<","<<x[1]<<","<<boundaryParent[x[0]]<<","<<boundaryParent[x[1]]<<"\n";
 std::uint64_t rimSidewallFaces=0;for(const auto& f:boundary){unsigned lo=3,hi=0;for(auto id:f){const unsigned layer=id/n;lo=std::min(lo,layer);hi=std::max(hi,layer);}if(lo!=hi)rimSidewallFaces++;}
 const bool countOK=boundary.size()==expectBoundary;
 const bool pass=orientBad==0&&zero==0&&weak==0&&nonman==0&&nonmanifoldFaces==0&&internalOrientation&&badEdgeCount==0&&badEdgeOrientation==0&&countOK&&rimSidewallFaces==4*open&&helper&&g.boundaryEdgeCount==0&&crossings.empty()&&g.volumeM3>0;
 std::ofstream j(out+"/result.json");j<<std::setprecision(17)<<"{\n"
 <<"\"scope\":\"offline compact exposed-boundary audit of exact two-band three-tet prism connectivity from C296 preparePairedBoundaryFEM; no geometry repair or production qualification\",\n"
 <<"\"candidate_txt_sha256\":\""<<argv[5]<<"\",\"layers_csv_sha256\":\""<<argv[6]<<"\",\"registered_csv_sha256\":\""<<argv[7]<<"\",\n"
 <<"\"vertices_per_layer\":"<<n<<",\"source_faces\":"<<nf<<",\"bands\":2,\"tetrahedra\":"<<tets.size()<<",\n"
 <<"\"raw_face_orientation\":{\"expected_sign_pattern\":\"[-parity,+parity,-parity] per band; parity is raw input face to sorted vertex order\",\"tet_mismatch_count\":"<<orientBad<<",\"zero_or_nonfinite_tets\":"<<zero<<",\"weak_abs_volume_le_1e-15\":"<<weak<<",\"min_positive_tet_volume_m3\":"<<minV<<",\"sum_positive_tet_volume_m3\":"<<sumV<<"},\n"
 <<"\"source_edges\":{\"open\":"<<open<<",\"nonmanifold\":"<<nonman<<"},\n"
 <<"\"compact_boundary\":{\"audit_surface_vertices\":"<<oldNode.size()<<",\"full_volume_nodes\":"<<p.size()<<",\"volume_nodes_not_on_exposed_boundary\":"<<(p.size()-oldNode.size())<<",\"rim_sidewall_faces\":"<<rimSidewallFaces<<",\"expected_rim_sidewall_faces\":"<<(4*open)<<",\"count_one_faces\":"<<boundary.size()<<",\"expected_2nf_plus_4_open_edges\":"<<expectBoundary<<",\"count_matches\":"<<(countOK?"true":"false")<<",\"nonmanifold_tet_faces\":"<<nonmanifoldFaces<<",\"internal_face_orientation_ok\":"<<(internalOrientation?"true":"false")<<",\"bad_boundary_edges\":"<<badEdgeCount<<",\"boundary_edge_orientation_errors\":"<<badEdgeOrientation<<",\"helper_ok\":"<<(helper?"true":"false")<<",\"helper_error\":\""<<err<<"\",\"helper_boundary_edges\":"<<g.boundaryEdgeCount<<",\"helper_boundary_loops\":"<<g.boundaryLoopCount<<",\"self_crossings\":"<<crossings.size()<<",\"signed_volume_m3\":"<<g.volumeM3<<"},\n"
 <<"\"overall_ok\":"<<(pass?"true":"false")<<"\n}\n";
 std::cout<<"overall="<<(pass?"PASS":"FAIL")<<" n="<<n<<" faces="<<nf<<" tets="<<tets.size()<<" orient_bad="<<orientBad<<" open="<<open<<" boundary="<<boundary.size()<<"/"<<expectBoundary<<" edge_bad="<<badEdgeCount<<","<<badEdgeOrientation<<" selfcross="<<crossings.size()<<" helper="<<helper<<" error="<<err<<"\n";
 return pass?0:1;
}
