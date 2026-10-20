#include "NumiHumanCoarseTissuePreparation.hpp"
#include "NumiHumanWholeBodySkinReduction.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using namespace metalrobo::human::coarse_tissue;

struct CrossSurfaceAuditV1 {
 std::uint64_t testedAABBPairs=0;
 std::vector<std::array<std::uint32_t,2>> crossings;
};
static bool auditCrossSurfaceF32(const std::vector<V3>& aPoints,const std::vector<Face>& aFaces,
 const std::vector<V3>& bPoints,const std::vector<Face>& bFaces,CrossSurfaceAuditV1& out,std::string& error){
 out={};
 if(aPoints.size()<3||bPoints.size()<3||aFaces.empty()||bFaces.empty()){error="cross-surface input is empty";return false;}
 auto f32=[](const std::vector<V3>& src){std::vector<V3> dst=src;for(auto& p:dst)for(double& x:p)x=static_cast<double>(static_cast<float>(x));return dst;};
 const auto ap=f32(aPoints),bp=f32(bPoints);
 V3 lo=ap[0],hi=lo;for(const auto& points:{&ap,&bp})for(const auto& p:*points)for(int k=0;k<3;k++){lo[k]=std::min(lo[k],p[k]);hi[k]=std::max(hi[k],p[k]);}
 const double scale=norm(sub(hi,lo));if(!std::isfinite(scale)||scale<=1e-8){error="cross-surface combined extent is invalid";return false;}
 const double eps=std::max(1e-12,scale*1e-10);
 struct Tri {std::array<V3,3> p;Face ids;V3 lo,hi;};
 auto build=[&](const std::vector<V3>& pts,const std::vector<Face>& faces,std::uint32_t offset,std::vector<Tri>& outTri){
  outTri.clear();outTri.reserve(faces.size());for(const auto& f:faces){if(f[0]>=pts.size()||f[1]>=pts.size()||f[2]>=pts.size()){error="cross-surface face index invalid";return false;}
   Tri t{};t.ids={f[0]+offset,f[1]+offset,f[2]+offset};for(int i=0;i<3;i++){t.p[i]=pts[f[i]];for(int k=0;k<3;k++){if(i==0)t.lo[k]=t.hi[k]=t.p[i][k];else{t.lo[k]=std::min(t.lo[k],t.p[i][k]);t.hi[k]=std::max(t.hi[k],t.p[i][k]);}}}outTri.push_back(t);}return true;
 };
 std::vector<Tri> at,bt;if(!build(ap,aFaces,0,at)||!build(bp,bFaces,static_cast<std::uint32_t>(ap.size()),bt))return false;
 std::vector<std::uint32_t> order(bt.size());for(std::uint32_t i=0;i<bt.size();i++)order[i]=i;
 std::sort(order.begin(),order.end(),[&](auto x,auto y){return bt[x].lo[0]<bt[y].lo[0];});
 double maxBWidth=0;for(const auto& t:bt)maxBWidth=std::max(maxBWidth,t.hi[0]-t.lo[0]);
 std::vector<std::uint32_t> orderA(at.size());for(std::uint32_t i=0;i<at.size();i++)orderA[i]=i;
 std::sort(orderA.begin(),orderA.end(),[&](auto x,auto y){return at[x].lo[0]<at[y].lo[0];});
 auto segTri=[&](V3 p,V3 q,const std::array<V3,3>& t){
  const V3 d=sub(q,p),e1=sub(t[1],t[0]),e2=sub(t[2],t[0]),h=cross(d,e2);double det=dot(e1,h);
  if(std::abs(det)<=eps*scale*scale)return false;double inv=1.0/det;V3 z=sub(p,t[0]);double u=dot(z,h)*inv;if(u<-1e-10||u>1+1e-10)return false;
  V3 r=cross(z,e1);double v=dot(d,r)*inv;if(v<-1e-10||u+v>1+1e-10)return false;double tval=dot(e2,r)*inv;
  return tval>1e-10&&tval<1-1e-10;
 };
 for(auto ai:orderA){const auto& a=at[ai];const double low=a.lo[0]-maxBWidth-eps,high=a.hi[0]+eps;
  auto begin=std::lower_bound(order.begin(),order.end(),low,[&](std::uint32_t bi,double x){return bt[bi].lo[0]<x;});
  for(auto it=begin;it!=order.end()&&bt[*it].lo[0]<=high;it++){const auto bi=*it;const auto& b=bt[bi];
   if(a.hi[1]+eps<b.lo[1]||b.hi[1]+eps<a.lo[1]||a.hi[2]+eps<b.lo[2]||b.hi[2]+eps<a.lo[2])continue;
   out.testedAABBPairs++;bool hit=false;
   for(int e=0;e<3;e++)hit=hit||segTri(a.p[e],a.p[(e+1)%3],b.p)||segTri(b.p[e],b.p[(e+1)%3],a.p);
   if(!hit)hit=coplanarTriangleOverlap(a.p,a.ids,b.p,b.ids,scale,std::max(eps*10.0,scale*1e-8));
   if(hit)out.crossings.push_back({ai,bi});
  }
 }
 return true;
}

struct F32TetAuditV1 {std::uint64_t count=0,nonpositive=0,degenerate=0;double minPositiveVolume=std::numeric_limits<double>::infinity(),sumSignedVolume=0,referenceVolume=0;};
static bool auditFloat32TetsV1(const std::vector<V3>& outer,const std::vector<V3>& inner,const std::vector<Face>& faces,double referenceVolume,F32TetAuditV1& out){
 out={};out.referenceVolume=referenceVolume;const double tol=std::max(1e-30,referenceVolume*1e-15);
 auto p32=[](V3 p){for(double& x:p)x=static_cast<double>(static_cast<float>(x));return p;};
 for(const auto& raw:faces){Face f=raw;std::sort(f.begin(),f.end());const std::uint32_t n=static_cast<std::uint32_t>(outer.size());
  std::array<std::array<std::uint32_t,4>,3> tets{{{{f[0],f[1],f[2],n+f[2]}},{{f[0],f[1],n+f[1],n+f[2]}},{{f[0],n+f[0],n+f[1],n+f[2]}}}};
  auto get=[&](std::uint32_t id,bool rounded){V3 q=id<n?outer[id]:inner[id-n];return rounded?p32(q):q;};
  for(auto ids:tets){const double vd=tetVolume(get(ids[0],false),get(ids[1],false),get(ids[2],false),get(ids[3],false));if(!std::isfinite(vd)||vd==0)return false;if(vd<0)std::swap(ids[0],ids[1]);
   const double vf=tetVolume(get(ids[0],true),get(ids[1],true),get(ids[2],true),get(ids[3],true));out.count++;
   if(!std::isfinite(vf))return false;out.sumSignedVolume+=vf;if(vf<=0)out.nonpositive++;if(vf>0)out.minPositiveVolume=std::min(out.minPositiveVolume,vf);if(vf<=tol)out.degenerate++;
  }
 }
 return true;
}
int main(int argc,char** argv){
 if(argc!=4){std::cerr<<"usage: audit candidate.txt layers.csv result.json\n";return 2;}
 std::ifstream red(argv[1]);unsigned n=0,nf=0;if(!(red>>n>>nf)||!n||!nf)return 3;
 std::vector<std::uint32_t> source(n),sourceFace(nf);std::vector<Face> faces(nf);
 for(auto& x:source)if(!(red>>x))return 4;
 for(auto& f:faces)for(auto& x:f)if(!(red>>x))return 5;
 for(auto& x:sourceFace)if(!(red>>x))return 6;
 std::ifstream csv(argv[2]);std::string line;if(!std::getline(csv,line))return 7;
 std::vector<V3> outer(n),skin(n),fat(n),directions(n);std::vector<std::uint32_t> donor(n);
 std::vector<double> sd(n),fd(n);std::vector<V3> sl(n),fl(n);
 for(std::size_t i=0;i<n;i++){
  if(!std::getline(csv,line))return 8;
  std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
  std::uint32_t sid=0;double a[20]{};
  if(!(row>>sid))return 9;for(double& x:a)if(!(row>>x))return 10;
  if(sid!=source[i]){std::cerr<<"source-order mismatch at "<<i<<"\n";return 11;}
  sd[i]=a[0];fd[i]=a[1];
  outer[i]={a[2],a[3],a[4]};skin[i]={a[5],a[6],a[7]};fat[i]={a[8],a[9],a[10]};
  sl[i]={a[11],a[12],a[13]};fl[i]={a[14],a[15],a[16]};directions[i]={a[17],a[18],a[19]};
  for(auto* p:{&outer[i],&skin[i],&fat[i]})for(double& x:*p)x=static_cast<double>(static_cast<float>(x));
  donor[i]=1;
 }
 auto surface=[&](const char* name,const std::vector<V3>& p,const std::vector<V3>& local){
  RegisteredSurface s;s.name=name;s.provider="source-derived mechanical surface";s.sourceMember="bounded generated layer coordinates";
  s.sourceMeshSHA256=std::string(64,'1');s.sourcePayloadSHA256=std::string(64,'2');s.aggregateIdentity=true;
  s.worldPoints=p;s.donorLocalPoints=local;s.faces=faces;s.sourceVertexIndices=source;s.sourceFaceIndices=sourceFace;s.donorBodyIndices=donor;return s;
 };
 auto outS=surface("outer",outer,outer),skinS=surface("skin-inner",skin,sl),fatS=surface("fat-inner",fat,fl);
 GeometryAudit ao{},as{},af{};std::string eo,es,ef;std::vector<std::array<std::uint32_t,2>> po,ps,pf;
 bool so=auditClosedSurface(outS,ao,eo,true,&po),ss=auditClosedSurface(skinS,as,es,true,&ps),sf=auditClosedSurface(fatS,af,ef,true,&pf);
 std::string pairPath(argv[3]);auto slash=pairPath.find_last_of('/');if(slash!=std::string::npos)pairPath.resize(slash+1);else pairPath.clear();std::ofstream pairFile(pairPath+"fat-self-crossings.csv");pairFile<<"face_a,face_b,source_face_a,source_face_b,source_vertices_a,source_vertices_b\n";for(auto p:pf){pairFile<<p[0]<<","<<p[1]<<","<<sourceFace[p[0]]<<","<<sourceFace[p[1]]<<",\"";for(unsigned k=0;k<3;k++){if(k)pairFile<<";";pairFile<<source[faces[p[0]][k]];}pairFile<<"\",\"";for(unsigned k=0;k<3;k++){if(k)pairFile<<";";pairFile<<source[faces[p[1]][k]];}pairFile<<"\"\n";}
 CrossSurfaceAuditV1 os{},sfX{},of{};std::string eos,esfx,eof;
 bool cos=auditCrossSurfaceF32(outer,faces,skin,faces,os,eos);
 bool csf=auditCrossSurfaceF32(skin,faces,fat,faces,sfX,esfx);
 bool cof=auditCrossSurfaceF32(outer,faces,fat,faces,of,eof);
 auto prismRef=[&](const std::vector<V3>& a,const std::vector<V3>& b){
  double total=0;
  for(auto f:faces){std::sort(f.begin(),f.end());const auto x=f[0],y=f[1],z=f[2];
   total+=std::abs(tetVolume(a[x],a[y],a[z],b[z]));
   total+=std::abs(tetVolume(a[x],a[y],b[y],b[z]));
   total+=std::abs(tetVolume(a[x],b[x],b[y],b[z]));
  }return total;
 };
 const double rvSkin=prismRef(outer,skin),rvFat=prismRef(skin,fat);
 F32TetAuditV1 ts{},tf{};bool tskin=auditFloat32TetsV1(outer,skin,faces,rvSkin,ts),tfat=auditFloat32TetsV1(skin,fat,faces,rvFat,tf);
 auto orientedGate=[&](const std::vector<V3>& a,const std::vector<V3>& b,const std::vector<double>& thickness,double total,std::uint64_t& bad,std::uint64_t& weak,double& minimum){
  bad=0;weak=0;minimum=std::numeric_limits<double>::infinity();
  for(auto f:faces){std::sort(f.begin(),f.end());auto x=f[0],y=f[1],z=f[2];
   const double tol=std::pow(std::max(1e-8,std::max({thickness[x],thickness[y],thickness[z]})),3)*1e-12;
   const double v0=tetVolume(a[x],a[y],a[z],b[z]),v1=tetVolume(a[x],a[y],b[y],b[z]),v2=tetVolume(a[x],b[x],b[y],b[z]);
   const V3 sourceFaceNormal=cross(sub(outer[y],outer[x]),sub(outer[z],outer[x]));
   const int expected=dot(sourceFaceNormal,mul(directions[z],-1.0))>0.0?1:-1;
   if(!std::isfinite(v0)||!std::isfinite(v1)||!std::isfinite(v2)||std::abs(v0)<=tol||std::abs(v1)<=tol||std::abs(v2)<=tol||((v0>0)!=(expected>0))||((v1>0)!=(expected<0))||((v2>0)!=(expected>0)))bad++;
   for(double v:{v0,v1,v2}){const double emitted=std::abs(v);minimum=std::min(minimum,emitted);if(!std::isfinite(emitted)||emitted<=total*1e-15)weak++;}
  }return bad==0&&weak==0;
 };
 std::uint64_t badSkinSign=0,badFatSign=0,weakSkin=0,weakFat=0;double minGateSkin=0,minGateFat=0;
 const bool gateSkin=orientedGate(outer,skin,sd,rvSkin,badSkinSign,weakSkin,minGateSkin);
 const bool gateFat=orientedGate(skin,fat,fd,rvFat,badFatSign,weakFat,minGateFat);
 bool pass=so&&ss&&sf&&po.empty()&&ps.empty()&&pf.empty()&&cos&&csf&&cof&&os.crossings.empty()&&sfX.crossings.empty()&&of.crossings.empty()&&gateSkin&&gateFat&&tskin&&tfat&&ts.nonpositive==0&&tf.nonpositive==0&&ts.degenerate==0&&tf.degenerate==0;
 std::ofstream o(argv[3]);if(!o)return 12;o<<std::setprecision(17);
 o<<"{\n  \"vertices\": "<<n<<", \"faces\": "<<nf<<",\n";
 o<<"  \"float32_coordinates\": true,\n";
 o<<"  \"surface_self_audit\": {\"outer_ok\": "<<so<<", \"skin_ok\": "<<ss<<", \"fat_ok\": "<<sf<<", \"outer_crossings\": "<<po.size()<<", \"skin_crossings\": "<<ps.size()<<", \"fat_crossings\": "<<pf.size()<<", \"boundary_edges\": ["<<ao.boundaryEdgeCount<<","<<as.boundaryEdgeCount<<","<<af.boundaryEdgeCount<<"], \"boundary_loops\": ["<<ao.boundaryLoopCount<<","<<as.boundaryLoopCount<<","<<af.boundaryLoopCount<<"], \"errors\": [\""<<eo<<"\",\""<<es<<"\",\""<<ef<<"\"]},\n";
 o<<"  \"surface_cross_audit\": {\"outer_skin_ok\": "<<cos<<", \"outer_skin_pairs\": "<<os.crossings.size()<<", \"outer_skin_tested\": "<<os.testedAABBPairs<<", \"skin_fat_ok\": "<<csf<<", \"skin_fat_pairs\": "<<sfX.crossings.size()<<", \"skin_fat_tested\": "<<sfX.testedAABBPairs<<", \"outer_fat_ok\": "<<cof<<", \"outer_fat_pairs\": "<<of.crossings.size()<<", \"outer_fat_tested\": "<<of.testedAABBPairs<<", \"errors\": [\""<<eos<<"\",\""<<esfx<<"\",\""<<eof<<"\"]},\n";
 o<<"  \"sorted_prism_tets\": {\"skin_ok\": "<<tskin<<", \"fat_ok\": "<<tfat<<", \"paired_volume_gate_skin_ok\": "<<gateSkin<<", \"paired_volume_gate_fat_ok\": "<<gateFat<<", \"bad_source_orientation_skin\": "<<badSkinSign<<", \"bad_source_orientation_fat\": "<<badFatSign<<", \"weak_emitted_tets_skin\": "<<weakSkin<<", \"weak_emitted_tets_fat\": "<<weakFat<<", \"gate_min_skin_tet_m3\": "<<minGateSkin<<", \"gate_min_fat_tet_m3\": "<<minGateFat<<", \"skin_count\": "<<ts.count<<", \"fat_count\": "<<tf.count<<", \"skin_nonpositive\": "<<ts.nonpositive<<", \"fat_nonpositive\": "<<tf.nonpositive<<", \"skin_degenerate\": "<<ts.degenerate<<", \"fat_degenerate\": "<<tf.degenerate<<", \"skin_min_volume_m3\": "<<ts.minPositiveVolume<<", \"fat_min_volume_m3\": "<<tf.minPositiveVolume<<", \"skin_sum_m3\": "<<ts.sumSignedVolume<<", \"fat_sum_m3\": "<<tf.sumSignedVolume<<", \"skin_reference_m3\": "<<rvSkin<<", \"fat_reference_m3\": "<<rvFat<<"},\n";
 o<<"  \"overall_float32_geometry_ok\": "<<pass<<"\n}\n";
 std::cout<<"pass="<<pass<<" self="<<po.size()<<","<<ps.size()<<","<<pf.size()<<" cross="<<os.crossings.size()<<","<<sfX.crossings.size()<<","<<of.crossings.size()<<" tets="<<ts.count<<","<<tf.count<<" bad="<<ts.nonpositive<<","<<tf.nonpositive<<" degenerate="<<ts.degenerate<<","<<tf.degenerate<<" minvol="<<ts.minPositiveVolume<<","<<tf.minPositiveVolume<<"\n";
 return pass?0:1;
}
