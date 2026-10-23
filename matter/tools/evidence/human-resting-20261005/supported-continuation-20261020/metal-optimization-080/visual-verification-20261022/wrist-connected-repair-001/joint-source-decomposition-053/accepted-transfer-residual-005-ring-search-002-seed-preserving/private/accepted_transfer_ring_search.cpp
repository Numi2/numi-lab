#include "NumiHumanAcceptedFEMSkinTransfer.hpp"
#include "NumiHumanElbowBaselineProlongation065.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using T=numi::human::AcceptedFEMSkinTransfer; using P=T::Point;
struct Input { std::vector<P> detailRest,detailCurrent,coarseRest; std::vector<uint32_t> detailSource,detailDonor,coarseSource,coarseDonor; std::vector<uint8_t> used; std::vector<T::Face> faces,sourceFaces; std::vector<std::uint32_t> stableFaceIds; std::vector<P> bary; std::vector<T::Pose> restPose,currentPose; std::vector<uint32_t> representatives; };
static void need(bool x,const std::string& s){if(!x)throw std::runtime_error(s);}
static Input readInput(const std::string& d){ Input a; std::ifstream f; size_t n;
 f.open(d+"/detail.txt");need(bool(f),"detail open "+d);f>>n;a.detailRest.resize(n);a.detailCurrent.resize(n);a.detailSource.resize(n);a.detailDonor.resize(n);a.used.resize(n);
 for(size_t i=0;i<n;i++){uint32_t id,donor;unsigned used;f>>id>>donor>>used;need(id==i,"detail id order");a.detailSource[i]=id;a.detailDonor[i]=donor;a.used[i]=uint8_t(used);for(double&v:a.detailRest[i])f>>v;for(double&v:a.detailCurrent[i])f>>v;}f.close();
 f.open(d+"/coarse.txt");need(bool(f),"coarse open");f>>n;a.coarseRest.resize(n);a.coarseSource.resize(n);a.coarseDonor.resize(n);a.sourceFaces.resize(n);a.bary.resize(n);
 for(size_t i=0;i<n;i++){f>>a.coarseSource[i]>>a.coarseDonor[i]>>a.sourceFaces[i][0];for(double&v:a.bary[i])f>>v;for(double&v:a.coarseRest[i])f>>v;}f.close();
 f.open(d+"/faces.txt");need(bool(f),"faces open");f>>n;a.faces.resize(n);for(auto&v:a.faces)f>>v[0]>>v[1]>>v[2];f.close();
 const std::string stablePath="/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/elbow-refinement-current-main-017-single-residual-prolongation-001/generated/faces.csv";
 std::ifstream stable(stablePath);need(bool(stable),"stable face identity CSV open");std::string line;std::getline(stable,line);a.stableFaceIds.reserve(a.faces.size());
 for(size_t i=0;i<a.faces.size();++i){need(bool(std::getline(stable,line)),"stable face CSV truncated");std::array<unsigned long long,6>x{};size_t pos=0;for(int k=0;k<6;k++){size_t next=line.find(',',pos);std::string field=line.substr(pos,next==std::string::npos?next:next-pos);x[k]=std::stoull(field);pos=next==std::string::npos?line.size():next+1;}need(x[0]==i&&x[1]==a.faces[i][0]&&x[2]==a.faces[i][1]&&x[3]==a.faces[i][2],"stable face CSV topology/order mismatch");a.stableFaceIds.push_back(std::uint32_t(x[5]));}
 std::getline(stable,line);need(line.empty(),"stable face CSV has excess rows");stable.close();
 std::ifstream sf("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/joint-source-decomposition-053/accepted-transfer-residual-002/source_faces.txt");size_t sn;sf>>sn;std::vector<T::Face> sourceMesh(sn);for(auto&v:sourceMesh)sf>>v[0]>>v[1]>>v[2];sf.close();for(auto& face:a.sourceFaces){const auto fi=face[0];need(fi<sourceMesh.size(),"coarse source face index");face=sourceMesh[fi];}


 f.open(d+"/poses.txt");need(bool(f),"poses open");f>>n;a.restPose.resize(n);a.currentPose.resize(n);for(size_t i=0;i<n;i++){size_t id;f>>id;need(id==i,"pose order");for(double&v:a.restPose[i].translation)f>>v;for(double&v:a.restPose[i].rotation)f>>v;for(double&v:a.currentPose[i].translation)f>>v;for(double&v:a.currentPose[i].rotation)f>>v;}f.close();
 f.open(d+"/representatives.txt");need(bool(f),"representatives open");f>>n;a.representatives.resize(n);for(auto&v:a.representatives)f>>v;need(a.representatives.size()==a.detailRest.size(),"representative count");return a; }
static double norm(const P& p){return std::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);}
static std::vector<P> mapped(const std::vector<T::GPUTransferRow>& rows,const std::vector<P>& x){need(rows.size()>0,"empty transfer rows");std::vector<P> y(rows.size());for(size_t i=0;i<rows.size();i++)for(int k=0;k<3;k++){auto v=rows[i].coarseVertex[k];need(v<x.size(),"transfer coarse bounds");for(int c=0;c<3;c++)y[i][c]+=double(rows[i].weight[k])*x[v][c];}return y;}
int main(int argc,char**argv){try{
 need(argc==4,"usage: executable INPUT_ROOT OUTPUT_DIR FACE_RINGS_1_TO_3");
 std::string root=argv[1],out=argv[2];
 const auto rings=std::stoul(argv[3]);need(rings>=1&&rings<=3,"face rings must be 1..3");
 auto prolong=numi::human::makeElbowPhysicalBaselineProlongation065(2411);
 Input ref=readInput(root+"/reference");T transfer;T::BuildReport report;std::string why;
 need(transfer.build(ref.detailRest,ref.detailSource,ref.detailDonor,ref.used,
     ref.coarseRest,ref.coarseSource,ref.coarseDonor,ref.faces,&report,&why,
     std::numeric_limits<double>::infinity(),&ref.representatives,
     &ref.sourceFaces,&ref.bary,&prolong,rings,&ref.stableFaceIds),why);
 std::ofstream map(out+"/selected_transfer_map.csv");need(bool(map),"map open");
 map<<"vertex,donor,used,representative,coarse_face,stable_source_face,projection_distance_m,coarse0,coarse1,coarse2,w0,w1,w2\n"<<std::setprecision(17);
 for(size_t i=0;i<transfer.rows().size();++i){
   const auto& tr=transfer.rows()[i];auto fi=transfer.selectedFaceIndex(i);
   map<<i<<","<<ref.detailDonor[i]<<","<<unsigned(ref.used[i])<<","<<ref.representatives[i]<<","<<fi<<","
      <<(fi<ref.stableFaceIds.size()?ref.stableFaceIds[fi]:std::numeric_limits<std::uint32_t>::max())<<","<<tr.restDistance<<","
      <<tr.coarseVertex[0]<<","<<tr.coarseVertex[1]<<","<<tr.coarseVertex[2]<<","
      <<tr.weight[0]<<","<<tr.weight[1]<<","<<tr.weight[2]<<"\n";
 }
 map.close();
 std::ofstream residual(out+"/source_lbs_minus_transfer_baseline.csv");need(bool(residual),"residual open");
 residual<<"case,vertex,donor,used,projection_distance_m,source_x_m,source_y_m,source_z_m,baseline_x_m,baseline_y_m,baseline_z_m,residual_x_m,residual_y_m,residual_z_m,residual_norm_m\n"<<std::setprecision(17);
 for(const std::string c:{"reference","shoulder-elevation-050","elbow-flexion-080"}){
   Input now=readInput(root+"/"+c);
   need(now.detailRest.size()==ref.detailRest.size()&&now.coarseRest.size()==ref.coarseRest.size(),"case dimensions changed");
   std::vector<P> bc;
   need(transfer.makeCurrentCoarseBaseline(ref.detailRest,now.detailCurrent,
       ref.restPose,now.currentPose,bc,&why),why);
   auto baselineDetail=mapped(transfer.rows(),bc);
   double maximum=0.0, sum2=0.0;std::size_t count=0;
   for(size_t i=0;i<ref.detailRest.size();++i){
     if(!now.used[i])continue;
     P r{};for(int j=0;j<3;++j)r[j]=now.detailCurrent[i][j]-baselineDetail[i][j];
     const double rn=norm(r);maximum=std::max(maximum,rn);sum2+=rn*rn;++count;
     const auto& tr=transfer.rows()[i];
     residual<<c<<","<<i<<","<<now.detailDonor[i]<<","<<unsigned(now.used[i])<<","<<tr.restDistance;
     for(double v:now.detailCurrent[i])residual<<","<<v;
     for(double v:baselineDetail[i])residual<<","<<v;
     for(double v:r)residual<<","<<v;
     residual<<","<<rn<<"\n";
   }
   std::cout<<"case="<<c<<" used_rows="<<count<<" actual_source_lbs_minus_Bbaseline_max_m="<<maximum
      <<" rms_m="<<(count?std::sqrt(sum2/count):0)<<"\n";
 }
 residual.close();std::cout<<"PASS actual_owner=AcceptedFEMSkinTransfer source_registration_only=1 face_rings="<<rings
      <<" exact_rows="<<report.exactAncestryRows<<" projected_rows="<<report.projectedRows
      <<" ancestry_fallback_rows="<<report.ancestryFallbackRows
      <<" max_projection_m="<<report.maximumProjectionDistance
      <<" max_fallback_m="<<report.maximumAncestryFallbackDistance
      <<" output="<<out<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 2;}}
