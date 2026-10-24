#include "ConnectedSkinTransferOwner015.hpp"
#include "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/wholebody-attachment-source-rebind-033-final091-regenerated-deltas-001/source/apps/NumiHumanAcceptedFEMSkinTransfer.hpp"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
using namespace metalrobo::human::connected_skin_transfer015;
using Owner=numi::human::AcceptedFEMSkinTransfer;
struct Candidate {
 std::size_t sourceVertexCount=0;
 std::vector<Point> outerWorld;
 std::vector<Face> faces,sourceFaces;
 std::vector<std::int64_t> sourceVertexIds;
 std::vector<std::uint32_t> sourceFaceIds;
 std::vector<Point> sourceBarycentric,sourceResidualWorldM;
 std::vector<std::uint8_t> vertexKinds;
 std::vector<float> sourceWeights86;
 std::vector<std::int64_t> sourceFaceStableIds;
 std::string sourceMeshSHA256=std::string(64,'a'),sourcePayloadSHA256=std::string(64,'b');
};
template<class T> bool readVector(std::ifstream& in,std::vector<T>& v,std::size_t n){v.resize(n);return in.read(reinterpret_cast<char*>(v.data()),std::streamsize(sizeof(T)*n)).good();}
template<class T> bool writeVector(std::ofstream& out,const std::vector<T>& v){return out.write(reinterpret_cast<const char*>(v.data()),std::streamsize(sizeof(T)*v.size())).good();}
int main(int argc,char**argv){
 if(argc!=5){std::cerr<<"usage: runner input.bin output.bin maxProjectionM representatives.bin\n";return 2;}
 const double maxDistance=std::stod(argv[3]);std::ifstream in(argv[1],std::ios::binary);if(!in){std::cerr<<"input open failed\n";return 2;}
 std::array<std::uint64_t,4> h{};if(!in.read(reinterpret_cast<char*>(h.data()),sizeof(h))){std::cerr<<"header read failed\n";return 2;}
 const auto ns=std::size_t(h[0]),nf=std::size_t(h[1]),no=std::size_t(h[2]),nof=std::size_t(h[3]);
 if(ns<4||nf<4||no<4||nof<4||ns>1000000||nf>6000000||no>2000000||nof>6000000){std::cerr<<"invalid input dimensions\n";return 2;}
 std::vector<float> sx(ns*3), weights(ns*86), ox(no*3);std::vector<std::uint32_t> sf(nf*3),bind(86),of(nof*3);
 if(!readVector(in,sx,ns*3)||!readVector(in,sf,nf*3)||!readVector(in,weights,ns*86)||!readVector(in,bind,86)||!readVector(in,ox,no*3)||!readVector(in,of,nof*3)){std::cerr<<"truncated array payload\n";return 2;}
 if(in.peek()!=std::char_traits<char>::eof()){std::cerr<<"unexpected trailing input bytes\n";return 2;}
 std::vector<Point> source(ns);for(std::size_t i=0;i<ns;++i)source[i]={sx[3*i],sx[3*i+1],sx[3*i+2]};
 std::vector<double> wdouble(weights.begin(),weights.end());std::vector<std::uint32_t> sourceDonor(ns);
 for(std::size_t i=0;i<ns;++i){std::size_t best=0;for(std::size_t b=1;b<86;++b)if(weights[i*86+b]>weights[i*86+best])best=b;sourceDonor[i]=bind[best];}
 Candidate c;c.sourceVertexCount=ns;c.sourceFaces.resize(nf);for(std::size_t f=0;f<nf;++f)c.sourceFaces[f]={sf[3*f],sf[3*f+1],sf[3*f+2]};
 c.outerWorld.resize(no);for(std::size_t i=0;i<no;++i)c.outerWorld[i]={ox[3*i],ox[3*i+1],ox[3*i+2]};
 c.faces.resize(nof);for(std::size_t f=0;f<nof;++f)c.faces[f]={of[3*f],of[3*f+1],of[3*f+2]};
 c.sourceVertexIds.assign(no,-1);c.sourceFaceIds.assign(no,std::numeric_limits<std::uint32_t>::max());c.sourceBarycentric.resize(no);c.sourceResidualWorldM.resize(no);c.vertexKinds.assign(no,1);c.sourceWeights86=weights;
 std::vector<std::uint32_t> physicalDonor;ProjectionMap probeMap;std::string error;Owner probe;
 auto resolver=[&](std::span<const double,86> row,const Point&,std::uint32_t& body,std::string& why){std::size_t best=0;for(std::size_t b=1;b<86;++b)if(row[b]>row[best])best=b;body=bind[best];why.clear();return true;};
 if(!inferPhysicalDonorBodies(probe,c,source,wdouble,bind,5.0,resolver,physicalDonor,probeMap,error)){std::cerr<<"donor probe failed: "<<error<<"\n";return 3;}
 ProjectionMap projection;Owner reverse;if(!buildReverseProjectionOwner(reverse,c,source,sourceDonor,physicalDonor,maxDistance,{},projection,error)){std::cerr<<"donor-filtered reverse projection failed: "<<error<<"\n";return 4;}
 Report report;std::vector<std::uint32_t> rows;std::vector<Point> bary,residual;if(!buildPhysicalSourceEmbedding(c,source,&reverse,&projection,2e-12,rows,bary,residual,report,error)){std::cerr<<"embedding closure failed: "<<error<<"\n";return 5;}
 // Existing accepted-transfer direction: active detail source rows map to the O surface.
 std::vector<std::uint8_t> usedSource(ns,0u);for(const auto&f:c.sourceFaces)for(auto v:f)usedSource[v]=1u;
 std::vector<std::uint8_t> physicalDonorPresent(157u,0u);for(std::size_t f=0;f<c.faces.size();++f)for(auto v:c.faces[f])if(v<physicalDonor.size()&&physicalDonor[v]<physicalDonorPresent.size())physicalDonorPresent[physicalDonor[v]]=1u;
 std::vector<std::uint8_t> mappedSource=usedSource;std::size_t activeSourceCount=0u,mappedSourceCount=0u;for(std::size_t i=0;i<ns;++i){activeSourceCount+=usedSource[i]!=0u;if(sourceDonor[i]>=physicalDonorPresent.size()||!physicalDonorPresent[sourceDonor[i]])mappedSource[i]=0u;mappedSourceCount+=mappedSource[i]!=0u;}
 std::vector<std::uint32_t> sourceIds(ns),coarseSource(no,std::numeric_limits<std::uint32_t>::max());for(std::size_t i=0;i<ns;++i)sourceIds[i]=static_cast<std::uint32_t>(i);
 Owner forward;Owner::BuildReport forwardReport{};if(!forward.build(source,sourceIds,sourceDonor,mappedSource,c.outerWorld,coarseSource,physicalDonor,c.faces,&forwardReport,&error,maxDistance)){std::cerr<<"accepted-direction detail-to-O transfer failed: "<<error<<"\n";return 7;}
 if(forwardReport.projectedRows!=mappedSourceCount||forward.rows().size()!=ns){std::cerr<<"accepted-direction coverage differs from donor-supported source rows\n";return 8;}
 // Source-connected fallback only for the active rows whose original dominant donor
 // has no physical-O faces. Each representative is pinned by the independent source
 // face graph/corner crosswalk and the existing owner searches only its incident ring.
 std::ifstream rin(argv[4],std::ios::binary);if(!rin){std::cerr<<"representative map open failed\n";return 9;}
 std::uint64_t nr=0;if(!rin.read(reinterpret_cast<char*>(&nr),sizeof(nr))||nr>ns){std::cerr<<"invalid representative map header\n";return 9;}
 std::vector<std::uint32_t> repSource(nr),repO(nr);for(std::size_t i=0;i<nr;++i){std::uint32_t pair[2]{};if(!rin.read(reinterpret_cast<char*>(pair),sizeof(pair))){std::cerr<<"truncated representative map\n";return 9;}repSource[i]=pair[0];repO[i]=pair[1];}
 if(rin.peek()!=std::char_traits<char>::eof()){std::cerr<<"trailing representative map bytes\n";return 9;}
 std::vector<std::uint32_t> expectedFallback;for(std::size_t i=0;i<ns;++i)if(usedSource[i]&&!mappedSource[i])expectedFallback.push_back(std::uint32_t(i));
 std::cerr<<"EXPECTED_FALLBACK_COUNT="<<expectedFallback.size()<<" IDS=";for(auto q:expectedFallback)std::cerr<<q<<",";std::cerr<<"\n";if(expectedFallback.size()!=nr||repSource!=expectedFallback){std::cerr<<"representative rows do not exactly match active unsupported donor rows\n";return 9;}
 std::vector<Point> fallbackDetail(nr);std::vector<std::uint32_t> fallbackDonor(nr),fallbackSourceIds(nr);std::vector<std::uint8_t> fallbackUsed(nr,1u);
 for(std::size_t i=0;i<nr;++i){if(repO[i]>=no){std::cerr<<"representative O row out of range\n";return 9;}fallbackDetail[i]=source[repSource[i]];fallbackDonor[i]=sourceDonor[repSource[i]];fallbackSourceIds[i]=repSource[i];}
 Owner fallback;Owner::BuildReport fallbackReport{};std::vector<std::uint32_t> representative(repO.begin(),repO.end());
 if(!fallback.build(fallbackDetail,fallbackSourceIds,fallbackDonor,fallbackUsed,c.outerWorld,coarseSource,physicalDonor,c.faces,&fallbackReport,&error,maxDistance,&representative,nullptr,nullptr,nullptr,1u,nullptr)){std::cerr<<"source-connected representative fallback failed: "<<error<<"\n";return 10;}
 if(fallback.rows().size()!=nr||fallbackReport.projectedRows!=nr){std::cerr<<"fallback did not cover all 26 unsupported rows\n";return 10;}
 std::vector<std::uint32_t> fallbackFace(nr*4),fallbackSupport(nr*3);std::vector<float> fallbackWeights(nr*3),fallbackDistance(nr);
 for(std::size_t i=0;i<nr;++i){const auto fi=fallback.selectedFaceIndex(i);if(fi>=c.faces.size()){std::cerr<<"fallback face index out of range\n";return 10;}const auto&face=c.faces[fi];bool incident=false;for(auto v:face)incident|=v==repO[i];if(!incident){std::cerr<<"fallback face is outside representative one-ring\n";return 10;}fallbackFace[4*i]=repSource[i];fallbackFace[4*i+1]=repO[i];fallbackFace[4*i+2]=fi;fallbackFace[4*i+3]=fallback.rows()[i].donorBodyIndex;double sum=0;for(unsigned k=0;k<3;++k){const auto v=fallback.rows()[i].coarseVertex[k];const float wt=fallback.rows()[i].weight[k];if(v>=no||!std::isfinite(wt)||wt<0){std::cerr<<"invalid fallback support row\n";return 10;}fallbackSupport[3*i+k]=v;fallbackWeights[3*i+k]=wt;sum+=wt;}if(std::abs(sum-1.0)>1e-4){std::cerr<<"fallback row does not normalize\n";return 10;}fallbackDistance[i]=fallback.rows()[i].restDistance;double parent2=0;for(unsigned a=0;a<3;++a){const double dx=source[repSource[i]][a]-c.outerWorld[repO[i]][a];parent2+=dx*dx;}const double parent=std::sqrt(parent2);if(double(fallbackDistance[i])>parent+1e-9||fallbackDistance[i]>maxDistance){std::cerr<<"fallback one-ring distance cap failed\n";return 10;}}
 std::ofstream out(argv[2],std::ios::binary|std::ios::trunc);if(!out){std::cerr<<"output open failed\n";return 2;}
 const std::array<std::uint64_t,7> oh{no,report.inferredConstructionEmbeddingRows,report.explicitSourceEmbeddingRows,activeSourceCount,forwardReport.projectedRows+fallbackReport.projectedRows,ns-activeSourceCount,nr};out.write(reinterpret_cast<const char*>(oh.data()),sizeof(oh));
 std::vector<float> rowsOut(no*3);for(std::size_t i=0;i<no;++i){rowsOut[3*i]=float(rows[i]);rowsOut[3*i+1]=float(reverse.rows()[i].restDistance);rowsOut[3*i+2]=float(physicalDonor[i]);}
 std::vector<double> bOut(no*3),rOut(no*3);std::vector<std::uint32_t> tetOut(no*3), donorOut(physicalDonor);
 for(std::size_t i=0;i<no;++i){for(unsigned k=0;k<3;++k){bOut[3*i+k]=bary[i][k];rOut[3*i+k]=residual[i][k];tetOut[3*i+k]=reverse.rows()[i].coarseVertex[k];}}
 std::vector<std::uint32_t> forwardVertices(ns*3);std::vector<float> forwardWeights(ns*3),forwardDistance(ns);
 for(std::size_t i=0;i<ns;++i){if(!mappedSource[i])continue;const auto&r=forward.rows()[i];for(unsigned k=0;k<3;++k){forwardVertices[3*i+k]=r.coarseVertex[k];forwardWeights[3*i+k]=r.weight[k];}forwardDistance[i]=r.restDistance;}
 if(!writeVector(out,rowsOut)||!writeVector(out,bOut)||!writeVector(out,rOut)||!writeVector(out,tetOut)||!writeVector(out,donorOut)||!writeVector(out,usedSource)||!writeVector(out,mappedSource)||!writeVector(out,forwardVertices)||!writeVector(out,forwardWeights)||!writeVector(out,forwardDistance)||!writeVector(out,fallbackFace)||!writeVector(out,fallbackSupport)||!writeVector(out,fallbackWeights)||!writeVector(out,fallbackDistance)){std::cerr<<"output write failed\n";return 6;}out.close();if(!out){std::cerr<<"output finalize failed\n";return 6;}
 std::cout.precision(17);std::cout<<"PASS O_to_detail_rows="<<no<<" inferred="<<report.inferredConstructionEmbeddingRows<<" max_projection_m="<<projection.maximumProjectionDistanceM<<" rms_projection_m="<<projection.rmsProjectionDistanceM<<" detail_to_O_active="<<activeSourceCount<<" mapped="<<(forwardReport.projectedRows+fallbackReport.projectedRows)<<" direct="<<forwardReport.projectedRows<<" source_connected_fallback="<<fallbackReport.projectedRows<<" fallback_max_m="<<fallbackReport.maximumProjectionDistance<<" fallback_rms_m="<<fallbackReport.rmsProjectionDistance<<" max_detail_to_O_m="<<forwardReport.maximumProjectionDistance<<" rms_detail_to_O_m="<<forwardReport.rmsProjectionDistance<<" provenance="<<report.sourceEmbeddingProvenance<<"\n";
 return 0;
}
