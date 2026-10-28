#include "NumiHumanPhysicalSkinChildRegisteredVolumeIntegration.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;
using Key=std::array<std::uint32_t,3>;

namespace {

std::string number(double value) {
    std::ostringstream out;out<<std::setprecision(9)<<static_cast<float>(value);return out.str();
}
std::string quoted(const std::string& value) {
    if(value.find_first_of(",\"\r\n")==std::string::npos)return value;
    std::string out="\"";
    for(const char c:value){if(c=='\"')out+='\"';out+=c;}
    out+='\"';return out;
}
void writeRow(std::ostream& out,const std::vector<std::string>& fields) {
    for(std::size_t i=0;i<fields.size();++i)out<<(i?",":"")<<quoted(fields[i]);
    out<<"\n";
}
void pin(const std::filesystem::path& path,std::string& hash) {
    std::string error;assert(physical_skin_child_mass_detail::sha256File(path,hash,error));
}

struct Fixture {
    std::array<V3,9> points{{
        V3{0,0,2},V3{1,0,2},V3{0,1,2},
        V3{0,0,1},V3{1,0,1},V3{0,1,1},
        V3{0,0,0},V3{1,0,0},V3{0,1,0}}};
    std::array<std::array<std::uint32_t,4>,6> tets{{
        {{3,4,5,0}},{{4,5,0,1}},{{5,0,1,2}},
        {{6,7,8,3}},{{7,8,3,4}},{{8,3,4,5}}}};
    std::array<std::uint32_t,6> materials{{15,15,15,16,16,16}};
    PhysicalSkinChildMassMappingBundle parentBundle;
    PhysicalSkinRegisteredSurfaceCSVSet registered;
    std::vector<PhysicalSkinRegisteredBodyPose> poses;
    std::vector<PhysicalSkinAuthoredHeadWallRow> walls;

    Fixture() {
        for(const auto& tet:tets)
            assert(physical_skin_registered_volume_import_detail::signedSixVolume(
                std::vector<V3>(points.begin(),points.end()),tet)>0.0);
        auto& parent=parentBundle.parent;
        parent.currentWorldPoints.assign(points.begin(),points.end());
        parent.referenceWorldPoints.assign(points.begin(),points.end());
        parent.tetrahedra.assign(tets.begin(),tets.end());
        parent.materialIndices.assign(materials.begin(),materials.end());
        parent.donorBodyIndices.assign(tets.size(),0u);
        parent.anatomicalRegionIndices.assign(tets.size(),42u);
        parent.stableSourceFaceIdentities={601u,602u,603u,604u,605u,606u};

        poses.resize(157u);
        for(std::size_t body=0;body<poses.size();++body) {
            auto& pose=poses[body];pose.available=true;
            pose.bodyIndex=static_cast<std::uint32_t>(body);
            pose.binding=NM_INVALID_INDEX;pose.currentOrigin={0,0,0};pose.restOrigin={0,0,0};
            pose.currentLocalToWorld={1,0,0,0,1,0,0,0,1};
            pose.restLocalToWorld=pose.currentLocalToWorld;
        }
        poses[0].binding=0u;
        registered.framesByDonor.resize(157u);
        registered.framesByDonor[0].available=true;
        registered.framesByDonor[0].binding=0u;
        registered.framesByDonor[0].currentR={1,0,0,0,1,0,0,0,1};
        registered.framesByDonor[0].restR=registered.framesByDonor[0].currentR;

        std::array<RegisteredSurface*,3> surfaces{{
            &registered.surfaces[0],&registered.surfaces[1],&registered.surfaces[2]}};
        const std::array<std::uint32_t,3> offsets{{0u,3u,6u}};
        const std::array<std::string,3> names{{"O","I","D"}};
        for(unsigned layer=0;layer<3;++layer) {
            auto& surface=*surfaces[layer];
            surface.name=names[layer];surface.sourceLayer=layer;
            surface.faces={{0,1,2}};
            surface.sourceFaceIndices={101u+layer};
            surface.donorBodyIndices.assign(3u,0u);
            surface.worldToDonorLocalLinear.assign(3u,{1,0,0,0,1,0,0,0,1});
            surface.sourceVertexIndices.resize(3u);
            surface.sourceEmbeddingFaceIndices.assign(3u,NM_INVALID_INDEX);
            surface.sourceEmbeddingBarycentric.resize(3u);
            surface.sourceEmbeddingResidualWorldM.resize(3u);
            for(unsigned row=0;row<3;++row) {
                const auto source=offsets[layer]+row;
                surface.worldPoints.push_back(points[source]);
                surface.donorLocalPoints.push_back(points[source]);
                surface.sourceVertexIndices[row]=source;
                PhysicalSkinNodeSourceRow sourceRow;
                sourceRow.vertex=source;sourceRow.face=-1;sourceRow.domain="fixture-detail";
                sourceRow.frameMethod="fixture-frame";
                registered.nodeSource[layer].push_back(sourceRow);
            }
        }

        const std::array<std::uint32_t,9> parentRows{{0,1,2,0,1,2,3,4,5}};
        for(unsigned layer=0;layer<3;++layer) {
            for(unsigned row=0;row<3;++row) {
                PhysicalSkinParentRegistrationRow registration;
                registration.available=true;registration.tet=parentRows[3u*layer+row];
                const auto node=offsets[layer]+row;
                for(unsigned corner=0;corner<4;++corner)
                    if(tets[registration.tet][corner]==node)registration.bary[corner]=1.0;
                registration.distance=0.0;registration.tieCount=0u;registration.residual={0,0,0};
                registered.nodeParent[layer].push_back(registration);
            }
        }
        auto facets=buildFacetRows();
        walls=std::move(facets.second);
    }

    std::pair<std::vector<std::vector<std::string>>,std::vector<PhysicalSkinAuthoredHeadWallRow>>
    buildFacetRows() const {
        std::map<Key,std::vector<std::uint32_t>> uses;
        constexpr std::array<std::array<unsigned,3>,4> corners{{
            {{1,2,3}},{{0,3,2}},{{0,1,3}},{{0,2,1}}}};
        for(std::uint32_t cell=0;cell<tets.size();++cell)for(unsigned omitted=0;omitted<4;++omitted) {
            Key key{};for(unsigned c=0;c<3;++c)key[c]=tets[cell][corners[omitted][c]];
            std::sort(key.begin(),key.end());uses[key].push_back(cell);
        }
        std::vector<std::vector<std::string>> rows;
        const auto registeredFace=[&](std::string partition,unsigned marker,std::string role,
                                      std::uint32_t cell,Key nodes,std::uint32_t stable) {
            std::vector<std::string> fields(24u,"-1");
            fields[0]=std::move(partition);fields[1]="0";fields[2]=std::to_string(marker);
            fields[3]=std::move(role);fields[4]=std::to_string(stable);
            fields[17]=std::to_string(cell+1u);fields[18]=std::to_string(cell);
            for(unsigned c=0;c<3;++c)fields[19u+c]=std::to_string(nodes[c]+1u);
            rows.push_back(std::move(fields));
        };
        registeredFace("skin",1u,"outer_O",2u,{0,1,2},701u);
        registeredFace("skin",2u,"inner_I",0u,{3,4,5},702u);
        registeredFace("fat",4u,"outer_I",5u,{3,4,5},703u);
        registeredFace("fat",5u,"inner_D",3u,{6,7,8},704u);

        std::vector<PhysicalSkinAuthoredHeadWallRow> authored;
        for(const auto& [face,cells]:uses) {
            if(cells.size()!=1u||(face==Key{0,1,2})||(face==Key{6,7,8}))continue;
            const auto cell=cells.front();PhysicalSkinAuthoredHeadWallRow wall;
            wall.stableWallFace=800u+static_cast<std::uint32_t>(authored.size());
            wall.childTet=cell;wall.childNodes=face;wall.parentTet=cell;
            wall.material=materials[cell];wall.parentNodes=face;
            authored.push_back(wall);
            const bool skin=wall.material==15u;
            std::vector<std::string> fields(24u,"-1");
            fields[0]=skin?"skin":"fat";fields[1]="0";fields[2]=skin?"3":"6";
            fields[3]=skin?"donor_skin_OI_headwall":"donor_fat_ID_headwall";
            fields[4]=std::to_string(wall.stableWallFace);fields[10]="5";
            fields[13]="[1,2,3]";fields[14]=std::to_string(wall.parentTet);
            fields[15]=std::to_string(wall.material);
            fields[16]="["+std::to_string(face[0])+","+std::to_string(face[1])+","+std::to_string(face[2])+"]";
            fields[17]=std::to_string(cell+1u);fields[18]=std::to_string(cell);
            for(unsigned c=0;c<3;++c)fields[19u+c]=std::to_string(face[c]+1u);
            fields[22]="[1,149,1]";fields[23]="[98,99,100]";
            rows.push_back(std::move(fields));
        }
        assert(authored.size()==12u);
        return {rows,authored};
    }

    void writeFiles(const std::filesystem::path& root,PhysicalSkinRegisteredVolumeAssetFiles& files) const {
        std::filesystem::create_directories(root);
        const auto node=root/"child.node",ele=root/"child.ele",labels=root/"child.labels.csv";
        const auto map=root/"combined-node-map.csv",facets=root/"facet-ancestry.csv";
        {
            std::ofstream out(node);out<<points.size()<<" 3 0 0\n";
            for(std::size_t i=0;i<points.size();++i)
                out<<(i+1u)<<" "<<number(points[i][0])<<" "<<number(points[i][1])<<" "<<number(points[i][2])<<"\n";
        }
        {
            std::ofstream out(ele);out<<tets.size()<<" 4 1\n";
            for(std::size_t i=0;i<tets.size();++i) {
                out<<(101u+7u*static_cast<unsigned>(i));
                for(const auto nodeRow:tets[i])out<<" "<<nodeRow+1u;
                out<<" "<<materials[i]<<"\n";
            }
        }
        {
            std::ofstream out(labels);out<<"child_tet_row,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id\n";
            for(std::size_t i=0;i<tets.size();++i)
                out<<i<<","<<materials[i]<<",0,42,"<<601u+i<<"\n";
        }
        {
            std::ofstream out(map);writeRow(out,physicalSkinCombinedNodeMapCSVHeader());
            std::vector<std::vector<std::string>> rows;
            const auto append=[&](unsigned node,const std::string& partition,std::uint32_t source,
                                  const std::string& layer,std::uint32_t surfaceRow,
                                  std::uint32_t interfaceRow,const std::string& peerPartition,
                                  std::uint32_t peerID) {
                std::vector<std::string> f(42u,"0");
                f[0]=std::to_string(node+1u);f[1]=partition;f[2]=std::to_string(source);
                f[3]=interfaceRow==NM_INVALID_INDEX?"-1":std::to_string(interfaceRow);
                f[4]=peerPartition;f[5]=peerPartition.empty()?"":std::to_string(peerID);
                f[6]=layer;f[7]=layer.empty()?"-1":std::to_string(surfaceRow);
                f[8]=layer.empty()?"interior":layer;
                for(unsigned axis=0;axis<3;++axis) {
                    f[9u+axis]=number(points[node][axis]);f[14u+axis]=number(points[node][axis]);
                    f[24u+axis]="0";f[34u+axis]="0";
                }
                f[12]="0";f[13]="0";f[17]="fixture-detail";f[18]="vertex";
                f[19]=std::to_string(node);f[20]="-1";f[21]=f[22]=f[23]="0";
                f[27]="fixture-frame";f[28]="source_parent_tet_reference_registration";
                const std::array<std::uint32_t,9> parentRows{{0,1,2,0,1,2,3,4,5}};
                const auto parentRow=parentRows[node];f[29]=std::to_string(parentRow);
                for(unsigned corner=0;corner<4;++corner)
                    f[30u+corner]=tets[parentRow][corner]==node?"1":"0";
                f[37]="0";f[38]="0";
                f[39]=node<3u?"32768":(node<6u?"98304":"65536");
                f[40]="0";f[41]="42";
                rows.push_back(std::move(f));
            };
            for(unsigned node=0;node<3;++node)append(node,"skin",110u+node,"O",node,NM_INVALID_INDEX,"",0u);
            for(unsigned row=0;row<3;++row) {
                const auto node=3u+row,skinID=210u+row,fatID=310u+row;
                append(node,"skin",skinID,"I",row,row,"fat",fatID);
                append(node,"fat",fatID,"I",row,row,"skin",skinID);
            }
            for(unsigned row=0;row<3;++row)append(6u+row,"fat",410u+row,"D",row,NM_INVALID_INDEX,"",0u);
            std::reverse(rows.begin(),rows.end());
            for(const auto& row:rows)writeRow(out,row);
        }
        {
            std::ofstream out(facets);writeRow(out,physicalSkinRegisteredFacetCSVHeader());
            auto faceRows=buildFacetRows().first;
            for(const auto& row:faceRows)writeRow(out,row);
        }
        files.tetgenNode=node;files.tetgenElements=ele;files.childLabels=labels;
        files.combinedNodeMapCSV=map;files.facetAncestryCSV=facets;
        pin(node,files.expectedSHA256[0]);pin(ele,files.expectedSHA256[1]);
        pin(labels,files.expectedSHA256[2]);pin(map,files.expectedSHA256[3]);
        pin(facets,files.expectedSHA256[4]);
    }
};

void checkFailureIsAtomic(Fixture& fixture,const PhysicalSkinRegisteredVolumeAssetFiles& files,
                          std::string_view expected) {
    auto bundle=fixture.parentBundle;
    PhysicalSkinRegisteredVolumeImportSidecars sidecars;
    sidecars.sourceDomains={"sentinel"};
    PhysicalSkinRegisteredVolumeFileImportReport report;
    report.nodeMap.fileRows=6789u;
    std::vector<PhysicalSkinStableDeepAttachmentRequest> attachments;
    std::string error;
    const bool result=importPhysicalSkinRegisteredVolumeFiles(files,std::string(64u,'0'),157u,0.02,
        0.02,"fixture-detail",9u,0u,fixture.registered,attachments,fixture.poses,bundle,
        sidecars,report,error);
    assert(!result&&error.find(expected)!=std::string::npos);
    assert(bundle.currentWorldPoints.empty()&&bundle.childTetrahedra.empty());
    assert(sidecars.sourceDomains==std::vector<std::string>{"sentinel"});
    assert(report.nodeMap.fileRows==6789u);
}
}

int main() {
    Fixture fixture;
    const auto root=std::filesystem::temp_directory_path()/
        ("numi-registered-volume-import-"+std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    PhysicalSkinRegisteredVolumeAssetFiles files;
    fixture.writeFiles(root,files);

    auto bundle=fixture.parentBundle;
    PhysicalSkinRegisteredVolumeImportSidecars sidecars;
    PhysicalSkinRegisteredVolumeFileImportReport report;
    std::vector<PhysicalSkinStableDeepAttachmentRequest> attachments;
    std::string error;
    const std::vector<PhysicalSkinSourceDomainRegistration> typedDomains{
        {"fixture-detail",PhysicalSkinSourceDomainRole::outerSurface,std::string(64u,'1'),9u,1u},
        {"fixture-detail",PhysicalSkinSourceDomainRole::innerSurface,std::string(64u,'2'),9u,1u},
        {"fixture-detail",PhysicalSkinSourceDomainRole::deepSurface,std::string(64u,'3'),9u,1u}};
    assert(importPhysicalSkinRegisteredVolumeFiles(files,std::string(64u,'0'),157u,0.02,
        0.02,"",0u,0u,fixture.registered,attachments,fixture.poses,
        bundle,sidecars,report,error,std::span<const PhysicalSkinSourceDomainRegistration>(typedDomains)));
    assert(report.registeredVolume.typedSourceDomainMapPopulated&&
           !report.registeredVolume.canonicalDetailedSourceMapPopulated&&
           report.registeredVolume.sourceDomainCount==3u);
    assert(bundle.detailedSourceVertexCount==0u&&bundle.detailedSourceFaceCount==0u);
    assert(bundle.independentSkinInner.sourceDomainIndices==std::vector<std::uint32_t>({1u,1u,1u}));
    assert(bundle.independentFatDeep.sourceDomainIndices==std::vector<std::uint32_t>({2u,2u,2u}));
    assert(report.tetgen.nodeCount==9u&&report.tetgen.cellCount==6u);
    assert(report.nodeMap.volumeNodes==9u&&report.nodeMap.fileRows==12u&&report.nodeMap.sharedInterfacePairs==3u);
    assert(report.facetAncestry.registeredRows==4u&&report.facetAncestry.authoredWallRows==12u);
    assert((report.facetMaps.layerFaceUses==std::array<std::size_t,3>{1u,2u,1u}));
    assert(report.facetMaps.authoredHeadWallFaces==12u&&report.physicalAdmission==false);
    assert(bundle.boundaryMap.outer==std::vector<std::uint32_t>({0u,1u,2u}));
    assert(bundle.boundaryMap.skinInner==std::vector<std::uint32_t>({3u,4u,5u}));
    assert(bundle.boundaryMap.fatDeep==std::vector<std::uint32_t>({6u,7u,8u}));
    assert(bundle.childRegisteredLayerFacetSurfaceFaces.size()==24u);
    assert(std::count(bundle.childAuthoredHeadSidewallFacetFlags.begin(),
        bundle.childAuthoredHeadSidewallFacetFlags.end(),1u)==12);
    assert(report.inputSHA256==files.expectedSHA256);

    auto badMap=files;
    badMap.combinedNodeMapCSV=root/"bad-map.csv";
    {std::ofstream out(badMap.combinedNodeMapCSV);out<<"not,the,existing,header\n";}
    pin(badMap.combinedNodeMapCSV,badMap.expectedSHA256[3]);
    checkFailureIsAtomic(fixture,badMap,"header differs");

    auto compressed=files;
    compressed.combinedNodeMapCSV=root/"combined-node-map.csv.gz";
    std::filesystem::copy_file(files.combinedNodeMapCSV,compressed.combinedNodeMapCSV);
    checkFailureIsAtomic(fixture,compressed,"plain existing CSV");

    std::filesystem::remove_all(root);
    std::cout<<"registered_volume_file_import=PASS positive_TetGen_prism=6cells O/I/D=1/2/1 walls=12"
        <<" exact_node_map_rows=12 malformed_csv_and_gzip_rejected atomic_failure_preserved=1"
        <<" physical_admission=NOT_CLAIMED\n";
}
