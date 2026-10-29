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
    std::ostringstream out;out<<std::setprecision(17)<<double(static_cast<float>(value));return out.str();
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
    std::vector<V3> points{
        V3{0,0,2},V3{1,0,2},V3{0,1,2},
        V3{0,0,1},V3{1,0,1},V3{0,1,1},
        V3{0,0,0},V3{1,0,0},V3{0,1,0}};
    std::vector<std::array<std::uint32_t,4>> tets{
        {{3,4,5,0}},{{4,5,0,1}},{{5,0,1,2}},
        {{6,7,8,3}},{{7,8,3,4}},{{8,3,4,5}}};
    std::vector<std::uint32_t> materials{15,15,15,16,16,16};
    std::vector<std::uint32_t> parentTetForChild;
    std::vector<std::uint32_t> sourceParentTetForNode;
    std::vector<std::array<double,4>> sourceParentBaryForNode;
    std::vector<std::uint32_t> operationalTetForNode;
    std::vector<double> historicalDistanceForNode;
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

        parentTetForChild.resize(tets.size());
        for(std::size_t i=0;i<tets.size();++i)parentTetForChild[i]=static_cast<std::uint32_t>(i);
        const std::array<std::uint32_t,9> parentRows{{0,1,2,0,1,2,3,4,5}};
        sourceParentTetForNode.assign(parentRows.begin(),parentRows.end());
        sourceParentBaryForNode.resize(points.size());
        for(std::size_t node=0;node<points.size();++node)
            for(unsigned corner=0;corner<4u;++corner)
                if(tets[sourceParentTetForNode[node]][corner]==node)
                    sourceParentBaryForNode[node][corner]=1.0;
        historicalDistanceForNode.assign(points.size(),0.0);

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

    void prepareSeparatedOperationalReference() {
        const auto original=tets.front();
        const auto newNode=static_cast<std::uint32_t>(points.size());
        points.push_back({double(float(0.255)),0.25,1.25});
        sourceParentTetForNode.push_back(0u);
        sourceParentBaryForNode.push_back({0.25,0.25,0.25,0.25});
        historicalDistanceForNode.push_back(double(float(points.back()[0]-0.25)));
        std::vector<std::array<std::uint32_t,4>> split;
        const std::array<std::array<unsigned,3>,4> faces{{
            {{1,2,3}},{{0,3,2}},{{0,1,3}},{{0,2,1}}}};
        // Keep the registered I face as the first child face. The new vertex
        // lies inside the parent tet but carries a separate 5 mm historical
        // source registration residual; its candidate reference map is exact.
        const std::array<unsigned,4> faceOrder{{3,0,1,2}};
        for(const auto omitted:faceOrder){
            std::array<std::uint32_t,4> child{{newNode,original[faces[omitted][0]],
                original[faces[omitted][1]],original[faces[omitted][2]]}};
            if(physical_skin_registered_volume_import_detail::signedSixVolume(points,child)<0.0)
                std::swap(child[2],child[3]);
            assert(physical_skin_registered_volume_import_detail::signedSixVolume(points,child)>0.0);
            split.push_back(child);
        }
        std::vector<std::array<std::uint32_t,4>> nextTets=split;
        nextTets.insert(nextTets.end(),tets.begin()+1,tets.end());
        std::vector<std::uint32_t> nextMaterials(4u,15u);
        nextMaterials.insert(nextMaterials.end(),materials.begin()+1,materials.end());
        std::vector<std::uint32_t> nextParents(4u,0u);
        nextParents.insert(nextParents.end(),parentTetForChild.begin()+1,parentTetForChild.end());
        tets=std::move(nextTets);materials=std::move(nextMaterials);
        parentTetForChild=std::move(nextParents);
        operationalTetForNode.assign(points.size(),NM_INVALID_INDEX);
        for(std::uint32_t cell=0;cell<tets.size();++cell)
            for(const auto node:tets[cell])
                if(operationalTetForNode[node]==NM_INVALID_INDEX)operationalTetForNode[node]=cell;
        operationalTetForNode.resize(points.size());
        registered.parentRegistrationIsHistoricalProvenanceOnly=true;
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
        const bool splitTopology=tets.size()==9u;
        registeredFace("skin",1u,"outer_O",splitTopology?5u:2u,{0,1,2},701u);
        registeredFace("skin",2u,"inner_I",0u,{3,4,5},702u);
        registeredFace("fat",4u,"outer_I",splitTopology?8u:5u,{3,4,5},703u);
        registeredFace("fat",5u,"inner_D",splitTopology?6u:3u,{6,7,8},704u);

        std::vector<PhysicalSkinAuthoredHeadWallRow> authored;
        for(const auto& [face,cells]:uses) {
            if(cells.size()!=1u||(face==Key{0,1,2})||(face==Key{6,7,8}))continue;
            const auto cell=cells.front();PhysicalSkinAuthoredHeadWallRow wall;
            wall.stableWallFace=800u+static_cast<std::uint32_t>(authored.size());
            wall.childTet=cell;wall.childNodes=face;wall.parentTet=parentTetForChild[cell];
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

    void writeFiles(const std::filesystem::path& root,PhysicalSkinRegisteredVolumeAssetFiles& files,
                    const bool separatedOperationalReference=false) const {
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
                out<<i<<","<<materials[i]<<",0,42,"
                   <<parentBundle.parent.stableSourceFaceIdentities[parentTetForChild[i]]<<"\n";
        }
        {
            std::ofstream out(map);writeRow(out,physicalSkinCombinedNodeMapCSVHeader(separatedOperationalReference));
            std::vector<std::vector<std::string>> rows;
            const auto append=[&](unsigned node,const std::string& partition,std::uint32_t source,
                                  const std::string& layer,std::uint32_t surfaceRow,
                                  std::uint32_t interfaceRow,const std::string& peerPartition,
                                  std::uint32_t peerID) {
                std::vector<std::string> f(separatedOperationalReference?54u:42u,"0");
                f[0]=std::to_string(node+1u);f[1]=partition;f[2]=std::to_string(source);
                f[3]=interfaceRow==NM_INVALID_INDEX?"-1":std::to_string(interfaceRow);
                f[4]=peerPartition;f[5]=peerPartition.empty()?"":std::to_string(peerID);
                f[6]=layer;f[7]=layer.empty()?"-1":std::to_string(surfaceRow);
                f[8]=layer.empty()?"interior":layer;
                for(unsigned axis=0;axis<3;++axis) {
                    f[9u+axis]=number(points[node][axis]);f[14u+axis]=number(points[node][axis]);
                    f[24u+axis]="0";f[34u+axis]="0";
                }
                f[12]="0";f[13]="0";
                f[17]=node<9u?"fixture-detail":"derived_volume_interior_no_surface_ancestry";
                f[18]=node<9u?"vertex":"none";
                f[19]=node<9u?std::to_string(node):"-1";f[20]="-1";f[21]=f[22]=f[23]="0";
                f[27]=node<9u?"fixture-frame":"derived_full_all_bodies_geometric_donor_pose";
                const auto parentRow=sourceParentTetForNode[node];
                f[28]=node<9u?"source_parent_tet_reference_registration":
                    "inferred_current_parent_tet_same_material_donor_region";
                f[29]=std::to_string(parentRow);
                for(unsigned corner=0;corner<4u;++corner)
                    f[30u+corner]=number(sourceParentBaryForNode[node][corner]);
                f[37]=number(historicalDistanceForNode[node]);f[38]="0";
                f[39]=node<3u?"32768":(node<6u?"98304":(node<9u?"65536":"32768"));
                f[40]="0";f[41]="42";
                const auto parentBase=physical_skin_registered_volume_import_detail::baryPoint(
                    parentBundle.parent.currentWorldPoints,parentBundle.parent.tetrahedra[parentRow],
                    sourceParentBaryForNode[node]);
                for(unsigned axis=0;axis<3u;++axis)
                    f[34u+axis]=number(points[node][axis]-parentBase[axis]);
                if(separatedOperationalReference){
                    f[42]=node<9u?std::to_string(node):"-1";
                    const auto operationalTet=operationalTetForNode[node];
                    f[43]=std::to_string(operationalTet);
                    for(unsigned corner=0;corner<4u;++corner)
                        f[44u+corner]=tets[operationalTet][corner]==node?"1":"0";
                    f[48]=f[49]=f[50]="0";f[51]="0";
                    f[52]=std::to_string(parentBundle.parent.donorBodyIndices[parentRow]);
                    f[53]=std::to_string(parentBundle.parent.anatomicalRegionIndices[parentRow]);
                }
                rows.push_back(std::move(f));
            };
            for(unsigned node=0;node<3;++node)append(node,"skin",110u+node,"O",node,NM_INVALID_INDEX,"",0u);
            for(unsigned row=0;row<3;++row) {
                const auto node=3u+row,skinID=210u+row,fatID=310u+row;
                append(node,"skin",skinID,"I",row,row,"fat",fatID);
                append(node,"fat",fatID,"I",row,row,"skin",skinID);
            }
            for(unsigned row=0;row<3;++row)append(6u+row,"fat",410u+row,"D",row,NM_INVALID_INDEX,"",0u);
            if(points.size()>9u)append(9u,"skin",999u,"",NM_INVALID_INDEX,NM_INVALID_INDEX,"",0u);
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
        files.registrationDomain=separatedOperationalReference?
            PhysicalSkinRegisteredVolumeRegistrationDomain::candidateOperationalReference:
            PhysicalSkinRegisteredVolumeRegistrationDomain::sourceMassParent;
        pin(node,files.expectedSHA256[0]);pin(ele,files.expectedSHA256[1]);
        pin(labels,files.expectedSHA256[2]);pin(map,files.expectedSHA256[3]);
        pin(facets,files.expectedSHA256[4]);
    }
};

PhysicalSkinRegisteredVolumeAssetFiles rewriteSplitMapField(
        const PhysicalSkinRegisteredVolumeAssetFiles& files,const std::filesystem::path& root,
        const std::string& name,const std::uint32_t node,const std::size_t column,
        const std::string& value) {
    auto changed=files;changed.combinedNodeMapCSV=root/name;
    std::ifstream input(files.combinedNodeMapCSV);
    std::ofstream output(changed.combinedNodeMapCSV);
    std::string line;
    assert(std::getline(input,line));
    writeRow(output,physicalSkinCombinedNodeMapCSVHeader(true));
    bool found=false;
    while(std::getline(input,line)){
        auto fields=physical_skin_child_mass_detail::splitCSV(line);
        assert(fields.size()==54u&&column<fields.size());
        std::uint32_t id=0;assert(physical_skin_child_mass_detail::parseNumber(fields[0],id));
        if(id==node+1u){fields[column]=value;found=true;}
        writeRow(output,fields);
    }
    assert(found);
    output.close();input.close();
    pin(changed.combinedNodeMapCSV,changed.expectedSHA256[3]);
    return changed;
}

void checkFailureIsAtomic(Fixture& fixture,const PhysicalSkinRegisteredVolumeAssetFiles& files,
                          std::string_view expected) {
    auto bundle=fixture.parentBundle;
    PhysicalSkinRegisteredVolumeImportSidecars sidecars;
    sidecars.sourceDomains={"sentinel"};
    PhysicalSkinRegisteredVolumeFileImportReport report;
    report.nodeMap.fileRows=6789u;
    std::vector<PhysicalSkinStableDeepAttachmentRequest> attachments;
    std::string error;
    const double parentRegistrationBound=files.registrationDomain==
        PhysicalSkinRegisteredVolumeRegistrationDomain::candidateOperationalReference?0.0004:0.02;
    const bool result=importPhysicalSkinRegisteredVolumeFiles(files,std::string(64u,'0'),157u,0.02,
        parentRegistrationBound,"fixture-detail",9u,0u,fixture.registered,attachments,fixture.poses,bundle,
        sidecars,report,error);
    if(result||error.find(expected)==std::string::npos)
        std::cerr<<"negative split/import expected ["<<expected<<"] got ["<<error<<"] result="<<result<<"\n";
    assert(!result&&error.find(expected)!=std::string::npos);
    assert(bundle.currentWorldPoints.empty()&&bundle.childTetrahedra.empty());
    assert(sidecars.sourceDomains==std::vector<std::string>{"sentinel"});
    assert(report.nodeMap.fileRows==6789u);
}
}

void checkNativeFloat32FrameContract(const std::filesystem::path& root) {
    using namespace physical_skin_registered_surface_csv_detail;
    const std::array<double,9> rounded{double(1.0f+2.0f*std::numeric_limits<float>::epsilon()),0,0,0,1,0,0,0,1};
    assert(!rotation(rounded)&&nativeFloat32Rotation(rounded));
    const V3 point{2.0,-3.0,4.0};
    const auto nativeInverse=worldToLocal(rounded,PhysicalSkinPoseArithmetic::nativeFloat32);
    const auto exactLocal=mul(nativeInverse,point);
    assert(norm(sub(mul(rounded,exactLocal),point))<1e-14);
    assert(norm(sub(mul(rounded,mul(tr(rounded),point)),point))>1e-7);
    assert(sameWorldToLocal(nativeInverse,nativeInverse,PhysicalSkinPoseArithmetic::nativeFloat32));
    auto perturbedInverse=nativeInverse;perturbedInverse[0]+=1e-12;
    assert(!sameWorldToLocal(perturbedInverse,nativeInverse,PhysicalSkinPoseArithmetic::nativeFloat32));
    assert(worldToLocal(rounded,PhysicalSkinPoseArithmetic::strictDouble)==tr(rounded));
    PhysicalSkinRegisteredSurfaceCSVFiles files;
    files.donorCount=1u;files.donorFrames=root/"native-f32-frames.csv";
    PhysicalSkinDonorFrameRow native;native.available=true;native.binding=0u;
    native.currentR=rounded;native.restR=rounded;
    const auto write=[&](const PhysicalSkinDonorFrameRow& row){
        std::ofstream out(files.donorFrames);out<<std::setprecision(17);
        out<<"binding,donor,current_x,current_y,current_z";
        for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)out<<",current_R"<<i<<j;
        out<<",rest_x,rest_y,rest_z";
        for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)out<<",rest_R"<<i<<j;
        out<<"\n0,0";
        for(double x:row.currentOrigin)out<<','<<x;
        for(double x:row.currentR)out<<','<<x;
        for(double x:row.restOrigin)out<<','<<x;
        for(double x:row.restR)out<<','<<x;
        out<<'\n';
    };
    const auto read=[&](){PhysicalSkinRegisteredSurfaceCSVSet set;
        PhysicalSkinRegisteredSurfaceCSVReport report;std::string error;
        return readFrames(files,set,report,error);};
    write(native);assert(!read()); // External/default SO3 gate is unchanged.
    std::array<PhysicalSkinDonorFrameRow,1> pins{native};
    files.nativeFloat32FramesByDonor=pins;assert(read());
    auto wrong=native;wrong.currentR[0]=std::nextafter(float(wrong.currentR[0]),2.0f);
    write(wrong);assert(!read()); // Even a bounded one-ULP mismatch is rejected.
    wrong=native;wrong.currentOrigin[0]=double(std::numeric_limits<float>::epsilon());
    write(wrong);assert(!read());
    for(const unsigned kind:{0u,1u,2u,3u}) {
        wrong=native;
        if(kind==0u)wrong.currentR[0]=-1.0; // reflection
        if(kind==1u)wrong.currentR[1]=0.01; // shear
        if(kind==2u)wrong.currentR[0]=1.0+1e-9; // not native F32
        if(kind==3u)wrong.currentR[0]=double(1.0f+32.0f*std::numeric_limits<float>::epsilon());
        pins[0]=wrong;write(wrong);assert(!read()); // Pins alone cannot waive bound.
    }
    pins[0]=native;write(native);
    files.nativeFloat32FramesByDonor=std::span<const PhysicalSkinDonorFrameRow>(pins.data(),0u);
    assert(!read());
    std::array<PhysicalSkinDonorFrameRow,2> badDomain{native,native};
    files.nativeFloat32FramesByDonor=badDomain;assert(!read());
}

int main() {
    Fixture fixture;
    const auto root=std::filesystem::temp_directory_path()/
        ("numi-registered-volume-import-"+std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    checkNativeFloat32FrameContract(root);
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

    Fixture splitFixture;
    splitFixture.prepareSeparatedOperationalReference();
    PhysicalSkinRegisteredVolumeAssetFiles splitFiles;
    splitFixture.writeFiles(root/"operational-reference-split",splitFiles,true);
    // CandidateOperationalReference cannot use the legacy source-only import
    // contract. The actual positive authenticated route is exercised against
    // the pinned 330-to-002-to-514 assets by the owner integration control.
    checkFailureIsAtomic(splitFixture,splitFiles,
        "requires authenticated source and candidate receipts plus a same-topology pullback proof");

    // The complete native body table uses the same explicit arithmetic policy,
    // including bodies without registered skin bindings. Default stays strict.
    auto roundedPoses=fixture.poses;
    roundedPoses[155u].currentLocalToWorld[0]=double(1.0f+2.0f*std::numeric_limits<float>::epsilon());
    auto poseBundle=fixture.parentBundle;
    assert(!importPhysicalSkinRegisteredVolumeFiles(files,std::string(64u,'0'),157u,0.02,
        0.02,"",0u,0u,fixture.registered,attachments,roundedPoses,
        poseBundle,sidecars,report,error,std::span<const PhysicalSkinSourceDomainRegistration>(typedDomains)));
    roundedPoses[155u].arithmetic=PhysicalSkinPoseArithmetic::nativeFloat32;
    assert(importPhysicalSkinRegisteredVolumeFiles(files,std::string(64u,'0'),157u,0.02,
        0.02,"",0u,0u,fixture.registered,attachments,roundedPoses,
        poseBundle,sidecars,report,error,std::span<const PhysicalSkinSourceDomainRegistration>(typedDomains)));

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
        <<" native_F32_exact_pins_and_roundoff_bound=PASS strict_default_unchanged=1"
        <<" candidate_operational_reference_missing_proof_rejected=PASS"
        <<" immutable_parent_packet=1 actual_authenticated_chain_separately_checked=1"
        <<" physical_admission=NOT_CLAIMED\n";
}
