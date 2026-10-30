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
#include <tuple>
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



    void scaleSyntheticGeometry(const double scale) {
        assert(std::isfinite(scale)&&scale>0.0);
        for(auto& point:points)for(auto& x:point)x*=scale;
        for(auto& point:parentBundle.parent.currentWorldPoints)for(auto& x:point)x*=scale;
        for(auto& point:parentBundle.parent.referenceWorldPoints)for(auto& x:point)x*=scale;
        for(auto& distance:historicalDistanceForNode)distance*=scale;
        for(auto& surface:registered.surfaces) {
            for(auto& point:surface.worldPoints)for(auto& x:point)x*=scale;
            for(auto& point:surface.donorLocalPoints)for(auto& x:point)x*=scale;
        }
    }

    void initializeSyntheticParentMass() {
        auto& parent=parentBundle.parent;
        const auto nodeCount=parent.referenceWorldPoints.size();
        const auto cellCount=parent.tetrahedra.size();
        assert(nodeCount==9u&&cellCount==6u);
        parent.compiledLumpedNodeMassKg.assign(nodeCount,0.0);
        parent.compiledGlobalNodeIndices.resize(nodeCount);
        parent.nodeFrameOwnerBodyIndices.assign(nodeCount,0u);
        parent.nodeFrameOwnerLocalPoints=parent.currentWorldPoints;
        std::map<std::tuple<std::uint32_t,std::uint32_t,std::uint32_t>,double> sourceMassByNodeMaterial;
        parent.referenceTetVolumeM3.resize(cellCount);
        parent.tetrahedronMassKg.resize(cellCount);
        parent.totalMassKg=0.0;
        for(std::size_t node=0;node<nodeCount;++node)
            parent.compiledGlobalNodeIndices[node]=static_cast<std::uint32_t>(node);
        for(std::size_t cell=0;cell<cellCount;++cell) {
            const auto volume=parentReferenceTetVolumeM3(parent.tetrahedra[cell],parent.referenceWorldPoints);
            const double density=parent.materialIndices[cell]==15u?1000.0:900.0;
            const double mass=volume*density;
            assert(std::isfinite(volume)&&volume>0.0&&std::isfinite(mass)&&mass>0.0);
            parent.referenceTetVolumeM3[cell]=volume;
            parent.tetrahedronMassKg[cell]=mass;
            parent.totalMassKg+=mass;
            for(const auto node:parent.tetrahedra[cell]) {
                parent.compiledLumpedNodeMassKg[node]+=mass*0.25;
                sourceMassByNodeMaterial[{node,parent.materialIndices[cell],0u}]+=mass*0.25*0.7;
                sourceMassByNodeMaterial[{node,parent.materialIndices[cell],1u}]+=mass*0.25*0.3;
            }
        }
        parent.sourceMassContributions.clear();
        for(const auto& [key,mass]:sourceMassByNodeMaterial) {
            const auto [node,material,donor]=key;
            parent.sourceMassContributions.push_back({
                .parentNodeIndex=node,.materialIndex=material,.donorBodyIndex=donor,
                .massKg=mass,.donorLocalPosition=parent.nodeFrameOwnerLocalPoints[node]});
        }
        parent.maximumNodalMassClosureErrorKg=0.0;
        parent.maximumNodalMassClosureRelativeError=0.0;
        auto hashPacket=spatial_mass_transport_detail::SHA256Builder{};
        hashPacket.string("synthetic-direct-source-parent-import-fixture.v1");
        for(const auto& point:parent.referenceWorldPoints)for(const auto x:point)hashPacket.real(x);
        for(std::size_t cell=0;cell<cellCount;++cell) {
            hashPacket.real(parent.referenceTetVolumeM3[cell]);
            hashPacket.real(parent.tetrahedronMassKg[cell]);
        }
        parent.parentMassPacketSHA256=hashPacket.finish();
        auto hashSources=spatial_mass_transport_detail::SHA256Builder{};
        hashSources.string("synthetic-source-node-material-donor-mass.v1");
        for(const auto& row:parent.sourceMassContributions) {
            hashSources.u32(row.parentNodeIndex);hashSources.u32(row.materialIndex);
            hashSources.u32(row.donorBodyIndex);hashSources.real(row.massKg);
            for(const auto x:row.donorLocalPosition)hashSources.real(x);
        }
        parent.parentSourceContributionSHA256=hashSources.finish();
        auto hashTopology=spatial_mass_transport_detail::SHA256Builder{};
        hashTopology.string("synthetic-parent-topology.v1");
        for(const auto& tet:parent.tetrahedra)for(const auto node:tet)hashTopology.u32(node);
        parent.parentTopologySHA256=hashTopology.finish();
        auto hashLabels=spatial_mass_transport_detail::SHA256Builder{};
        hashLabels.string("synthetic-parent-labels.v1");
        for(std::size_t cell=0;cell<cellCount;++cell) {
            hashLabels.u32(parent.materialIndices[cell]);hashLabels.u32(parent.donorBodyIndices[cell]);
            hashLabels.u32(parent.anatomicalRegionIndices[cell]);
            hashLabels.u32(parent.stableSourceFaceIdentities[cell]);
        }
        parent.parentLabelSHA256=hashLabels.finish();
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

[[maybe_unused]] PhysicalSkinRegisteredVolumeAssetFiles rewriteSplitMapField(
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


PhysicalSkinRegisteredVolumeAssetFiles rewriteLegacyMapField(
        const PhysicalSkinRegisteredVolumeAssetFiles& files,const std::filesystem::path& root,
        const std::string& name,const std::uint32_t node,const std::size_t column,
        const std::string& value) {
    auto changed=files;changed.combinedNodeMapCSV=root/name;
    std::ifstream input(files.combinedNodeMapCSV);
    std::ofstream output(changed.combinedNodeMapCSV);
    std::string line;assert(std::getline(input,line));
    writeRow(output,physicalSkinCombinedNodeMapCSVHeader(false));
    bool found=false;
    while(std::getline(input,line)) {
        auto fields=physical_skin_child_mass_detail::splitCSV(line);
        assert(fields.size()==42u&&column<fields.size());
        std::uint32_t id=0u;assert(physical_skin_child_mass_detail::parseNumber(fields[0],id));
        if(id==node+1u){fields[column]=value;found=true;}
        writeRow(output,fields);
    }
    assert(found);output.close();input.close();
    pin(changed.combinedNodeMapCSV,changed.expectedSHA256[3]);
    return changed;
}

PhysicalSkinRegisteredVolumeAssetFiles rewriteLabelRegion(
        const PhysicalSkinRegisteredVolumeAssetFiles& files,const std::filesystem::path& root,
        const std::size_t childRow,const std::uint32_t region) {
    auto changed=files;changed.childLabels=root/"bad-semantic-labels.csv";
    std::ifstream input(files.childLabels);std::ofstream output(changed.childLabels);
    std::string line;assert(std::getline(input,line));output<<line<<"\n";
    std::size_t row=0u;
    while(std::getline(input,line)) {
        if(row==childRow) {
            const auto comma=line.find(',',line.find(',',line.find(',')+1u)+1u);
            const auto next=line.find(',',comma+1u);
            assert(comma!=std::string::npos&&next!=std::string::npos);
            line.replace(comma+1u,next-comma-1u,std::to_string(region));
        }
        output<<line<<"\n";++row;
    }
    assert(row>childRow);output.close();input.close();
    pin(changed.childLabels,changed.expectedSHA256[2]);
    return changed;
}

PhysicalSkinRegisteredVolumeAssetFiles reverseFirstElement(
        const PhysicalSkinRegisteredVolumeAssetFiles& files,const std::filesystem::path& root) {
    auto changed=files;changed.tetgenElements=root/"inverted-first-child.ele";
    std::ifstream input(files.tetgenElements);std::ofstream output(changed.tetgenElements);
    std::string line;assert(std::getline(input,line));output<<line<<"\n";
    assert(std::getline(input,line));
    std::istringstream parser(line);std::vector<std::string> fields;
    for(std::string field;parser>>field;)fields.push_back(field);
    assert(fields.size()==6u);
    std::swap(fields[2],fields[3]);
    for(std::size_t i=0;i<fields.size();++i)output<<(i?" ":"")<<fields[i];
    output<<"\n";
    while(std::getline(input,line))output<<line<<"\n";
    output.close();input.close();
    pin(changed.tetgenElements,changed.expectedSHA256[1]);
    return changed;
}

void checkFailureIsAtomic(Fixture& fixture,const PhysicalSkinRegisteredVolumeAssetFiles& files,
                          std::string_view expected);

void checkSourceMassParentSplitControl(
        Fixture& fixture,const std::vector<PhysicalSkinSourceDomainRegistration>& typedDomains,
        const std::filesystem::path& root) {
    fixture.initializeSyntheticParentMass();
    PhysicalSkinRegisteredVolumeAssetFiles files;
    fixture.writeFiles(root/"source-parent-split",files,false);
    assert(files.registrationDomain==PhysicalSkinRegisteredVolumeRegistrationDomain::sourceMassParent);
    auto bundle=fixture.parentBundle;
    PhysicalSkinRegisteredVolumeImportSidecars sidecars;
    PhysicalSkinRegisteredVolumeFileImportReport report;
    std::vector<PhysicalSkinStableDeepAttachmentRequest> attachments;
    std::string error;
    assert(importPhysicalSkinRegisteredVolumeFiles(files,std::string(64u,'0'),157u,0.02,
        0.02,"",0u,0u,fixture.registered,attachments,fixture.poses,bundle,
        sidecars,report,error,std::span<const PhysicalSkinSourceDomainRegistration>(typedDomains)));
    assert(report.registrationDomain==PhysicalSkinRegisteredVolumeRegistrationDomain::sourceMassParent);
    assert(!report.physicalAdmission);
    assert(!report.operationalReferencePullbackVerified&&!report.conformingTissueRefinementVerified);
    assert(!bundle.usesSeparatedOperationalReference&&!bundle.conformingLineageMassRouteRequired);
    assert(bundle.parent.tetrahedra.size()==6u&&bundle.childTetrahedra.size()==9u);
    assert(bundle.childMaterialIndices.size()==9u&&bundle.childStableSourceFaceIdentities.size()==9u);
    assert(bundle.volumeNodeParentTetrahedra.size()==10u);
    assert(bundle.volumeNodeParentTetrahedra[9]==0u);
    assert((bundle.volumeNodeParentBarycentric[9]==std::array<double,4>{0.25,0.25,0.25,0.25}));
    assert(bundle.volumeNodeParentResidualWorldM[9][0]>0.000049&&
           bundle.volumeNodeParentResidualWorldM[9][0]<0.000051);
    assert(bundle.parent.sourceMassContributions.size()>0u);
    assert(physical_skin_registered_volume_integration_detail::sameImmutableParentPacket(
        fixture.parentBundle.parent,bundle.parent));

    std::vector<double> lower(17u,1.0),upper(17u,2000.0);
    lower[15]=900.0;upper[15]=1100.0;lower[16]=850.0;upper[16]=950.0;
    const bool mapped=preparePhysicalSkinChildMassMapping(bundle,lower,upper,0.02,157u,error);
    if(!mapped)std::cerr<<"source-parent split mass mapping failed: "<<error<<"\n";
    assert(mapped);
    const auto& transfer=bundle.spatialMassTransportResult;
    assert(!bundle.exactTopologyIdentityTransportUsed&&!bundle.conformingLineageMassTransportUsed);
    assert(transfer.method==UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport);
    assert(transfer.childMassKg.size()==bundle.childTetrahedra.size());
    assert(transfer.childDominantParentRows.size()==bundle.childTetrahedra.size());
    assert(transfer.groupClosure.size()==2u&&transfer.sourceDonorClosure.size()==4u);
    assert(transfer.maximumParentMassClosureErrorKg<1.0e-12);
    assert(transfer.maximumParentMeasureClosureErrorM3<1.0e-18);
    assert(transfer.maximumSourceDonorMassClosureErrorKg<1.0e-12);
    for(const auto& group:transfer.groupClosure) {
        assert((group.materialIndex==15u||group.materialIndex==16u)&&
               group.donorBodyIndex==0u&&group.anatomicalRegionIndex==42u);
        assert(std::abs(group.parentMassKg-group.childMassKg)<1.0e-12);
        assert(group.childMinimumCapacityKg<=group.parentMassKg+1.0e-12&&
               group.parentMassKg<=group.childMaximumCapacityKg+1.0e-12);
    }
    for(const auto& source:transfer.sourceDonorClosure) {
        assert((source.materialIndex==15u||source.materialIndex==16u)&&source.sourceDonorBodyIndex<2u);
        assert(std::abs(source.parentMassKg-source.childMassKg)<1.0e-12);
    }
    double mappedMass=0.0;
    for(std::size_t child=0;child<transfer.childMassKg.size();++child) {
        const auto material=bundle.childMaterialIndices[child];
        assert(material==15u||material==16u);
        const auto rho=transfer.childImpliedDensityKgM3[child];
        assert(rho>=lower[material]-1.0e-8&&rho<=upper[material]+1.0e-8);
        const auto parent=transfer.childDominantParentRows[child];
        assert(parent<fixture.parentBundle.parent.tetrahedra.size());
        assert(bundle.childDonorBodyIndices[child]==0u&&bundle.childAnatomicalRegionIndices[child]==42u);
        assert(bundle.childStableSourceFaceIdentities[child]==
               fixture.parentBundle.parent.stableSourceFaceIdentities[parent]);
        mappedMass+=transfer.childMassKg[child];
    }
    assert(std::abs(mappedMass-fixture.parentBundle.parent.totalMassKg)<1.0e-12);
    assert(physical_skin_child_mass_mapping_detail::validatePreparedPhysicalSkinChildMassMappingPath(
        bundle,error));
    assert(physical_skin_registered_volume_integration_detail::sameImmutableParentPacket(
        fixture.parentBundle.parent,bundle.parent));
    std::cout<<"source_mass_parent_split_import=PASS parent_cells=6 child_cells=9 interior_nodes=1"
        <<" generic_spatial_transport=PASS edges="<<transfer.transportEdgeCount
        <<" min_density="<<transfer.minimumChildImpliedDensityKgM3
        <<" max_density="<<transfer.maximumChildImpliedDensityKgM3
        <<" parent_mass_error_kg="<<transfer.maximumParentMassClosureErrorKg
        <<" parent_measure_error_m3="<<transfer.maximumParentMeasureClosureErrorM3
        <<" source_donor_closure_rows="<<transfer.sourceDonorClosure.size()<<" source_donor_error_kg="<<transfer.maximumSourceDonorMassClosureErrorKg<<"\n";

    const auto badRegistration=rewriteLegacyMapField(files,root,"bad-source-registration.csv",9u,34u,"0");
    checkFailureIsAtomic(fixture,badRegistration,
        "current-parent residual does not match its recorded distance or barycentric sum");

    const auto badOrientation=reverseFirstElement(files,root);
    checkFailureIsAtomic(fixture,badOrientation,
        "child TetGen cell is nonpositive after exact Float32 emission");

    const auto badRegion=rewriteLabelRegion(files,root,0u,43u);
    auto semanticBundle=fixture.parentBundle;
    PhysicalSkinRegisteredVolumeImportSidecars semanticSidecars;
    PhysicalSkinRegisteredVolumeFileImportReport semanticReport;
    error.clear();
    assert(importPhysicalSkinRegisteredVolumeFiles(badRegion,std::string(64u,'0'),157u,0.02,
        0.02,"",0u,0u,fixture.registered,attachments,fixture.poses,semanticBundle,
        semanticSidecars,semanticReport,error,std::span<const PhysicalSkinSourceDomainRegistration>(typedDomains)));
    assert(semanticBundle.childAnatomicalRegionIndices[0]==43u);
    assert(!preparePhysicalSkinChildMassMapping(semanticBundle,lower,upper,0.02,157u,error));
    assert(error.find("child anatomical/material/donor group has no parent supply")!=std::string::npos);
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


void checkOperationalSpatialReference(const std::filesystem::path& root) {
    std::filesystem::create_directories(root);
    Fixture f;
    auto& p=f.parentBundle.parent;
    p.nodeFrameOwnerBodyIndices.assign(p.referenceWorldPoints.size(),0u);
    p.nodeFrameOwnerLocalPoints=p.referenceWorldPoints;
    p.compiledGlobalNodeIndices.resize(p.referenceWorldPoints.size());
    std::iota(p.compiledGlobalNodeIndices.begin(),p.compiledGlobalNodeIndices.end(),0u);
    p.compiledLumpedNodeMassKg.assign(p.referenceWorldPoints.size(),0.0);
    std::map<std::pair<unsigned,unsigned>,double> shares;
    for(unsigned cell=0;cell<p.tetrahedra.size();++cell){
        const auto tet=p.tetrahedra[cell];
        const auto v=parentReferenceTetVolumeM3(tet,p.referenceWorldPoints);
        const auto mass=v*(p.materialIndices[cell]==15u?1000.0:900.0);
        p.referenceTetVolumeM3.push_back(v);p.tetrahedronMassKg.push_back(mass);p.totalMassKg+=mass;
        for(const auto node:tet){p.compiledLumpedNodeMassKg[node]+=mass*0.25;
            shares[{node,p.materialIndices[cell]}]+=mass*0.25;}
    }
    for(const auto& [key,mass]:shares)
        p.sourceMassContributions.push_back({.parentNodeIndex=key.first,.materialIndex=key.second,
            .donorBodyIndex=0u,.massKg=mass,.donorLocalPosition=p.referenceWorldPoints[key.first]});
    const auto packet=root/"parent-packet.csv";
    {std::ofstream out(packet);out<<"bounded synthetic parent packet fixture\n";}
    pin(packet,p.parentMassPacketSHA256);
    p.parentSourceContributionSHA256=std::string(64u,'4');
    p.parentTopologySHA256=std::string(64u,'5');p.parentLabelSHA256=std::string(64u,'6');
    PhysicalSkinRegisteredVolumeAssetFiles sourceFiles;
    f.writeFiles(root/"source",sourceFiles);
    f.prepareSeparatedOperationalReference(); // 6 source cells -> 9 new cells.
    PhysicalSkinRegisteredVolumeAssetFiles files;
    f.writeFiles(root/"candidate",files,true);
    PhysicalSkinOperationalReferencePullbackAssets proof;
    proof.sourceParentPacketCSV=packet;
    proof.sourceTetgenNode=sourceFiles.tetgenNode;
    proof.sourceTetgenElements=sourceFiles.tetgenElements;proof.sourceLabels=sourceFiles.childLabels;
    std::copy_n(sourceFiles.expectedSHA256.begin(),3u,proof.sourceTetGenSHA256.begin());
    proof.sourceVolumeAsset=root/"source.npz";proof.candidateVolumeAsset=root/"candidate.npz";
    {std::ofstream out(proof.sourceVolumeAsset);out<<"synthetic positive 6-cell source\n";}
    {std::ofstream out(proof.candidateVolumeAsset);out<<"synthetic positive 9-cell candidate\n";}
    pin(proof.sourceVolumeAsset,proof.sourceVolumeSHA256);pin(proof.candidateVolumeAsset,proof.candidateVolumeSHA256);
    proof.sourceProducerReceipt=root/"source-producer.json";
    {std::ofstream out(proof.sourceProducerReceipt);
     out<<"{\"schema\": \"NumiHuman.reduced-parent-boundary-candidate.v1\","
          "\"status\":\"bounded_reduced_boundary_candidate_not_physical_admission\","
          "\"physical_admission\":false,\"all_child_rows_match_parent_row\":true,"
          "\"candidate.npz\":\""<<proof.sourceVolumeSHA256<<"\","
          "\"parent-reduced-boundary.node\":\""<<sourceFiles.expectedSHA256[0]<<"\","
          "\"parent-reduced-boundary.ele\":\""<<sourceFiles.expectedSHA256[1]<<"\","
          "\"parent-reduced-boundary.labels.csv\":\""<<sourceFiles.expectedSHA256[2]<<"\"}\n";}
    proof.candidateProducerReceipt=root/"candidate-producer.json";
    const auto frame=std::string(64u,'0');
    {std::ofstream out(proof.candidateProducerReceipt);
     out<<"{\"schema\": \"numi-human.registered-volume-parent-identity-source-control.v1\","
          "\"physical_admission\":false,\"registration_domain\":\"candidateOperationalReference\","
          "\"candidate_npz_sha256\":\""<<proof.candidateVolumeSHA256<<"\","
          "\"candidate_node_sha256\":\""<<files.expectedSHA256[0]<<"\","
          "\"candidate_element_sha256\":\""<<files.expectedSHA256[1]<<"\","
          "\"candidate_labels_sha256\":\""<<files.expectedSHA256[2]<<"\","
          "\"node_map_sha256\":\""<<files.expectedSHA256[3]<<"\","
          "\""<<(root/"volume/facet-ancestry.csv").string()<<"\":\""<<files.expectedSHA256[4]<<"\","
          "\"coordinate_frame_sha256\":\""<<frame<<"\","
          "\"immutable_source_parent_packet_sha256\":\""<<p.parentMassPacketSHA256<<"\","
          "\"audit\":{\"candidate_npz_sha256\":\""<<proof.candidateVolumeSHA256<<"\","
          "\"coordinate_frame_sha256\":\""<<frame<<"\",\"physical_admission\":false}}\n";}
    pin(proof.sourceProducerReceipt,proof.sourceProducerReceiptSHA256);
    pin(proof.candidateProducerReceipt,proof.candidateProducerReceiptSHA256);
    const auto writeSelf=[&](const std::filesystem::path& path,const std::string& volume,bool bad=false){
        std::ofstream out(path);
        out<<"{\"status\":\"exact_coarse_material_boundary_self_audit\",\"candidate_sha256\":\""<<volume
           <<"\",\"physical_admission\":false,\"skin\":{\"self_intersection_pairs\":"<<(bad?1:0)
           <<"},\"fat\":{\"self_intersection_pairs\":0}}\n";};
    proof.sourceSelfAuditReceipt=root/"source-self.json";
    proof.candidateSelfAuditReceipt=root/"candidate-self.json";
    writeSelf(proof.sourceSelfAuditReceipt,proof.sourceVolumeSHA256);
    writeSelf(proof.candidateSelfAuditReceipt,proof.candidateVolumeSHA256);
    pin(proof.sourceSelfAuditReceipt,proof.sourceSelfAuditReceiptSHA256);
    pin(proof.candidateSelfAuditReceipt,proof.candidateSelfAuditReceiptSHA256);
    files.operationalReferencePullback=proof;
    std::vector<PhysicalSkinStableDeepAttachmentRequest> attachments;
    const std::vector<PhysicalSkinSourceDomainRegistration> domains{
        {"fixture-detail",PhysicalSkinSourceDomainRole::outerSurface,std::string(64u,'1'),9u,1u},
        {"fixture-detail",PhysicalSkinSourceDomainRole::innerSurface,std::string(64u,'2'),9u,1u},
        {"fixture-detail",PhysicalSkinSourceDomainRole::deepSurface,std::string(64u,'3'),9u,1u}};
    std::string error;
    const auto import=[&](const PhysicalSkinRegisteredVolumeAssetFiles& input,
                          PhysicalSkinChildMassMappingBundle& bundle,
                          PhysicalSkinRegisteredVolumeFileImportReport& report){
        PhysicalSkinRegisteredVolumeImportSidecars sidecars;
        return importPhysicalSkinRegisteredVolumeFiles(input,frame,157u,2.0,
            0.0004,"",0u,0u,f.registered,attachments,f.poses,bundle,sidecars,report,error,domains);
    };
    auto bundle=f.parentBundle;PhysicalSkinRegisteredVolumeFileImportReport report;
    assert(!import(files,bundle,report));
    assert(error.find("exact source/candidate cell and label correspondence")!=std::string::npos);
    assert(bundle.childTetrahedra.empty()); // Legacy path stays fail-closed.
    files.operationalReferencePullback->conservativeSpatialReference=true;
    if(!import(files,bundle,report)){std::cerr<<"spatial import rejected "<<error<<"\n";assert(false);}
    assert(report.operationalReferenceSpatialTransportVerified&&!report.operationalReferencePullbackVerified);
    assert(!report.conformingTissueRefinementVerified&&!report.physicalAdmission);
    assert(bundle.usesSeparatedOperationalReference&&bundle.operationalSpatialMassRouteRequired);
    assert(bundle.parent.tetrahedra.size()==6u&&bundle.childTetrahedra.size()==9u);
    assert(bundle.volumeNodeSourceParentRegistrationDistanceM.back()>0.004);
    assert(bundle.volumeNodeOperationalReferenceDistanceM.back()==0.0);
    std::vector<double> lower(17u,0.0),upper(17u,0.0);
    lower[15]=900;upper[15]=1100;lower[16]=850;upper[16]=950;
    const auto receipt=bundle.spatialMassTransportResult.receiptSHA256;
    assert(preparePhysicalSkinChildMassMapping(bundle,lower,upper,2.0,157u,error));
    assert(receipt==bundle.spatialMassTransportResult.receiptSHA256);
    assert(!bundle.exactTopologyIdentityTransportUsed&&!bundle.conformingLineageMassTransportUsed);
    assert(physical_skin_child_mass_mapping_detail::validatePreparedPhysicalSkinChildMassMappingPath(bundle,error));
    const auto originalParent=p;
    assert(physical_skin_registered_volume_integration_detail::sameImmutableParentPacket(bundle.parent,originalParent));
    assert(validatePreparedOperationalSpatialInstallation(bundle,report,2.0,157u,error));
    for(unsigned kind=0u;kind<8u;++kind){
        auto badBundle=bundle;auto badReport=report;
        if(kind==0u)badBundle.usesSeparatedOperationalReference=false;
        if(kind==1u)badBundle.operationalSpatialMassRouteRequired=false;
        if(kind==2u)badReport.operationalReferenceSpatialTransportVerified=false;
        if(kind==3u)badReport.operationalReferenceSpatialTransportVerificationSHA256=std::string(64u,'8');
        if(kind==4u)badReport.maximumSourceParentRegistrationDistanceM=0.000401;
        if(kind==5u)badReport.maximumSpatialMassTransportDistanceM=2.001;
        if(kind==6u)badBundle.childAnatomicalRegionIndices[0]+=1u;
        if(kind==7u)badBundle.spatialMassTransportResult.childMassKg[0]*=1.001;
        assert(!validatePreparedOperationalSpatialInstallation(badBundle,badReport,2.0,157u,error));
    }
    assert(validateOperationalSpatialInstallationMode(true,true,true,true,false,false,true,true,error));
    for(unsigned kind=0u;kind<7u;++kind){
        bool install=true,separated=true,spatial=true,conforming=false,sourceOnly=false,
             preflight=true,preparationOnly=true;
        if(kind==0u)install=false;
        if(kind==1u)separated=false;
        if(kind==2u)spatial=false;
        if(kind==3u)conforming=true;
        if(kind==4u)sourceOnly=true;
        if(kind==5u)preflight=false;
        if(kind==6u)preparationOnly=false;
        assert(!validateOperationalSpatialInstallationMode(true,install,separated,spatial,
                    conforming,sourceOnly,preflight,preparationOnly,error));
    }

    // Cached transport never authorizes changed physical inputs.
    for(unsigned kind=0;kind<7;++kind){
        auto bad=bundle;
        if(kind==0)bad.referenceWorldPoints[9][0]+=0.001;
        if(kind==1)bad.childAnatomicalRegionIndices[0]+=1u;
        if(kind==2)bad.coordinateFrameSHA256=std::string(64u,'9');
        if(kind==3)bad.spatialMassTransportResult.childMassKg[0]*=1.001;
        if(kind==4)bad.operationalSpatialTransportReceiptSHA256=std::string(64u,'8');
        if(kind==5)bad.operationalSpatialMassRouteRequired=false;
        if(kind==6)bad.conformingLineageMassRouteRequired=true;
        const auto before=bad.massMappingPathSHA256;
        assert(!preparePhysicalSkinChildMassMapping(bad,lower,upper,2.0,157u,error));
        assert(bad.massMappingPathSHA256==before);
    }
    const auto expectImportReject=[&](PhysicalSkinRegisteredVolumeAssetFiles bad,const std::string& match){
        auto untouched=f.parentBundle;PhysicalSkinRegisteredVolumeFileImportReport r;r.nodeMap.fileRows=123u;
        assert(!import(bad,untouched,r));
        if(error.find(match)==std::string::npos)std::cerr<<"expected "<<match<<" got "<<error<<"\n";
        assert(error.find(match)!=std::string::npos);
        assert(untouched.childTetrahedra.empty()&&r.nodeMap.fileRows==123u);
    };
    auto bad=files;bad.operationalReferencePullback->candidateVolumeSHA256=std::string(64u,'a');
    expectImportReject(bad,"hash differs");
    bad=files;bad.expectedSHA256[3]=std::string(64u,'b');
    expectImportReject(bad,"hash differs");
    bad=files;bad.operationalReferencePullback->candidateSelfAuditReceipt=root/"bad-self.json";
    writeSelf(bad.operationalReferencePullback->candidateSelfAuditReceipt,proof.candidateVolumeSHA256,true);
    pin(bad.operationalReferencePullback->candidateSelfAuditReceipt,bad.operationalReferencePullback->candidateSelfAuditReceiptSHA256);
    expectImportReject(bad,"nonzero");
    bad=files;bad.operationalReferencePullback->conformingTissueRefinement=PhysicalSkinConformingTissueRefinementAssets{};
    expectImportReject(bad,"mutually exclusive");
    auto infeasible=bundle;infeasible.operationalSpatialMassRouteRequired=false;
    infeasible.operationalSpatialReferenceVerificationSHA256.clear();
    infeasible.operationalSpatialTransportReceiptSHA256.clear();
    for(auto& point:infeasible.referenceWorldPoints)point[0]*=2.0;
    infeasible.currentWorldPoints=infeasible.referenceWorldPoints;
    assert(!preparePhysicalSkinChildMassMapping(infeasible,lower,upper,2.0,157u,error));
    std::cout<<"operational_spatial_reference 6_to_9_cells=PASS historical_vs_operational=PASS"
        <<" certified_cache_reuse=PASS stale_geometry_labels_frame_flows_receipts=PASS"
        <<" authenticated_file_failures_atomic=PASS density_infeasible=PASS physical_admission=NOT_CLAIMED\n";
}


bool runtimeQualityPreflightChecks() {
    using namespace physical_skin_registered_volume_integration_detail;
    PhysicalSkinChildMassMappingBundle bundle;
    bundle.referenceWorldPoints={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    bundle.childTetrahedra={{{0u,1u,2u,3u}}};
    bundle.childMaterialIndices={15u};
    PhysicalSkinRuntimeMeshQualityPreflightReport report;
    std::string error;
    if(!preflightPhysicalSkinRuntimeMeshQuality(bundle,report,error)||
       report.childTetrahedra!=1u||report.normalizedDeterminantBelowFloor!=0u||
       report.signedVolumeAtOrBelowFloor!=0u) {
        std::cerr<<"well-shaped runtime tetrahedron failed the preflight: "<<error<<"\n";
        return false;
    }
    bundle.referenceWorldPoints[3][2]=1.0e-6;
    if(preflightPhysicalSkinRuntimeMeshQuality(bundle,report,error)||
       report.normalizedDeterminantBelowFloor!=1u||report.signedVolumeAtOrBelowFloor!=0u||
       report.firstNormalizedDeterminantFailureByRole[0]!=0u||
       error.find("geometry-not-ready:")!=0u||error.find("first Skin/Fat rows=0/none")==std::string::npos) {
        std::cerr<<"thin runtime tetrahedron did not fail only the normalized determinant gate: "<<error<<"\n";
        return false;
    }
    bundle.referenceWorldPoints={{0,0,0},{1.0e-7,0,0},{0,1.0e-7,0},{0,0,1.0e-7}};
    bundle.childMaterialIndices={16u};
    if(preflightPhysicalSkinRuntimeMeshQuality(bundle,report,error)||
       report.normalizedDeterminantBelowFloor!=0u||report.signedVolumeAtOrBelowFloor!=1u||
       report.firstSignedVolumeFailureByRole[1]!=0u||
       error.find("first Skin/Fat rows=none/0")==std::string::npos) {
        std::cerr<<"small runtime tetrahedron did not fail the absolute volume gate: "<<error<<"\n";
        return false;
    }
    std::cout<<"runtime_mesh_quality_preflight=PASS float32_normalized_determinant=1e-5 "
        <<"absolute_signed_volume=1e-18m3 first_witnesses=PASS geometry_not_ready=PASS\n";
    return true;
}


bool runtimeQualityStopsBeforeNodeMap(const std::filesystem::path& root) {
    Fixture tiny;
    tiny.scaleSyntheticGeometry(1.0e-7);
    const auto directory=root/"runtime-quality-preflight-before-map";
    std::filesystem::create_directories(directory);
    PhysicalSkinRegisteredVolumeAssetFiles files;
    tiny.writeFiles(directory,files);
    files.combinedNodeMapCSV=directory/"intentionally-unavailable-map.csv";
    auto bundle=tiny.parentBundle;
    PhysicalSkinRegisteredVolumeImportSidecars sidecars;
    PhysicalSkinRegisteredVolumeFileImportReport report;
    std::vector<PhysicalSkinStableDeepAttachmentRequest> attachments;
    std::string error;
    if(importPhysicalSkinRegisteredVolumeFiles(files,std::string(64u,'0'),157u,0.02,
       0.02,"",0u,0u,tiny.registered,attachments,tiny.poses,
       bundle,sidecars,report,error)||
       error.find("geometry-not-ready:")!=0u) {
        std::cerr<<"runtime geometry gate did not stop before the unavailable node map: "<<error<<"\n";
        return false;
    }
    std::cout<<"runtime_mesh_quality_import_order=PASS quality_gate_precedes_node_map_import\n";
    return true;
}

int main() {
    assert(runtimeQualityPreflightChecks());
    Fixture fixture;
    const auto root=std::filesystem::temp_directory_path()/
        ("numi-registered-volume-import-"+std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    assert(runtimeQualityStopsBeforeNodeMap(root));
    checkNativeFloat32FrameContract(root);
    checkOperationalSpatialReference(root/"spatial-route");
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

    // Legacy 002->514-style pullbacks retain exact source/candidate row-domain
    // equality, while a proven 002->530->006 route compares the candidate to
    // the separately authenticated refined domain instead.
    PhysicalSkinChildMassMappingBundle sourceDomain,legacyCandidateDomain;
    sourceDomain.referenceWorldPoints.resize(4u);
    sourceDomain.childTetrahedra={{{0u,1u,2u,3u}}};
    sourceDomain.childMaterialIndices={15u};
    sourceDomain.childDonorBodyIndices={66u};
    sourceDomain.childAnatomicalRegionIndices={53u};
    sourceDomain.childStableSourceFaceIdentities={835u};
    legacyCandidateDomain=sourceDomain;
    assert(physical_skin_registered_volume_integration_detail::sameOperationalReferenceCellDomain(
        legacyCandidateDomain,sourceDomain));
    PhysicalSkinChildMassMappingBundle refinedCandidateDomain;
    refinedCandidateDomain.referenceWorldPoints.resize(5u);
    refinedCandidateDomain.childTetrahedra={{{0u,1u,2u,4u}},{{0u,2u,1u,4u}}};
    refinedCandidateDomain.childMaterialIndices={15u,15u};
    refinedCandidateDomain.childDonorBodyIndices={66u,66u};
    refinedCandidateDomain.childAnatomicalRegionIndices={53u,53u};
    refinedCandidateDomain.childStableSourceFaceIdentities={835u,835u};
    PhysicalSkinVerifiedConformingReference refinedDomain;
    refinedDomain.referenceWorldPoints=refinedCandidateDomain.referenceWorldPoints;
    refinedDomain.tetrahedra=refinedCandidateDomain.childTetrahedra;
    refinedDomain.materialIndices=refinedCandidateDomain.childMaterialIndices;
    refinedDomain.geometricDonorBodyIndices=refinedCandidateDomain.childDonorBodyIndices;
    refinedDomain.anatomicalRegionIndices=refinedCandidateDomain.childAnatomicalRegionIndices;
    refinedDomain.stableSourceFaceIds=refinedCandidateDomain.childStableSourceFaceIdentities;
    assert(!physical_skin_registered_volume_integration_detail::sameOperationalReferenceCellDomain(
        refinedCandidateDomain,sourceDomain));
    assert(physical_skin_registered_volume_integration_detail::sameOperationalReferenceCellDomain(
        refinedCandidateDomain,refinedDomain));
    auto wrongRefinedDomain=refinedDomain;wrongRefinedDomain.stableSourceFaceIds[0]=836u;
    assert(!physical_skin_registered_volume_integration_detail::sameOperationalReferenceCellDomain(
        refinedCandidateDomain,wrongRefinedDomain));


    Fixture sourceParentSplitFixture;
    sourceParentSplitFixture.prepareSeparatedOperationalReference();
    sourceParentSplitFixture.scaleSyntheticGeometry(0.01);
    sourceParentSplitFixture.registered.parentRegistrationIsHistoricalProvenanceOnly=false;
    checkSourceMassParentSplitControl(sourceParentSplitFixture,typedDomains,root);

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
        <<" legacy_exact_topology_and_refined_530_candidate_domain_routing=PASS"
        <<" immutable_parent_packet=1 actual_authenticated_chain_separately_checked=1"
        <<" invalid_source_registration=REJECTED semantic_group_without_parent_supply=REJECTED"
        <<" inverted_cell=REJECTED"
        <<" physical_admission=NOT_CLAIMED\n";
}
