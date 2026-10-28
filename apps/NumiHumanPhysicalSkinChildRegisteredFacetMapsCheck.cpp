#include "NumiHumanPhysicalSkinChildRegisteredFacetMaps.hpp"
#include <cassert>
#include <iostream>
#include <sstream>

using namespace metalrobo::human::coarse_tissue;
using Key=std::array<std::uint32_t,3>;

namespace {
struct Fixture {
    PhysicalSkinChildMassMappingBundle bundle;
    std::vector<PhysicalSkinAuthoredHeadWallRow> walls;
    Fixture() {
        bundle.currentWorldPoints={{0,0,2},{1,0,2},{0,1,2},
                                   {0,0,1},{1,0,1},{0,1,1},
                                   {0,0,0},{1,0,0},{0,1,0}};
        bundle.referenceWorldPoints=bundle.currentWorldPoints;
        bundle.childTetrahedra={{3,4,5,0},{4,5,0,1},{5,0,1,2},
                               {6,7,8,3},{7,8,3,4},{8,3,4,5}};
        bundle.childMaterialIndices={15,15,15,16,16,16};
        bundle.parent.currentWorldPoints=bundle.currentWorldPoints;
        bundle.parent.referenceWorldPoints=bundle.currentWorldPoints;
        bundle.parent.tetrahedra=bundle.childTetrahedra;
        bundle.parent.materialIndices=bundle.childMaterialIndices;
        bundle.boundaryMap.volumeNodeCount=9u;
        bundle.boundaryMap.outer={0,1,2};bundle.boundaryMap.skinInner={3,4,5};
        bundle.boundaryMap.fatDeep={6,7,8};
        std::array<RegisteredSurface*,3> surfaces{
            &bundle.outerSkin,&bundle.independentSkinInner,&bundle.independentFatDeep};
        for(unsigned role=0;role<3;++role) {
            auto& surface=*surfaces[role];surface.faces={{0,1,2}};
            surface.sourceFaceIndices={101u+role};
            for(unsigned row=0;row<3;++row)surface.worldPoints.push_back(bundle.currentWorldPoints[3u*role+row]);
        }
        std::map<Key,std::vector<std::uint32_t>> faces;
        for(std::size_t cell=0;cell<bundle.childTetrahedra.size();++cell) {
            assert(physical_skin_registered_volume_import_detail::signedSixVolume(
                bundle.currentWorldPoints,bundle.childTetrahedra[cell])>0.0);
            for(unsigned omitted=0;omitted<4u;++omitted) {
                Key key{};unsigned at=0;
                for(unsigned c=0;c<4u;++c)if(c!=omitted)key[at++]=bundle.childTetrahedra[cell][c];
                std::sort(key.begin(),key.end());faces[key].push_back(static_cast<std::uint32_t>(cell));
            }
        }
        for(const auto& [key,uses]:faces)if(uses.size()==1u&&key!=Key{0,1,2}&&key!=Key{6,7,8}) {
            const auto cell=uses[0];PhysicalSkinAuthoredHeadWallRow wall;
            wall.stableWallFace=500u+static_cast<std::uint32_t>(walls.size());
            wall.childTet=cell;wall.childNodes=key;wall.parentTet=cell;
            wall.material=bundle.childMaterialIndices[cell];wall.parentNodes=key;walls.push_back(wall);
        }
        assert(walls.size()==12u);
    }
};
std::string serializeWalls(const Fixture& fixture) {
    std::ostringstream output;
    const auto write=[&](const std::vector<std::string>& fields) {
        for(std::size_t i=0;i<fields.size();++i) {
            if(i)output<<',';
            if(fields[i].find_first_of(",\"")!=std::string::npos) {
                output<<'"';for(char c:fields[i]){if(c=='"')output<<'"';output<<c;}output<<'"';
            }else output<<fields[i];
        }
        output<<"\r\n";
    };
    write(physicalSkinRegisteredFacetCSVHeader());
    for(const auto& wall:fixture.walls) {
        std::vector<std::string> fields(24u,"-1");
        const bool skin=wall.material==15u;
        fields[0]=skin?"skin":"fat";fields[1]="0";fields[2]=skin?"3":"6";
        fields[3]=skin?"donor_skin_OI_headwall":"donor_fat_ID_headwall";
        fields[4]=std::to_string(wall.stableWallFace);fields[10]="5";
        fields[13]="[1,2,3]";fields[14]=std::to_string(wall.parentTet);
        fields[15]=std::to_string(wall.material);
        fields[16]="["+std::to_string(wall.parentNodes[0])+","+std::to_string(wall.parentNodes[1])+","+
                   std::to_string(wall.parentNodes[2])+"]";
        fields[17]=std::to_string(wall.childTet+1u);fields[18]=std::to_string(wall.childTet);
        for(unsigned c=0;c<3u;++c)fields[19u+c]=std::to_string(wall.childNodes[c]+1u);
        fields[22]="[1,149,1]";fields[23]="[98,99,100]";write(fields);
    }
    return output.str();
}
void rejectCSV(std::string text,std::string_view expected) {
    std::istringstream input(text);std::vector<PhysicalSkinAuthoredHeadWallRow> rows(1);
    rows[0].stableWallFace=98765u;PhysicalSkinRegisteredFacetCSVReport report;report.fileRows=76543u;
    std::string error;
    assert(!readPhysicalSkinAuthoredHeadWallsCSV(input,6u,9u,rows,report,error));
    assert(rows.size()==1u&&rows[0].stableWallFace==98765u&&report.fileRows==76543u);
    if(error.find(expected)==std::string::npos){std::cerr<<expected<<" vs "<<error<<'\n';std::abort();}
}
void reject(Fixture fixture,std::string_view expected) {
    fixture.bundle.childRegisteredLayerFacetSurfaceFaces={76543u};
    PhysicalSkinRegisteredFacetMapReport report;report.authoredHeadWallFaces=12345u;
    std::string error;
    assert(!preparePhysicalSkinRegisteredFacetMaps(fixture.bundle,fixture.walls,15u,16u,report,error));
    assert(fixture.bundle.childRegisteredLayerFacetSurfaceFaces==std::vector<std::uint32_t>{76543u});
    assert(report.authoredHeadWallFaces==12345u);
    if(error.find(expected)==std::string::npos) {
        std::cerr<<"expected "<<expected<<", got "<<error<<'\n';std::abort();
    }
}
}

int main() {
    Fixture fixture;PhysicalSkinRegisteredFacetMapReport report;std::string error;
    assert(preparePhysicalSkinRegisteredFacetMaps(fixture.bundle,fixture.walls,15u,16u,report,error));
    assert((report.layerFaceUses==std::array<std::size_t,3>{1,2,1}));
    assert(report.authoredHeadWallFaces==12u&&report.internalFacets==5u);
    assert(fixture.bundle.childRegisteredLayerFacetSurfaceFaces.size()==24u);
    assert(std::count(fixture.bundle.childAuthoredHeadSidewallFacetFlags.begin(),
        fixture.bundle.childAuthoredHeadSidewallFacetFlags.end(),1u)==12);
    const auto csv=serializeWalls(fixture);std::istringstream csvInput(csv);
    std::vector<PhysicalSkinAuthoredHeadWallRow> parsedWalls;PhysicalSkinRegisteredFacetCSVReport csvReport;
    assert(readPhysicalSkinAuthoredHeadWallsCSV(csvInput,6u,9u,parsedWalls,csvReport,error));
    assert(parsedWalls.size()==12u&&csvReport.authoredWallRows==12u&&csvReport.fileRows==12u);
    auto parsedFixture=Fixture{};
    assert(preparePhysicalSkinRegisteredFacetMaps(parsedFixture.bundle,parsedWalls,15u,16u,report,error));
    rejectCSV("bad_header\n","header");
    rejectCSV(csv+csv.substr(csv.find("\n")+1),"repeats a child facet");
    auto badCSV=csv;const auto quote=badCSV.find("\"[1,2,3]\"");assert(quote!=std::string::npos);
    badCSV.replace(quote,9u,"\"[1,2,3]x");rejectCSV(badCSV,"quoting/width");
    auto wrongParent=Fixture{};wrongParent.walls[0].parentNodes={0,0,0};
    rejectCSV(serializeWalls(wrongParent),"lacks exact");
    auto bad=Fixture{};bad.walls.pop_back();reject(bad,"no exact registered");
    bad=Fixture{};bad.walls[0].parentNodes[0]=8u;reject(bad,"not a face");
    bad=Fixture{};bad.walls[1].stableWallFace=bad.walls[0].stableWallFace;reject(bad,"duplicate");
    bad=Fixture{};bad.walls[0].parentTet=NM_INVALID_INDEX;reject(bad,"missing/duplicate");
    bad=Fixture{};bad.bundle.currentWorldPoints[bad.walls[0].childNodes[0]][0]+=0.0001;
    reject(bad,"no longer matches");
    bad=Fixture{};bad.bundle.independentSkinInner.faces={{0,0,2}};reject(bad,"repeated/degenerate");
    bad=Fixture{};bad.bundle.independentFatDeep.sourceFaceIndices[0]=NM_INVALID_INDEX;
    reject(bad,"repeated/degenerate");
    bad=Fixture{};bad.bundle.childTetrahedra.push_back(bad.bundle.childTetrahedra[0]);
    bad.bundle.childMaterialIndices.push_back(15u);reject(bad,"nonmanifold");
    bad=Fixture{};std::swap(bad.bundle.childTetrahedra[1][0],bad.bundle.childTetrahedra[1][1]);
    reject(bad,"equally oriented");
    bad=Fixture{};bad.walls[0].material=16u;reject(bad,"missing/duplicate");
    std::cout<<"physical_skin_registered_facet_maps=PASS positive_prism_cells=6 exact_O_I_D_uses=1,2,1"
        <<" retained_wall_faces=12 corruptions_rejected=10 failure_transaction_preserved=1"
        <<" csv_quoted_ancestry_roundtrip=PASS csv_corruptions_rejected=4 full_scene_admission=NOT_CLAIMED\n";
}
