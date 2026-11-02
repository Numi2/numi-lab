#include "NumiHumanCurrentFractionalSupportNPZ.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <zlib.h>

using namespace metalrobo::human::coarse_tissue;
namespace fs=std::filesystem;

static const std::string header=
    "record_kind,role,carrier_construction,packet_carrier_row,local_parent_row,"
    "global_source330_tet_row,child_local_row,child_global_row,material,donor,region,"
    "provenance_mode,p1_group_id,projection_csv_row,bary0,bary1,bary2,bary3,weight,"
    "source_x_m,source_y_m,source_z_m,child_centroid_x_m,child_centroid_y_m,"
    "child_centroid_z_m,distance_m\n";

static std::string row(const std::string& kind,const std::string& role,
        const std::uint32_t carrier,const std::uint32_t parentLocal,
        const std::uint32_t parentGlobal,const std::uint32_t childLocal,
        const std::uint32_t childGlobal,const std::uint32_t material,
        const std::uint32_t donor,const std::uint32_t region,
        const std::string& mode,const std::string& p1,const std::string& projection,
        const double distance,const std::string& construction){
    return kind+","+role+","+construction+","+std::to_string(carrier)+","+
        std::to_string(parentLocal)+","+std::to_string(parentGlobal)+","+
        std::to_string(childLocal)+","+std::to_string(childGlobal)+","+
        std::to_string(material)+","+std::to_string(donor)+","+
        std::to_string(region)+","+mode+","+p1+","+projection+
        ",0.25,0.25,0.25,0.25,0.05,0.25,0.25,0.25,0.25,0.25,0.25,"+
        std::to_string(distance)+"\n";
}

static fs::path writeSkin(const std::string& name,const bool extra,
        const double extraDistance=0.0,const std::uint32_t extraRegion=43u,
        const bool duplicateExtra=false,const bool omitPrimary=false){
    const auto path=fs::temp_directory_path()/("h706-"+name+"-skin.csv");
    std::ofstream out(path,std::ios::binary|std::ios::trunc);out<<header;
    if(!omitPrimary)
        out<<row("carrier","Skin",0,0,0,0,0,15,43,43,
            "p1_exact_tuple_slot","0","-1",0.0,"inherited_090_quadrature");
    if(extra){
        const auto e=row("carrier_child_slot","Skin",0,0,0,1,2,15,43,extraRegion,
            "CurrentCarrierPointProjection","-1","-1",extraDistance,
            "current_source_carrier_edge_support");
        out<<e;if(duplicateExtra)out<<e;
    }
    return path;
}

static fs::path writeFat(){
    const auto path=fs::temp_directory_path()/"h706-valid-fat.csv";
    std::ofstream out(path,std::ios::binary|std::ios::trunc);out<<header;
    out<<row("carrier","Fat",0,0,1,0,1,16,42,42,
        "p1_exact_tuple_slot","1","-1",0.0,"inherited_090_quadrature");
    return path;
}

struct Fixture {
    ConservativeSpatialMassTransportInput input{};
    ConservativeSpatialChildSemanticSupportData support;
    CurrentSourceCarrierQuadratureData carrier;
    std::vector<std::uint32_t> skinRows{0u,2u},fatRows{1u};
    std::array<SpatialMassTransportPoint,2> parentCentroids{{{0.25,0.25,0.25},{0.25,0.25,0.25}}};
    std::array<double,2> parentVolumes{1.0/6.0,1.0/6.0},parentMass{0.5,0.5};
    std::array<std::uint32_t,2> parentMaterial{15u,16u},parentDonor{43u,42u},parentRegion{43u,42u};
    std::array<SpatialMassTransportPoint,3> childCentroids{{{0.25,0.25,0.25},{0.25,0.25,0.25},{0.25,0.25,0.25}}};
    std::array<double,3> childVolumes{1e-3,1e-3,1e-3};
    std::array<std::uint32_t,3> childMaterial{15u,16u,15u},childDonor{43u,42u,43u},childRegion{43u,42u,43u};

    Fixture(){
        input.parentCentroidsWorld=parentCentroids;input.parentReferenceVolumeM3=parentVolumes;
        input.parentMassKg=parentMass;input.parentMaterialIndices=parentMaterial;
        input.parentDonorBodyIndices=parentDonor;input.parentAnatomicalRegionIndices=parentRegion;
        input.childCentroidsWorld=childCentroids;input.childReferenceVolumeM3=childVolumes;
        input.childMaterialIndices=childMaterial;input.childDonorBodyIndices=childDonor;
        input.childAnatomicalRegionIndices=childRegion;input.maximumTransportDistanceM=0.019353;

        support.childOffsets={0u,1u,2u,3u};support.slotGroupIndices={0u,1u,0u};
        support.volumeFractions={1.0,1.0,1.0};
        support.childAdmissibleGroupOffsets=support.childOffsets;
        support.childAdmissibleGroupIndices=support.slotGroupIndices;
        support.groupMaterialIndices={15u,16u};support.donorBodyIndices={43u,42u};
        support.anatomicalRegionIndices={43u,42u};support.derivedGroupIdentityIDs={1u,2u};
        support.parentStableSourceFaceIdentities={100u,101u};
        support.sourceParentTetOffsets={0u,1u,2u};support.sourceParentTetRows={0u,1u};
        support.sourceStableFaceOffsets={0u,1u,2u};support.sourceStableFaceRows={100u,101u};
        support.cellAdmissibleRegionOffsets={0u,1u,2u,3u};
        support.cellAdmissibleRegionIds={43u,42u,43u};
        support.admissibilitySourceSupportSHA256=std::string(64u,'b');
        support.admissibilityMaximumSourceRegionHops=1u;support.typedSupportSHA256=std::string(64u,'c');

        carrier.parentOffsets={0u,1u,2u};carrier.globalSourceTetRows={0u,1u};
        carrier.sourceTetNodeRows={{{0u,1u,2u,3u}},{{4u,1u,2u,3u}}};
        const std::array<std::array<double,3>,4> tet{{{{0,0,0}},{{1,0,0}},{{0,1,0}},{{0,0,1}}}};
        carrier.parentReferenceTetWorldVertices={tet,tet};
        carrier.carrierWeights={0.05,0.05};
        carrier.carrierBarycentric={{0.25,0.25,0.25,0.25},{0.25,0.25,0.25,0.25}};
        carrier.witnessChildRows={0u,1u};carrier.witnessChildCentroidDistancesM={0.0,0.0};
        carrier.carrierRoleLocalRows={0u,0u};carrier.carrierRoleMaterialIndices={15u,16u};
        carrier.carrierRoleLocalChildRows={0u,0u};carrier.carrierRoleLocalParentRows={0u,0u};
        carrier.sourceCarrierPacketSHA256=std::string(64u,'d');
    }

    bool load(const fs::path& path,const std::uint32_t material,
              std::vector<CurrentCarrierWitnessEvidenceRow>& rows,
              CurrentCarrierWitnessCSVSummary* summary=nullptr){
        std::string sha,error;CurrentCarrierWitnessCSVSummary local;
        if(!physical_skin_child_mass_detail::sha256File(path,sha,error)||
           !loadCurrentCarrierWitnessCSV(path,sha,material,rows,local,error))return false;
        if(summary)*summary=local;return true;
    }

    bool bind(const std::vector<CurrentCarrierWitnessEvidenceRow>& rows,
              const std::string& evidence=std::string(64u,'f'),
              const std::vector<std::uint32_t>& skin={0u,2u},
              const std::vector<std::uint32_t>& fat={1u}){
        std::string error;
        return bindCurrentCarrierWitnessAllowlist(support,input,carrier,rows,{},
            std::string(64u,'e'),evidence,error,skin,fat);
    }
};


static fs::path gzipCopy(const fs::path& source,const std::string& name){
    const auto target=fs::temp_directory_path()/("h706-"+name+".csv.gz");
    std::ifstream in(source,std::ios::binary);
    gzFile out=gzopen(target.string().c_str(),"wb6");
    if(!in||!out)return {};
    std::array<char,65536> buffer{};
    while(in){
        in.read(buffer.data(),static_cast<std::streamsize>(buffer.size()));
        const auto n=in.gcount();
        if(n>0&&gzwrite(out,buffer.data(),static_cast<unsigned>(n))!=n){
            (void)gzclose(out);return {};
        }
    }
    if(!in.eof()||gzclose(out)!=Z_OK)return {};
    return target;
}
static std::string pairedSHA(const fs::path& skin,const fs::path& fat){
    std::string sh,fh,error;
    if(!physical_skin_child_mass_detail::sha256File(skin,sh,error)||
       !physical_skin_child_mass_detail::sha256File(fat,fh,error))return {};
    spatial_mass_transport_detail::SHA256Builder digest;
    digest.string("numi.human.current-paired-1046c-carrier-witness-csvs.v1");
    digest.string(sh);digest.string(fh);return digest.finish();
}
static bool streamBind(Fixture& fixture,const fs::path& skin,const fs::path& fat,
                       std::uint64_t* rowCount=nullptr){
    std::string error;const auto evidence=pairedSHA(skin,fat);
    if(evidence.empty())return false;
    CurrentCarrierWitnessAllowlistStreamState state;
    if(!state.begin(fixture.support,fixture.input,fixture.carrier,{},
            std::string(64u,'e'),evidence,error,fixture.skinRows,fixture.fatRows))return false;
    const std::function<bool(const CurrentCarrierWitnessEvidenceRow&,std::string&)> consume=
        [&state](const CurrentCarrierWitnessEvidenceRow& row,std::string& rowError){
            return state.consume(row,rowError);
        };
    CurrentCarrierWitnessCSVSummary skinSummary,fatSummary;
    std::string skinSHA,fatSHA;
    if(!physical_skin_child_mass_detail::sha256File(skin,skinSHA,error)||
       !physical_skin_child_mass_detail::sha256File(fat,fatSHA,error))return false;
    if(!loadCurrentCarrierWitnessCSVEach(skin,skinSHA,15u,skinSummary,consume,error)||
       !loadCurrentCarrierWitnessCSVEach(fat,fatSHA,16u,fatSummary,consume,error)||
       skinSummary.carrierRecordCount==0u||fatSummary.carrierRecordCount==0u||
       !state.finish(error))return false;
    if(rowCount)*rowCount=state.rowCount();
    return true;
}

int main(){
    const auto skinPath=writeSkin("valid",true);
    const auto fatPath=writeFat();
    Fixture valid;
    std::vector<CurrentCarrierWitnessEvidenceRow> skin,fat;
    CurrentCarrierWitnessCSVSummary summary;
    if(!valid.load(skinPath,15u,skin,&summary)||summary.carrierRecordCount!=1u||
       summary.additionalSupportRecordCount!=1u||skin.size()!=2u){
        std::cerr<<"valid primary/extra sidecar failed parse\n";return 1;
    }
    if(!valid.load(fatPath,16u,fat)||fat.size()!=1u){
        std::cerr<<"valid Fat primary sidecar failed parse\n";return 2;
    }
    skin.insert(skin.end(),fat.begin(),fat.end());
    const auto beforeOffsets=valid.support.childOffsets;
    const auto beforeSlots=valid.support.slotGroupIndices;
    const auto beforeFractions=valid.support.volumeFractions;
    if(!valid.bind(skin)){
        std::cerr<<"valid extra slot failed exact binding\n";return 3;
    }
    if(valid.support.childOffsets!=beforeOffsets||valid.support.slotGroupIndices!=beforeSlots||
       valid.support.volumeFractions!=beforeFractions){
        std::cerr<<"extra slot changed P1 preferences\n";return 4;
    }
    if(valid.support.childAdmissibleGroupOffsets!=std::vector<std::uint32_t>{0u,1u,2u,3u}||
       valid.support.childAdmissibleGroupIndices!=std::vector<std::uint32_t>{0u,1u,0u}){
        std::cerr<<"extra slot failed to add its exact group to the mapped receiver\n";return 5;
    }

    const auto skinGzip=gzipCopy(skinPath,"stream-parity-skin");
    const auto fatGzip=gzipCopy(fatPath,"stream-parity-fat");
    Fixture vectorGzip,streamed;
    std::vector<CurrentCarrierWitnessEvidenceRow> gzSkin,gzFat;
    if(!vectorGzip.load(skinGzip,15u,gzSkin)||!vectorGzip.load(fatGzip,16u,gzFat))return 14;
    gzSkin.insert(gzSkin.end(),gzFat.begin(),gzFat.end());
    const auto pairedEvidence=pairedSHA(skinGzip,fatGzip);
    if(pairedEvidence.empty()||!vectorGzip.bind(gzSkin,pairedEvidence))return 15;
    std::uint64_t streamedRows=0;
    if(!streamBind(streamed,skinGzip,fatGzip,&streamedRows)||streamedRows!=3u||
       streamed.support.childAdmissibleGroupOffsets!=vectorGzip.support.childAdmissibleGroupOffsets||
       streamed.support.childAdmissibleGroupIndices!=vectorGzip.support.childAdmissibleGroupIndices||
       streamed.support.volumeFractions!=vectorGzip.support.volumeFractions||
       streamed.support.admissibleGroupEvidenceSHA256!=vectorGzip.support.admissibleGroupEvidenceSHA256){
        std::cerr<<"streamed gzip binder differs from legacy full-vector binder\\n";return 16;
    }
    auto rejectWithoutPublication=[&](const fs::path& badSkin,const fs::path& goodFat){
        Fixture rejected;
        const auto beforeOffsets=rejected.support.childAdmissibleGroupOffsets;
        const auto beforeIndices=rejected.support.childAdmissibleGroupIndices;
        const auto beforeHash=rejected.support.admissibleGroupEvidenceSHA256;
        if(streamBind(rejected,badSkin,goodFat))return false;
        return rejected.support.childAdmissibleGroupOffsets==beforeOffsets&&
               rejected.support.childAdmissibleGroupIndices==beforeIndices&&
               rejected.support.admissibleGroupEvidenceSHA256==beforeHash;
    };
    const auto duplicateStreamPlain=writeSkin("stream-duplicate",true,0.0,43u,true);
    const auto duplicateStreamGzip=gzipCopy(duplicateStreamPlain,"stream-duplicate");
    if(!rejectWithoutPublication(duplicateStreamGzip,fatGzip))return 17;
    const auto missingStreamPlain=writeSkin("stream-missing",false,0.0,43u,false,true);
    const auto missingStreamGzip=gzipCopy(missingStreamPlain,"stream-missing");
    if(!rejectWithoutPublication(missingStreamGzip,fatGzip))return 18;
    const auto wrongRegionPlain=writeSkin("stream-region",true,0.0,44u);
    const auto wrongRegionGzip=gzipCopy(wrongRegionPlain,"stream-region");
    if(!rejectWithoutPublication(wrongRegionGzip,fatGzip))return 19;
    const auto badClaimPath=fs::temp_directory_path()/"h706-stream-bad-claim-skin.csv";
    {
        std::ofstream out(badClaimPath,std::ios::binary|std::ios::trunc);out<<header;
        out<<row("carrier","Skin",0,0,0,0,0,15,43,43,
                 "p1_exact_tuple_slot","0","-1",0.01,"inherited_090_quadrature");
    }
    if(!rejectWithoutPublication(badClaimPath,fatPath))return 21;
    const auto truncatedGzip=fs::temp_directory_path()/"h706-stream-truncated.csv.gz";
    fs::copy_file(skinGzip,truncatedGzip,fs::copy_options::overwrite_existing);
    fs::resize_file(truncatedGzip,fs::file_size(truncatedGzip)-4u);
    if(!rejectWithoutPublication(truncatedGzip,fatGzip))return 20;

    const auto wrongMapPath=writeSkin("wrong-map",true);Fixture wrongMap;
    std::vector<CurrentCarrierWitnessEvidenceRow> wrongMapRows,wrongMapFat;
    if(!wrongMap.load(wrongMapPath,15u,wrongMapRows)||!wrongMap.load(fatPath,16u,wrongMapFat))return 6;
    wrongMapRows.insert(wrongMapRows.end(),wrongMapFat.begin(),wrongMapFat.end());
    if(wrongMap.bind(wrongMapRows,std::string(64u,'f'),{0u,1u},{1u})){
        std::cerr<<"wrong role-local child map accepted\n";return 7;
    }

    const auto farPath=writeSkin("far",true,0.1);Fixture far;
    std::vector<CurrentCarrierWitnessEvidenceRow> farRows,farFat;
    if(!far.load(farPath,15u,farRows)||!far.load(fatPath,16u,farFat))return 8;
    farRows.insert(farRows.end(),farFat.begin(),farFat.end());
    if(far.bind(farRows)){std::cerr<<"out-of-radius extra slot accepted\n";return 9;}

    const auto regionPath=writeSkin("region",true,0.0,44u);Fixture region;
    std::vector<CurrentCarrierWitnessEvidenceRow> regionRows,regionFat;
    if(!region.load(regionPath,15u,regionRows)||!region.load(fatPath,16u,regionFat))return 10;
    regionRows.insert(regionRows.end(),regionFat.begin(),regionFat.end());
    if(region.bind(regionRows)){std::cerr<<"wrong source tuple/region extra slot accepted\n";return 11;}

    const auto duplicatePath=writeSkin("duplicate",true,0.0,43u,true);Fixture duplicate;
    std::vector<CurrentCarrierWitnessEvidenceRow> duplicateRows;
    if(duplicate.load(duplicatePath,15u,duplicateRows)){
        std::cerr<<"duplicate carrier/child support pair accepted\n";return 12;
    }
    const auto missingPrimaryPath=writeSkin("missing",false,0.0,43u,false,true);Fixture missing;
    std::vector<CurrentCarrierWitnessEvidenceRow> missingRows;
    if(missing.load(missingPrimaryPath,15u,missingRows)){
        std::cerr<<"missing primary carrier row accepted\n";return 13;
    }

    for(const auto& p:{skinPath,fatPath,wrongMapPath,farPath,regionPath,duplicatePath,missingPrimaryPath,
                       skinGzip,fatGzip,duplicateStreamPlain,duplicateStreamGzip,
                       missingStreamPlain,missingStreamGzip,wrongRegionPlain,wrongRegionGzip,
                       badClaimPath,truncatedGzip})
        fs::remove(p);
    std::cout<<"PASS legacy-vs-streamed full-vector/gzip binding: exact CSR/hash parity, duplicate/missing-primary/wrong-one-hop/bad-distance/truncated rejection without publication\n";
    return 0;
}
