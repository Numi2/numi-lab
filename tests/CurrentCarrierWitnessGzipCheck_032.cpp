#include "NumiHumanCurrentFractionalSupportNPZ.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <zlib.h>

using namespace metalrobo::human::coarse_tissue;

namespace {
constexpr std::string_view header =
    "record_kind,role,carrier_construction,packet_carrier_row,local_parent_row,"
    "global_source330_tet_row,child_local_row,child_global_row,material,donor,region,"
    "provenance_mode,p1_group_id,projection_csv_row,bary0,bary1,bary2,bary3,weight,"
    "source_x_m,source_y_m,source_z_m,child_centroid_x_m,child_centroid_y_m,"
    "child_centroid_z_m,distance_m\n";

std::string row(std::string_view kind,std::string_view construction,
                std::string_view packet,std::string_view child,
                std::string_view mode,std::string_view p1,
                std::string_view projection,std::string_view leadingZeros={}){
    std::string out(kind);out+=",Skin,";out+=construction;out+=",";
    out+=leadingZeros;out+=packet;
    out+=",0,0,0,";out+=child;
    out+=",15,69,11,";out+=mode;out+=",";out+=p1;out+=",";out+=projection;
    out+=",0.25,0.25,0.25,0.25,1,0,0,0,0,0,0,0\n";
    return out;
}

std::string validCSV(){
    std::string out(header);
    out+=row("carrier","source_tetra_anchor_refinement_1046d","0","0",
             "CurrentCarrierPointProjection","-1","-1");
    out+=row("carrier","source_tetra_anchor_refinement_1046d","1","1",
             "CurrentCarrierPointProjection","-1","-1");
    out+=row("carrier_child_slot","current_source_carrier_edge_support","0","2",
             "CurrentCarrierPointProjection","-1","-1");
    return out;
}

bool writeGzip(const std::filesystem::path& path,std::string_view bytes){
    gzFile f=gzopen(path.string().c_str(),"wb6");
    if(!f)return false;
    std::size_t offset=0;
    while(offset<bytes.size()){
        const auto n=std::min<std::size_t>(bytes.size()-offset,1u<<20u);
        const int written=gzwrite(f,bytes.data()+offset,static_cast<unsigned int>(n));
        if(written<=0||static_cast<std::size_t>(written)!=n){(void)gzclose(f);return false;}
        offset+=n;
    }
    return gzclose(f)==Z_OK;
}

bool sha(const std::filesystem::path& p,std::string& value){
    std::string error;
    return physical_skin_child_mass_detail::sha256File(p,value,error);
}

bool load(const std::filesystem::path& path,std::uint32_t role,
          std::vector<CurrentCarrierWitnessEvidenceRow>& rows,
          CurrentCarrierWitnessCSVSummary& summary,
          const CurrentCarrierWitnessCSVLimits& limits,
          std::string& error,std::string_view overrideSHA={}){
    std::string digest;
    if(!sha(path,digest)){error="fixture SHA failed";return false;}
    if(!overrideSHA.empty())digest=std::string(overrideSHA);
    return loadCurrentCarrierWitnessCSV(path,digest,role,rows,summary,error,limits);
}

bool sameRows(const std::vector<CurrentCarrierWitnessEvidenceRow>& a,
              const std::vector<CurrentCarrierWitnessEvidenceRow>& b){
    if(a.size()!=b.size())return false;
    for(std::size_t i=0;i<a.size();++i){
        const auto& x=a[i];const auto& y=b[i];
        if(x.roleMaterial!=y.roleMaterial||x.roleLocalCarrierRow!=y.roleLocalCarrierRow||
           x.sourceParentLocalRow!=y.sourceParentLocalRow||x.sourceParentRow!=y.sourceParentRow||
           x.roleLocalChildRow!=y.roleLocalChildRow||x.childRow!=y.childRow||
           x.material!=y.material||x.donor!=y.donor||x.region!=y.region||x.mode!=y.mode||
           x.projectionCSVRow!=y.projectionCSVRow||x.declaredP1GroupID!=y.declaredP1GroupID||
           x.claimedBarycentric!=y.claimedBarycentric||x.claimedWeight!=y.claimedWeight||
           x.claimedSourcePointWorld!=y.claimedSourcePointWorld||
           x.claimedChildCentroidWorld!=y.claimedChildCentroidWorld||
           x.claimedDistanceM!=y.claimedDistanceM||
           x.carrierConstruction!=y.carrierConstruction||x.hasCSVProof!=y.hasCSVProof||
           x.extraChildSlot!=y.extraChildSlot)return false;
    }
    return true;
}

bool expectPass(const std::filesystem::path& p,const CurrentCarrierWitnessCSVLimits& limits,
                std::size_t carriers,std::size_t slots){
    std::vector<CurrentCarrierWitnessEvidenceRow> rows;
    CurrentCarrierWitnessCSVSummary summary;std::string error;
    if(!load(p,15u,rows,summary,limits,error)){
        std::cerr<<"expected PASS "<<p.filename()<<": "<<error<<"\n";return false;
    }
    return summary.carrierRecordCount==carriers&&
           summary.additionalSupportRecordCount==slots&&rows.size()==carriers+slots;
}
bool expectFail(const std::filesystem::path& p,const CurrentCarrierWitnessCSVLimits& limits,
                std::string_view expectedError={}){
    std::vector<CurrentCarrierWitnessEvidenceRow> rows;
    CurrentCarrierWitnessCSVSummary summary;std::string error;
    if(load(p,15u,rows,summary,limits,error)){
        std::cerr<<"expected FAIL "<<p.filename()<<"\n";return false;
    }
    if(!expectedError.empty()&&error.find(expectedError)==std::string::npos){
        std::cerr<<"unexpected failure "<<p.filename()<<": "<<error<<"\n";return false;
    }
    return true;
}
bool writePlain(const std::filesystem::path& p,std::string_view bytes){
    std::ofstream f(p,std::ios::binary);
    f.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(f);
}
bool corruptLastByte(const std::filesystem::path& p){
    std::fstream f(p,std::ios::binary|std::ios::in|std::ios::out);
    f.seekg(-1,std::ios::end);char c=0;f.read(&c,1);
    if(!f)return false;
    c=static_cast<char>(c^0x55);
    f.seekp(-1,std::ios::end);f.write(&c,1);
    return static_cast<bool>(f);
}
}

int main(int argc,char** argv){
    if(argc!=2)return 2;
    const std::filesystem::path dir=argv[1];
    std::filesystem::create_directories(dir);
    CurrentCarrierWitnessCSVLimits defaults;
    const auto bytes=validCSV();
    const auto plain=dir/"valid.csv",gzip=dir/"valid.csv.gz";
    if(!writePlain(plain,bytes)||!writeGzip(gzip,bytes))return 3;
    if(!expectPass(plain,defaults,2u,1u)||!expectPass(gzip,defaults,2u,1u))return 4;

    std::vector<CurrentCarrierWitnessEvidenceRow> plainRows,gzipRows;
    CurrentCarrierWitnessCSVSummary plainSummary,gzipSummary;std::string error;
    if(!load(plain,15u,plainRows,plainSummary,defaults,error)||
       !load(gzip,15u,gzipRows,gzipSummary,defaults,error)||
       !sameRows(plainRows,gzipRows)||
       plainSummary.carrierRecordCount!=gzipSummary.carrierRecordCount||
       plainSummary.additionalSupportRecordCount!=gzipSummary.additionalSupportRecordCount)
        return 5;

    std::string longRow(header);
    longRow+=row("carrier","source_tetra_anchor_refinement_1046d",
                 "0","0","CurrentCarrierPointProjection","-1","-1",
                 std::string(90000,'0'));
    const auto longPath=dir/"long.csv.gz";
    if(!writeGzip(longPath,longRow))return 6;
    CurrentCarrierWitnessCSVLimits longLimits=defaults;longLimits.maxLineBytes=120000u;
    if(!expectPass(longPath,longLimits,1u,0u))return 7;
    CurrentCarrierWitnessCSVLimits shortLimits=defaults;shortLimits.maxLineBytes=4096u;
    if(!expectFail(longPath,shortLimits,"line exceeds"))return 8;

    CurrentCarrierWitnessCSVLimits byteLimits=defaults;byteLimits.maxDecompressedBytes=64u;
    if(!expectFail(gzip,byteLimits,"decompressed-byte limit"))return 9;
    CurrentCarrierWitnessCSVLimits rowLimits=defaults;rowLimits.maxDataRows=1u;
    if(!expectFail(plain,rowLimits,"row limit"))return 10;

    const auto badHeader=dir/"bad-header.csv";
    if(!writePlain(badHeader,"wrong,header\n"+bytes.substr(header.size())))return 11;
    if(!expectFail(badHeader,defaults,"header is not"))return 12;

    const auto duplicate=dir/"duplicate.csv";
    std::string dup(header);
    dup+=row("carrier","source_tetra_anchor_refinement_1046d","0","0",
             "CurrentCarrierPointProjection","-1","-1");
    dup+=row("carrier_child_slot","current_source_carrier_edge_support","0","0",
             "CurrentCarrierPointProjection","-1","-1");
    if(!writePlain(duplicate,dup)||!expectFail(duplicate,defaults,"repeats a carrier/child support pair"))return 13;

    std::vector<CurrentCarrierWitnessEvidenceRow> badSHArows;
    CurrentCarrierWitnessCSVSummary badSHAsummary;std::string badSHAerror;
    if(load(gzip,15u,badSHArows,badSHAsummary,defaults,badSHAerror,
            "0000000000000000000000000000000000000000000000000000000000000000")||
       badSHAerror.find("differs from its explicit SHA pin")==std::string::npos)return 14;

    const auto badCRC=dir/"bad-crc.csv.gz";
    if(!writeGzip(badCRC,bytes)||!corruptLastByte(badCRC)||
       !expectFail(badCRC,defaults,"gzip decode failed"))return 15;
    std::vector<CurrentCarrierWitnessEvidenceRow> corruptPinnedRows;
    CurrentCarrierWitnessCSVSummary corruptPinnedSummary;std::string corruptPinnedError;
    if(load(badCRC,15u,corruptPinnedRows,corruptPinnedSummary,defaults,
            corruptPinnedError,
            "0000000000000000000000000000000000000000000000000000000000000000")||
       corruptPinnedError.find("differs from its explicit SHA pin")==std::string::npos)return 20;

    const auto truncated=dir/"truncated.csv.gz";
    if(!writeGzip(truncated,bytes))return 16;
    std::filesystem::resize_file(truncated,std::filesystem::file_size(truncated)-4u);
    if(!expectFail(truncated,defaults))return 17;

    const auto trailing=dir/"trailing.csv.gz";
    if(!writeGzip(trailing,bytes))return 18;
    {std::ofstream f(trailing,std::ios::binary|std::ios::app);f.put('x');}
    if(!expectFail(trailing,defaults,"trailing bytes"))return 19;

    std::cout<<"PASS plain_gzip_parity=3_rows long_row_bytes=90000 bounded_line_byte_row_crc_truncation_trailing_sha_header_duplicate=PASS\n";
    return 0;
}
