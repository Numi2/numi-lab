#pragma once

#include "ConservativeSpatialFractionalSemanticSupport.hpp"
#include "NumiHumanPhysicalSkinChildMassMapping.hpp"
#include "ConservativeSpatialSourceCarrierQuadrature.hpp"

#include <zlib.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace metalrobo::human::coarse_tissue {
namespace current_fractional_support_npz_detail {

struct NpyArray {
    std::string descriptor;
    std::vector<std::size_t> shape;
    std::vector<std::uint8_t> bytes;
};

inline std::uint16_t u16(const std::vector<std::uint8_t>& b,std::size_t p){
    return static_cast<std::uint16_t>(b[p])|
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(b[p+1u])<<8u);
}
inline std::uint32_t u32(const std::vector<std::uint8_t>& b,std::size_t p){
    return static_cast<std::uint32_t>(b[p])|
        (static_cast<std::uint32_t>(b[p+1u])<<8u)|
        (static_cast<std::uint32_t>(b[p+2u])<<16u)|
        (static_cast<std::uint32_t>(b[p+3u])<<24u);
}
inline bool parseNpy(const std::vector<std::uint8_t>& raw,NpyArray& out,std::string& error){
    const auto fail=[&](std::string m){error=std::move(m);return false;};
    if(raw.size()<10u||std::memcmp(raw.data(),"\x93NUMPY",6u)!=0)
        return fail("fractional support NPZ member is not a NumPy array");
    const auto major=raw[6u];
    std::size_t headerLength=0u,headerStart=0u;
    if(major==1u){headerLength=u16(raw,8u);headerStart=10u;}
    else if(major==2u||major==3u){
        if(raw.size()<12u)return fail("fractional support NPY v2/v3 header is truncated");
        headerLength=u32(raw,8u);headerStart=12u;
    } else return fail("fractional support NPY version is unsupported");
    if(headerLength>raw.size()-headerStart)return fail("fractional support NPY header exceeds its member");
    const std::string header(reinterpret_cast<const char*>(raw.data()+headerStart),headerLength);
    const auto descriptorAt=header.find("'descr'");
    const auto falseOrder=header.find("False",header.find("'fortran_order'"));
    const auto shapeAt=header.find("'shape'");
    if(descriptorAt==std::string::npos||falseOrder==std::string::npos||shapeAt==std::string::npos)
        return fail("fractional support NPY header lacks C-order descriptor or shape");
    const auto descriptorColon=header.find(':',descriptorAt);
    const auto descriptorQuote=header.find('\'',descriptorColon+1u);
    const auto descriptorEnd=header.find('\'',descriptorQuote+1u);
    if(descriptorColon==std::string::npos||descriptorQuote==std::string::npos||descriptorEnd==std::string::npos)
        return fail("fractional support NPY descriptor is malformed");
    out.descriptor=header.substr(descriptorQuote+1u,descriptorEnd-descriptorQuote-1u);
    const auto shapeOpen=header.find('(',shapeAt),shapeClose=header.find(')',shapeOpen);
    if(shapeOpen==std::string::npos||shapeClose==std::string::npos)
        return fail("fractional support NPY shape is malformed");
    out.shape.clear();
    std::size_t i=shapeOpen+1u;
    while(i<shapeClose){
        while(i<shapeClose&&(header[i]==' '||header[i]==','))++i;
        if(i==shapeClose)break;
        std::size_t value=0u;bool found=false;
        while(i<shapeClose&&header[i]>='0'&&header[i]<='9'){
            const auto digit=static_cast<std::size_t>(header[i]-'0');
            if(value>(std::numeric_limits<std::size_t>::max()-digit)/10u)
                return fail("fractional support NPY dimension overflows");
            value=value*10u+digit;++i;found=true;
        }
        if(!found)return fail("fractional support NPY dimension is malformed");
        out.shape.push_back(value);
        while(i<shapeClose&&header[i]==' ')++i;
        if(i<shapeClose&&header[i]!=',')return fail("fractional support NPY shape separator is malformed");
    }
    std::size_t itemBytes=0u;
    if(out.descriptor=="<i4"||out.descriptor=="<u4"||out.descriptor=="<f4")itemBytes=4u;
    else if(out.descriptor=="<i8"||out.descriptor=="<u8"||out.descriptor=="<f8")itemBytes=8u;
    else if(out.descriptor=="|u1"||out.descriptor=="|i1")itemBytes=1u;
    else if(out.descriptor.size()>2u&&out.descriptor[1]=='U'){
        std::size_t count=0u;
        for(std::size_t j=2u;j<out.descriptor.size();++j){
            if(out.descriptor[j]<'0'||out.descriptor[j]>'9')return fail("fractional support NPY unicode width is malformed");
            count=count*10u+static_cast<std::size_t>(out.descriptor[j]-'0');
        }
        itemBytes=count*4u;
    } else return fail("fractional support NPY dtype is unsupported: "+out.descriptor);
    std::size_t count=1u;
    for(const auto dimension:out.shape){
        if(dimension!=0u&&count>std::numeric_limits<std::size_t>::max()/dimension)
            return fail("fractional support NPY element count overflows");
        count*=dimension;
    }
    if(count>std::numeric_limits<std::size_t>::max()/itemBytes||
       raw.size()-headerStart-headerLength!=count*itemBytes)
        return fail("fractional support NPY payload size differs from its declared shape");
    out.bytes.assign(raw.begin()+static_cast<std::ptrdiff_t>(headerStart+headerLength),raw.end());
    return true;
}

inline bool readRequestedArrays(const std::filesystem::path& path,
        const std::set<std::string>& wanted,std::map<std::string,NpyArray>& arrays,
        std::string& error){
    const auto fail=[&](std::string m){error=std::move(m);return false;};
    std::ifstream input(path,std::ios::binary|std::ios::ate);
    if(!input)return fail("cannot open fractional support NPZ");
    const auto end=input.tellg();
    if(end<22||static_cast<unsigned long long>(end)>2ull*1024ull*1024ull*1024ull)
        return fail("fractional support NPZ size is outside its bounded reader domain");
    std::vector<std::uint8_t> zip(static_cast<std::size_t>(end));
    input.seekg(0);input.read(reinterpret_cast<char*>(zip.data()),static_cast<std::streamsize>(zip.size()));
    if(!input||zip.size()<22u)return fail("failed reading complete fractional support NPZ");
    const auto searchBegin=zip.size()>65557u?zip.size()-65557u:0u;
    std::size_t eocd=std::string::npos;
    for(std::size_t p=zip.size()-22u;;--p){
        if(u32(zip,p)==0x06054b50u){eocd=p;break;}
        if(p==searchBegin)break;
    }
    if(eocd==std::string::npos||eocd+22u>zip.size())return fail("fractional support NPZ central directory is missing");
    const auto entries=u16(zip,eocd+10u);
    const auto directorySize=u32(zip,eocd+12u);
    const auto directoryOffset=u32(zip,eocd+16u);
    if(entries==0xffffu||directorySize==0xffffffffu||directoryOffset==0xffffffffu||
       static_cast<std::size_t>(directoryOffset)+directorySize>eocd)
        return fail("fractional support NPZ ZIP64 or malformed directory is unsupported");
    std::set<std::string> seen;
    std::size_t cursor=directoryOffset;
    for(std::uint32_t entry=0u;entry<entries;++entry){
        if(cursor+46u>zip.size()||u32(zip,cursor)!=0x02014b50u)
            return fail("fractional support NPZ central-directory entry is malformed");
        const auto flags=u16(zip,cursor+8u),method=u16(zip,cursor+10u);
        const auto expectedCRC=u32(zip,cursor+16u),compressed=u32(zip,cursor+20u),uncompressed=u32(zip,cursor+24u);
        const auto nameLength=u16(zip,cursor+28u),extraLength=u16(zip,cursor+30u),commentLength=u16(zip,cursor+32u);
        const auto localOffset=u32(zip,cursor+42u);
        const auto next=cursor+46u+nameLength+extraLength+commentLength;
        if(next>zip.size())return fail("fractional support NPZ directory name exceeds file bounds");
        std::string member(reinterpret_cast<const char*>(zip.data()+cursor+46u),nameLength);
        if(member.find("../")!=std::string::npos||member.starts_with('/'))
            return fail("fractional support NPZ contains a nonlocal member name");
        const bool requestedMember=member.ends_with(".npy")?
            wanted.contains(member.substr(0u,member.size()-4u)):wanted.contains(member);
        if(requestedMember){
            if(!member.ends_with(".npy"))return fail("fractional support NPZ requested member lacks the .npy suffix");
            if(!seen.insert(member).second)return fail("fractional support NPZ repeats a requested member");
            if((flags&1u)!=0u||localOffset+30u>zip.size()||u32(zip,localOffset)!=0x04034b50u)
                return fail("fractional support NPZ requested member is encrypted or has a bad local header");
            const auto localName=u16(zip,localOffset+26u),localExtra=u16(zip,localOffset+28u);
            const auto dataOffset=static_cast<std::size_t>(localOffset)+30u+localName+localExtra;
            if(dataOffset+compressed>zip.size())return fail("fractional support NPZ member payload exceeds file bounds");
            std::vector<std::uint8_t> raw(uncompressed);
            if(method==0u){
                if(compressed!=uncompressed)return fail("fractional support stored member has inconsistent sizes");
                std::copy_n(zip.begin()+static_cast<std::ptrdiff_t>(dataOffset),uncompressed,raw.begin());
            }else if(method==8u){
                z_stream stream{};
                stream.next_in=const_cast<Bytef*>(reinterpret_cast<const Bytef*>(zip.data()+dataOffset));
                stream.avail_in=compressed;
                stream.next_out=reinterpret_cast<Bytef*>(raw.data());
                stream.avail_out=uncompressed;
                if(inflateInit2(&stream,-MAX_WBITS)!=Z_OK)return fail("cannot initialize fractional support deflate reader");
                const auto status=inflate(&stream,Z_FINISH);
                inflateEnd(&stream);
                if(status!=Z_STREAM_END||stream.total_out!=uncompressed)
                    return fail("fractional support deflate member did not finish at its declared size");
            }else return fail("fractional support NPZ compression method is unsupported");
            if(::crc32(0L,reinterpret_cast<const Bytef*>(raw.data()),static_cast<uInt>(raw.size()))!=expectedCRC)
                return fail("fractional support NPZ member CRC differs");
            NpyArray array;
            if(!parseNpy(raw,array,error))return false;
            arrays.emplace(member.substr(0u,member.size()-4u),std::move(array));
        }
        cursor=next;
    }
    for(const auto& name:wanted)if(!seen.contains(name+".npy"))
        return fail("fractional support NPZ omits required member "+name+".npy");
    return true;
}

inline bool readU32(const NpyArray& a,std::vector<std::uint32_t>& out,std::string& error){
    if(a.descriptor!="<i4"&&a.descriptor!="<u4"&&a.descriptor!="<i8"&&a.descriptor!="<u8")
        {error="fractional support integer CSR has an unexpected dtype";return false;}
    const std::size_t width=(a.descriptor=="<i4"||a.descriptor=="<u4")?4u:8u;
    out.clear();out.reserve(a.bytes.size()/width);
    for(std::size_t i=0;i<a.bytes.size();i+=width){
        std::uint64_t value=0u;for(std::size_t j=0;j<width;++j)value|=std::uint64_t(a.bytes[i+j])<<(8u*j);
        if((a.descriptor=="<i4"&&static_cast<std::int32_t>(value)<0)||
           (a.descriptor=="<i8"&&static_cast<std::int64_t>(value)<0)||value>=std::numeric_limits<std::uint32_t>::max())
            {error="fractional support integer value is negative or exceeds the CSR index range";return false;}
        out.push_back(static_cast<std::uint32_t>(value));
    }
    return true;
}
inline bool readU64(const NpyArray& a,std::vector<std::uint64_t>& out,std::string& error){
    if(a.descriptor!="<i8"&&a.descriptor!="<u8"&&a.descriptor!="<i4"&&a.descriptor!="<u4")
        {error="fractional support identity array has an unexpected dtype";return false;}
    const std::size_t width=(a.descriptor=="<i4"||a.descriptor=="<u4")?4u:8u;
    out.clear();out.reserve(a.bytes.size()/width);
    for(std::size_t i=0;i<a.bytes.size();i+=width){
        std::uint64_t value=0u;for(std::size_t j=0;j<width;++j)value|=std::uint64_t(a.bytes[i+j])<<(8u*j);
        if((a.descriptor=="<i8"&&static_cast<std::int64_t>(value)<0)||value==0u)
            {error="fractional support identity is zero or negative";return false;}
        out.push_back(value);
    }
    return true;
}
inline bool readFractions(const NpyArray& numerator,const NpyArray& denominator,
                          std::vector<double>& fractions,std::string& error){
    std::vector<std::uint32_t> n,d;
    if(!readU32(numerator,n,error)||!readU32(denominator,d,error)||n.size()!=d.size())
        {if(error.empty())error="fractional support score numerator/denominator shapes differ";return false;}
    fractions.resize(n.size());
    for(std::size_t i=0;i<n.size();++i){
        if(d[i]==0u||n[i]==0u||n[i]>d[i]){error="fractional support P1 score ratio is outside (0,1]";return false;}
        fractions[i]=static_cast<double>(n[i])/static_cast<double>(d[i]);
    }
    return true;
}
inline bool readF64(const NpyArray& a,std::vector<double>& out,std::string& error){
    if(a.descriptor!="<f8"||a.bytes.size()%sizeof(double)!=0u){error="fractional support numeric array is not little-endian float64";return false;}
    out.resize(a.bytes.size()/sizeof(double));
    for(std::size_t i=0;i<out.size();++i){
        std::uint64_t bits=0u;for(std::size_t b=0;b<8u;++b)bits|=std::uint64_t(a.bytes[i*8u+b])<<(8u*b);
        out[i]=std::bit_cast<double>(bits);
    }
    return true;
}
inline bool readSHA256(const NpyArray& a,std::string& out,std::string& error){
    if(a.descriptor!="<U64"||a.bytes.size()!=256u){error="fractional support SHA member is not one Unicode-64 value";return false;}
    out.clear();
    for(std::size_t i=0;i<64u;++i){
        const std::uint32_t cp=std::uint32_t(a.bytes[i*4u])|
            (std::uint32_t(a.bytes[i*4u+1u])<<8u)|(std::uint32_t(a.bytes[i*4u+2u])<<16u)|
            (std::uint32_t(a.bytes[i*4u+3u])<<24u);
        if(cp==0u)break;if(cp>0x7fu){error="fractional support SHA member is not ASCII";return false;}
        out.push_back(static_cast<char>(cp));
    }
    if(!physical_skin_child_mass_detail::validSHA256(out)){error="fractional support embedded SHA is malformed";return false;}
    return true;
}
inline bool readIntScalar(const NpyArray& a,std::uint32_t& out,std::string& error){
    std::vector<std::uint32_t> values;
    if(!readU32(a,values,error)||values.size()!=1u){error="fractional support hop-count scalar is malformed";return false;}
    out=values.front();return true;
}
} // namespace current_fractional_support_npz_detail

struct CurrentExactTupleProjectionRow {
    std::uint32_t childRow=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t material=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t geometricDonor=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t anatomicalRegion=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t sourceParentRow=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t csvRow=std::numeric_limits<std::uint32_t>::max();
    std::array<double,4> sourceParentBarycentric{};
    std::array<double,3> projectedCurrentPointWorld{};
    std::array<double,3> projectedReferencePointWorld{};
    double currentProjectionDistanceM=0.0;
    double referenceTransportDistanceM=0.0;
    bool referenceBasisAllowed=false;
    std::string sourceCSVSHA256;
};

struct CurrentExactTupleProjectionReject {
    std::uint32_t csvRow=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t sourceParentRow=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t childRow=std::numeric_limits<std::uint32_t>::max();
    double currentDistanceM=0.0;
    double referenceDistanceM=0.0;
    std::string reason;
};

// Merge independently audited exact-tuple/source-carrier projections into the
// per-child allowlist. P1 scores remain preferences; this adds eligibility only
// when the caller supplies the actual pinned projection proof hash.
[[nodiscard]] inline bool bindCurrentExactTupleProjectionRows(
        ConservativeSpatialChildSemanticSupportData& support,
        const ConservativeSpatialMassTransportInput& input,
        const PhysicalSkinParentMassPacket& parent,
        const std::span<const CurrentExactTupleProjectionRow> rows,
        const std::string_view evidenceSHA256,
        std::string& error){
    using Tuple=std::tuple<std::uint32_t,std::uint32_t,std::uint32_t>;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const std::size_t childCount=input.childReferenceVolumeM3.size(),np=input.parentMassKg.size();
    const double maxDistance=input.maximumTransportDistanceM;
    if(rows.empty()||!physical_skin_child_mass_detail::validSHA256(evidenceSHA256)||
       support.childOffsets.size()!=childCount+1u||support.slotGroupIndices.size()!=support.volumeFractions.size()||
       support.groupMaterialIndices.size()!=support.donorBodyIndices.size()||
       support.groupMaterialIndices.size()!=support.anatomicalRegionIndices.size()||
       support.sourceParentTetOffsets.size()!=support.groupMaterialIndices.size()+1u||
       support.sourceParentTetOffsets.empty()||support.sourceParentTetOffsets.front()!=0u||
       support.sourceParentTetOffsets.back()!=support.sourceParentTetRows.size()||
       support.cellAdmissibleRegionOffsets.size()!=childCount+1u||
       support.cellAdmissibleRegionOffsets.empty()||support.cellAdmissibleRegionOffsets.front()!=0u||
       support.cellAdmissibleRegionOffsets.back()!=support.cellAdmissibleRegionIds.size()||
       support.admissibilityMaximumSourceRegionHops!=1u||
       input.parentMaterialIndices.size()!=np||input.parentDonorBodyIndices.size()!=np||
       input.parentAnatomicalRegionIndices.size()!=np||input.childMaterialIndices.size()!=childCount||
       input.childCentroidsWorld.size()!=childCount||parent.tetrahedra.size()!=np||
       parent.currentWorldPoints.size()!=parent.referenceWorldPoints.size()||
       !std::isfinite(maxDistance)||maxDistance<=0.0)
        return fail("exact projection binding requires aligned current parent/child and authenticated one-hop domains");
    const auto coordinateDistance=[](const auto& a,const auto& b){
        return source_carrier_transport_detail::pointDistance(a,b);
    };
    std::map<Tuple,std::uint32_t> groupByTuple;
    for(std::size_t g=0;g<support.groupMaterialIndices.size();++g){
        const Tuple tuple{support.groupMaterialIndices[g],support.donorBodyIndices[g],
                          support.anatomicalRegionIndices[g]};
        if(!groupByTuple.emplace(tuple,static_cast<std::uint32_t>(g)).second)
            return fail("exact tuple projection support has duplicate group identities");
        const auto begin=support.sourceParentTetOffsets[g],end=support.sourceParentTetOffsets[g+1u];
        if(begin>=end||end>support.sourceParentTetRows.size())
            return fail("exact tuple projection source-parent CSR has an empty or invalid group row");
        for(std::uint32_t i=begin;i<end;++i){
            const auto p=support.sourceParentTetRows[i];
            if(p>=np||input.parentMaterialIndices[p]!=std::get<0>(tuple)||
               input.parentDonorBodyIndices[p]!=std::get<1>(tuple)||
               input.parentAnatomicalRegionIndices[p]!=std::get<2>(tuple))
                return fail("exact tuple projection group CSR differs from current parent ownership");
        }
    }
    std::vector<std::vector<std::uint32_t>> allowed(childCount);
    const bool explicitAllowlist=!support.childAdmissibleGroupOffsets.empty()||
                                 !support.childAdmissibleGroupIndices.empty();
    if(explicitAllowlist){
        if(support.childAdmissibleGroupOffsets.size()!=childCount+1u||
           support.childAdmissibleGroupOffsets.front()!=0u||
           support.childAdmissibleGroupOffsets.back()!=support.childAdmissibleGroupIndices.size())
            return fail("exact tuple projection existing group allowlist CSR is malformed");
        for(std::size_t c=0;c<childCount;++c){
            const auto begin=support.childAdmissibleGroupOffsets[c];
            const auto end=support.childAdmissibleGroupOffsets[c+1u];
            if(begin>=end||end>support.childAdmissibleGroupIndices.size())
                return fail("exact tuple projection existing group allowlist has an empty child row");
            allowed[c].assign(support.childAdmissibleGroupIndices.begin()+begin,
                              support.childAdmissibleGroupIndices.begin()+end);
        }
    }else{
        for(std::size_t c=0;c<childCount;++c)
            for(std::uint32_t i=support.childOffsets[c];i<support.childOffsets[c+1u];++i)
                allowed[c].push_back(support.slotGroupIndices[i]);
    }
    for(std::size_t c=0;c<childCount;++c){
        auto& row=allowed[c];
        std::sort(row.begin(),row.end());
        if(std::adjacent_find(row.begin(),row.end())!=row.end())
            return fail("exact tuple projection existing group allowlist repeats a group");
        for(std::uint32_t i=support.childOffsets[c];i<support.childOffsets[c+1u];++i){
            const auto g=support.slotGroupIndices[i];
            if(g>=support.groupMaterialIndices.size()||
               !std::binary_search(row.begin(),row.end(),g))
                return fail("exact tuple projection would discard an existing P1 preference slot");
        }
    }
    std::vector<std::vector<std::uint32_t>> additions(childCount);
    std::set<std::tuple<std::uint32_t,std::uint32_t,std::uint32_t,std::uint32_t>> uniqueRows;
    spatial_mass_transport_detail::SHA256Builder validatedWitnesses;
    validatedWitnesses.string("numi.human.current-exact-projection-reference-replay.v1");
    validatedWitnesses.string(evidenceSHA256);
    for(const auto& row:rows){
        if(!row.referenceBasisAllowed)continue;
        if(row.sourceCSVSHA256!=evidenceSHA256||
           row.sourceParentRow>=np||row.childRow>=childCount||
           row.material==std::numeric_limits<std::uint32_t>::max()||
           row.geometricDonor==std::numeric_limits<std::uint32_t>::max()||
           row.anatomicalRegion==std::numeric_limits<std::uint32_t>::max()||
           !uniqueRows.emplace(row.childRow,row.material,row.geometricDonor,row.anatomicalRegion).second)
            return fail("exact tuple projection row is unauthenticated, invalid or duplicated");
        const auto groupIt=groupByTuple.find({row.material,row.geometricDonor,row.anatomicalRegion});
        if(groupIt==groupByTuple.end())return fail("exact tuple projection references a group absent from the current parent registry");
        const auto group=groupIt->second;
        const auto parentBegin=support.sourceParentTetOffsets[group];
        const auto parentEnd=support.sourceParentTetOffsets[group+1u];
        if(std::find(support.sourceParentTetRows.begin()+parentBegin,
                     support.sourceParentTetRows.begin()+parentEnd,row.sourceParentRow)==
                     support.sourceParentTetRows.begin()+parentEnd)
            return fail("exact tuple projection source tet row is not a member of its exact semantic group");
        const auto p=row.sourceParentRow,c=row.childRow;
        if(input.parentMaterialIndices[p]!=row.material||input.parentDonorBodyIndices[p]!=row.geometricDonor||
           input.parentAnatomicalRegionIndices[p]!=row.anatomicalRegion||
           input.childMaterialIndices[c]!=row.material||
           parent.materialIndices[p]!=row.material||parent.donorBodyIndices[p]!=row.geometricDonor||
           parent.anatomicalRegionIndices[p]!=row.anatomicalRegion)
            return fail("exact tuple projection row no longer matches current parent and child material identity");
        const auto regionBegin=support.cellAdmissibleRegionOffsets[c];
        const auto regionEnd=support.cellAdmissibleRegionOffsets[c+1u];
        if(regionBegin>=regionEnd||regionEnd>support.cellAdmissibleRegionIds.size()||
           !std::is_sorted(support.cellAdmissibleRegionIds.begin()+regionBegin,
                           support.cellAdmissibleRegionIds.begin()+regionEnd)||
           !std::binary_search(support.cellAdmissibleRegionIds.begin()+regionBegin,
                               support.cellAdmissibleRegionIds.begin()+regionEnd,row.anatomicalRegion))
            return fail("exact tuple projection source region is outside the authenticated one-hop child support");
        const auto tet=parent.tetrahedra[p];
        std::array<std::array<double,3>,4> currentTet{},referenceTet{};
        for(std::size_t k=0;k<4u;++k){
            if(tet[k]>=parent.currentWorldPoints.size()||tet[k]>=parent.referenceWorldPoints.size())
                return fail("exact tuple projection parent tet node exceeds the pinned source node domain");
            currentTet[k]=parent.currentWorldPoints[tet[k]];
            referenceTet[k]=parent.referenceWorldPoints[tet[k]];
        }
        const auto currentPoint=source_carrier_transport_detail::barycentricPoint(currentTet,row.sourceParentBarycentric);
        const auto referencePoint=source_carrier_transport_detail::barycentricPoint(referenceTet,row.sourceParentBarycentric);
        const auto currentDistance=coordinateDistance(currentPoint,input.childCentroidsWorld[c]);
        const auto referenceDistance=coordinateDistance(referencePoint,input.childCentroidsWorld[c]);
        const double tol=std::max(1.0e-12,64.0*std::numeric_limits<double>::epsilon()*
            std::max({1.0,currentDistance,referenceDistance,maxDistance}));
        if(coordinateDistance(currentPoint,row.projectedCurrentPointWorld)>5.0e-9||
           coordinateDistance(referencePoint,row.projectedReferencePointWorld)>5.0e-9||
           std::abs(currentDistance-row.currentProjectionDistanceM)>5.0e-9||
           std::abs(referenceDistance-row.referenceTransportDistanceM)>5.0e-9||
           currentDistance>maxDistance+tol||referenceDistance>maxDistance+tol)
            return fail("exact tuple projection source-point geometry fails current/reference rest replay");
        additions[c].push_back(group);
        validatedWitnesses.u32(row.csvRow);validatedWitnesses.u32(p);validatedWitnesses.u32(c);
        validatedWitnesses.u32(group);validatedWitnesses.real(currentDistance);
        validatedWitnesses.real(referenceDistance);
    }
    for(std::size_t c=0;c<childCount;++c){
        auto& row=allowed[c];
        row.insert(row.end(),additions[c].begin(),additions[c].end());
        std::sort(row.begin(),row.end());
        row.erase(std::unique(row.begin(),row.end()),row.end());
    }
    std::vector<std::uint32_t> offsets(childCount+1u,0u),indices;
    for(std::size_t child=0;child<childCount;++child){
        if(allowed[child].empty())return fail("exact tuple projection leaves a child without source tuple support");
        indices.insert(indices.end(),allowed[child].begin(),allowed[child].end());
        offsets[child+1u]=static_cast<std::uint32_t>(indices.size());
    }
    support.childAdmissibleGroupOffsets=std::move(offsets);
    support.childAdmissibleGroupIndices=std::move(indices);
    spatial_mass_transport_detail::SHA256Builder proofDigest;
    proofDigest.string("numi.human.current-exact-tuple-projection-reference-proof.v2");
    proofDigest.string(support.admissibleGroupEvidenceSHA256);
    proofDigest.string(evidenceSHA256);
    proofDigest.string(validatedWitnesses.finish());
    support.admissibleGroupEvidenceSHA256=proofDigest.finish();
    error.clear();return true;
}

struct CurrentFractionalSupportPins {
    std::string supportSHA256;
    std::string currentStateSHA256;
    std::string geometrySHA256;
    std::string volumeSHA256;
    std::string source330MassSHA256;
};

[[nodiscard]] inline bool loadCurrentP1SemanticSupportNPZ(
        const std::filesystem::path& path,const CurrentFractionalSupportPins& pins,
        const PhysicalSkinParentMassPacket& parent,const std::size_t childCount,
        ConservativeSpatialChildSemanticSupportData& output,std::string& error){
    using namespace current_fractional_support_npz_detail;
    const auto fail=[&](std::string m){error=std::move(m);return false;};
    std::string fileSHA;
    if(!physical_skin_child_mass_detail::sha256File(path,fileSHA,error))return false;
    if(fileSHA!=pins.supportSHA256||!physical_skin_child_mass_detail::validSHA256(pins.currentStateSHA256)||
       !physical_skin_child_mass_detail::validSHA256(pins.geometrySHA256)||
       !physical_skin_child_mass_detail::validSHA256(pins.volumeSHA256)||
       !physical_skin_child_mass_detail::validSHA256(pins.source330MassSHA256))
        return fail("fractional support artifact or current state/geometry/volume/source330 pin is invalid");
    const std::set<std::string> wanted{
        "cell_group_offsets","cell_group_ids","cell_group_p1_score_numerators",
        "cell_group_p1_score_denominators","group_tuples","cell_admissible_region_offsets",
        "cell_admissible_region_ids","admissibility_max_source_region_hops",
        "current_state_sha256","geometry_sha256","volume_sha256","source330_mass_sha256",
        "admissibility_source_support_sha256"};
    std::map<std::string,NpyArray> arrays;
    if(!readRequestedArrays(path,wanted,arrays,error))return false;
    const auto& offsets=arrays.at("cell_group_offsets");
    const auto& groups=arrays.at("cell_group_ids");
    const auto& tuples=arrays.at("group_tuples");
    if(offsets.shape!=std::vector<std::size_t>{childCount+1u}||
       tuples.shape.size()!=2u||tuples.shape[1]!=3u)
        return fail("fractional support CSR or exact tuple registry has the wrong current child domain");
    ConservativeSpatialChildSemanticSupportData next;
    if(!readU32(offsets,next.childOffsets,error)||!readU32(groups,next.slotGroupIndices,error)||
       !readFractions(arrays.at("cell_group_p1_score_numerators"),
                      arrays.at("cell_group_p1_score_denominators"),next.volumeFractions,error))return false;
    if(next.childOffsets.size()!=childCount+1u||next.childOffsets.front()!=0u||
       next.childOffsets.back()!=next.slotGroupIndices.size()||
       next.slotGroupIndices.size()!=next.volumeFractions.size())
        return fail("fractional support P1 CSR arrays have inconsistent row lengths");
    next.childAdmissibleGroupOffsets=next.childOffsets;
    next.childAdmissibleGroupIndices=next.slotGroupIndices;
    const std::size_t pinnedGroupCount=tuples.shape[0];
    if(tuples.descriptor!="<i4"||tuples.bytes.size()!=pinnedGroupCount*3u*4u)
        return fail("fractional support exact tuple registry must be a C-order int32 matrix");
    using ExactTuple=std::tuple<std::uint32_t,std::uint32_t,std::uint32_t>;
    std::vector<ExactTuple> groupTuples;groupTuples.reserve(pinnedGroupCount);
    std::map<ExactTuple,std::uint32_t> groupByTuple;
    for(std::size_t g=0;g<pinnedGroupCount;++g){
        std::array<std::uint32_t,3> key{};
        for(std::size_t j=0;j<3u;++j){
            std::uint32_t v=0;for(std::size_t b=0;b<4u;++b)v|=std::uint32_t(tuples.bytes[(g*3u+j)*4u+b])<<(8u*b);
            const auto signedV=static_cast<std::int32_t>(v);
            if(signedV<0){error="fractional support tuple registry contains a negative label";return false;}
            key[j]=static_cast<std::uint32_t>(signedV);
        }
        const ExactTuple tuple{key[0],key[1],key[2]};
        if(!groupByTuple.emplace(tuple,static_cast<std::uint32_t>(g)).second)
            return fail("fractional support exact tuple registry contains duplicate tuples");
        groupTuples.push_back(tuple);
    }
    if(parent.tetrahedra.empty()||parent.materialIndices.size()!=parent.tetrahedra.size()||
       parent.donorBodyIndices.size()!=parent.tetrahedra.size()||
       parent.anatomicalRegionIndices.size()!=parent.tetrahedra.size()||
       parent.stableSourceFaceIdentities.size()!=parent.tetrahedra.size())
        return fail("current source330 parent packet has incomplete typed rows for support binding");
    std::set<ExactTuple> missingParentTuples;
    for(std::size_t p=0;p<parent.tetrahedra.size();++p){
        const ExactTuple key{parent.materialIndices[p],parent.donorBodyIndices[p],parent.anatomicalRegionIndices[p]};
        if(!groupByTuple.contains(key))missingParentTuples.insert(key);
    }
    for(const auto& tuple:missingParentTuples){
        const auto index=static_cast<std::uint32_t>(groupTuples.size());
        groupTuples.push_back(tuple);groupByTuple.emplace(tuple,index);
    }
    const std::size_t groupCount=groupTuples.size();
    next.groupMaterialIndices.resize(groupCount);next.donorBodyIndices.resize(groupCount);
    next.anatomicalRegionIndices.resize(groupCount);next.derivedGroupIdentityIDs.resize(groupCount);
    for(std::size_t g=0;g<groupCount;++g){
        const auto [material,donor,region]=groupTuples[g];
        next.groupMaterialIndices[g]=material;next.donorBodyIndices[g]=donor;
        next.anatomicalRegionIndices[g]=region;next.derivedGroupIdentityIDs[g]=static_cast<std::uint64_t>(g)+1u;
    }
    next.parentStableSourceFaceIdentities=parent.stableSourceFaceIdentities;
    std::vector<std::vector<std::uint32_t>> parentsByGroup(groupCount);
    for(std::size_t p=0;p<parent.tetrahedra.size();++p){
        const ExactTuple key{parent.materialIndices[p],parent.donorBodyIndices[p],parent.anatomicalRegionIndices[p]};
        const auto found=groupByTuple.find(key);
        if(found==groupByTuple.end())return fail("current parent tuple registry construction failed closed");
        parentsByGroup[found->second].push_back(static_cast<std::uint32_t>(p));
    }
    if(parent.tetrahedra.empty()||parent.tetrahedra.size()>=std::numeric_limits<std::uint32_t>::max())
        return fail("current source330 parent domain is empty or exceeds the CSR index range");
    next.sourceParentTetOffsets.assign(groupCount+1u,0u);
    next.sourceStableFaceOffsets.assign(groupCount+1u,0u);
    for(std::size_t g=0;g<groupCount;++g){
        auto& rows=parentsByGroup[g];
        if(rows.empty())return fail("pinned fractional tuple has no current source330 parent rows");
        std::set<std::uint32_t> faces;
        next.sourceParentTetOffsets[g]=static_cast<std::uint32_t>(next.sourceParentTetRows.size());
        for(const auto row:rows){next.sourceParentTetRows.push_back(row);faces.insert(parent.stableSourceFaceIdentities[row]);}
        next.sourceParentTetOffsets[g+1u]=static_cast<std::uint32_t>(next.sourceParentTetRows.size());
        next.sourceStableFaceOffsets[g]=static_cast<std::uint32_t>(next.sourceStableFaceRows.size());
        next.sourceStableFaceRows.insert(next.sourceStableFaceRows.end(),faces.begin(),faces.end());
        next.sourceStableFaceOffsets[g+1u]=static_cast<std::uint32_t>(next.sourceStableFaceRows.size());
    }
    if(!readU32(arrays.at("cell_admissible_region_offsets"),next.cellAdmissibleRegionOffsets,error)||
       !readU32(arrays.at("cell_admissible_region_ids"),next.cellAdmissibleRegionIds,error)||
       !readIntScalar(arrays.at("admissibility_max_source_region_hops"),
                      next.admissibilityMaximumSourceRegionHops,error)||
       !readSHA256(arrays.at("admissibility_source_support_sha256"),
                   next.admissibilitySourceSupportSHA256,error))return false;
    std::string stateSHA,geometrySHA,volumeSHA,sourceMassSHA;
    if(!readSHA256(arrays.at("current_state_sha256"),stateSHA,error)||
       !readSHA256(arrays.at("geometry_sha256"),geometrySHA,error)||
       !readSHA256(arrays.at("volume_sha256"),volumeSHA,error)||
       !readSHA256(arrays.at("source330_mass_sha256"),sourceMassSHA,error))return false;
    if(stateSHA!=pins.currentStateSHA256||geometrySHA!=pins.geometrySHA256||
       volumeSHA!=pins.volumeSHA256||sourceMassSHA!=pins.source330MassSHA256||
       parent.parentMassPacketSHA256!=pins.source330MassSHA256)
        return fail("fractional support embedded/current parent state, geometry, volume or source330 mass hashes differ from explicit caller pins");
    next.typedSupportSHA256=fileSHA;
    output=std::move(next);error.clear();return true;
}


inline bool mergeCurrentPairedP1SemanticSupport(
        const ConservativeSpatialChildSemanticSupportData& skin,
        const ConservativeSpatialChildSemanticSupportData& fat,
        const std::span<const std::uint32_t> skinChildRows,
        const std::span<const std::uint32_t> fatChildRows,
        const std::span<const std::uint32_t> childMaterialIndices,
        const PhysicalSkinParentMassPacket& parent,
        ConservativeSpatialChildSemanticSupportData& output,std::string& error){
    using Tuple=std::tuple<std::uint32_t,std::uint32_t,std::uint32_t>;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const std::size_t childCount=childMaterialIndices.size();
    const auto validRole=[&](const ConservativeSpatialChildSemanticSupportData& role,
                             const std::size_t count){
        return role.childOffsets.size()==count+1u&&!role.childOffsets.empty()&&
            role.childOffsets.front()==0u&&role.childOffsets.back()==role.slotGroupIndices.size()&&
            role.slotGroupIndices.size()==role.volumeFractions.size()&&
            role.cellAdmissibleRegionOffsets.size()==count+1u&&
            role.cellAdmissibleRegionOffsets.front()==0u&&
            role.cellAdmissibleRegionOffsets.back()==role.cellAdmissibleRegionIds.size()&&
            role.groupMaterialIndices.size()==role.donorBodyIndices.size()&&
            role.groupMaterialIndices.size()==role.anatomicalRegionIndices.size()&&
            role.groupMaterialIndices.size()==role.derivedGroupIdentityIDs.size()&&
            role.sourceParentTetOffsets.size()==role.groupMaterialIndices.size()+1u&&
            role.sourceStableFaceOffsets.size()==role.groupMaterialIndices.size()+1u&&
            role.admissibilityMaximumSourceRegionHops==1u&&
            physical_skin_child_mass_detail::validSHA256(role.admissibilitySourceSupportSHA256)&&
            physical_skin_child_mass_detail::validSHA256(role.typedSupportSHA256);
    };
    if(childCount==0u||skinChildRows.size()+fatChildRows.size()!=childCount||
       !validRole(skin,skinChildRows.size())||!validRole(fat,fatChildRows.size())||
       skin.admissibilityMaximumSourceRegionHops!=fat.admissibilityMaximumSourceRegionHops||
       parent.tetrahedra.empty()||parent.materialIndices.size()!=parent.tetrahedra.size()||
       parent.donorBodyIndices.size()!=parent.tetrahedra.size()||
       parent.anatomicalRegionIndices.size()!=parent.tetrahedra.size()||
       parent.stableSourceFaceIdentities.size()!=parent.tetrahedra.size())
        return fail("paired current P1 inputs lack aligned role child/source domains or a common one-hop source proof");
    const auto tuplesFor=[&](const ConservativeSpatialChildSemanticSupportData& role){
        std::map<Tuple,std::uint32_t> result;
        for(std::size_t g=0;g<role.groupMaterialIndices.size();++g)
            if(!result.emplace(Tuple{role.groupMaterialIndices[g],role.donorBodyIndices[g],
                                    role.anatomicalRegionIndices[g]},
                               static_cast<std::uint32_t>(g)).second)
                result.clear();
        return result;
    };
    auto skinByTuple=tuplesFor(skin),fatByTuple=tuplesFor(fat);
    if(skinByTuple.empty()||fatByTuple.empty()||
       skinByTuple.size()!=skin.groupMaterialIndices.size()||
       fatByTuple.size()!=fat.groupMaterialIndices.size())
        return fail("paired current P1 inputs contain duplicate or empty exact tuple registries");
    std::set<Tuple> tupleSet;
    for(const auto& [tuple,index]:skinByTuple){(void)index;tupleSet.insert(tuple);}
    for(const auto& [tuple,index]:fatByTuple){(void)index;tupleSet.insert(tuple);}
    ConservativeSpatialChildSemanticSupportData next;
    std::map<Tuple,std::uint32_t> mergedByTuple;
    for(const auto& tuple:tupleSet){
        const auto g=static_cast<std::uint32_t>(next.groupMaterialIndices.size());
        mergedByTuple.emplace(tuple,g);
        const auto [material,donor,region]=tuple;
        next.groupMaterialIndices.push_back(material);
        next.donorBodyIndices.push_back(donor);
        next.anatomicalRegionIndices.push_back(region);
        next.derivedGroupIdentityIDs.push_back(static_cast<std::uint64_t>(g)+1u);
    }
    next.parentStableSourceFaceIdentities=parent.stableSourceFaceIdentities;
    next.sourceParentTetOffsets.assign(next.groupMaterialIndices.size()+1u,0u);
    next.sourceStableFaceOffsets.assign(next.groupMaterialIndices.size()+1u,0u);
    for(std::size_t g=0;g<next.groupMaterialIndices.size();++g){
        std::set<std::uint32_t> faces;
        for(std::size_t p=0;p<parent.tetrahedra.size();++p){
            if(parent.materialIndices[p]!=next.groupMaterialIndices[g]||
               parent.donorBodyIndices[p]!=next.donorBodyIndices[g]||
               parent.anatomicalRegionIndices[p]!=next.anatomicalRegionIndices[g])continue;
            next.sourceParentTetRows.push_back(static_cast<std::uint32_t>(p));
            faces.insert(parent.stableSourceFaceIdentities[p]);
        }
        if(faces.empty())return fail("paired P1 tuple registry includes a group without current source330 parents");
        next.sourceParentTetOffsets[g+1u]=static_cast<std::uint32_t>(next.sourceParentTetRows.size());
        next.sourceStableFaceRows.insert(next.sourceStableFaceRows.end(),faces.begin(),faces.end());
        next.sourceStableFaceOffsets[g+1u]=static_cast<std::uint32_t>(next.sourceStableFaceRows.size());
    }
    std::vector<bool> seenChild(childCount,false);
    std::vector<std::vector<std::pair<std::uint32_t,double>>> slotsByChild(childCount);
    std::vector<std::vector<std::uint32_t>> regionsByChild(childCount);
    const auto appendRole=[&](const ConservativeSpatialChildSemanticSupportData& role,
                              const std::span<const std::uint32_t> childRows)->bool{
        for(std::size_t local=0;local<childRows.size();++local){
            const auto child=childRows[local];
            if(child>=childCount||seenChild[child])return false;
            seenChild[child]=true;
            for(std::uint32_t row=role.childOffsets[local];row<role.childOffsets[local+1u];++row){
                const auto oldGroup=role.slotGroupIndices[row];
                if(oldGroup>=role.groupMaterialIndices.size())return false;
                const Tuple key{role.groupMaterialIndices[oldGroup],role.donorBodyIndices[oldGroup],
                                role.anatomicalRegionIndices[oldGroup]};
                const auto found=mergedByTuple.find(key);
                if(found==mergedByTuple.end()||
                   next.groupMaterialIndices[found->second]!=childMaterialIndices[child])return false;
                slotsByChild[child].emplace_back(found->second,role.volumeFractions[row]);
            }
            const auto rb=role.cellAdmissibleRegionOffsets[local];
            const auto re=role.cellAdmissibleRegionOffsets[local+1u];
            if(rb>=re||re>role.cellAdmissibleRegionIds.size())return false;
            regionsByChild[child].assign(role.cellAdmissibleRegionIds.begin()+static_cast<std::ptrdiff_t>(rb),
                                         role.cellAdmissibleRegionIds.begin()+static_cast<std::ptrdiff_t>(re));
        }
        return true;
    };
    if(!appendRole(skin,skinChildRows)||!appendRole(fat,fatChildRows)||
       std::find(seenChild.begin(),seenChild.end(),false)!=seenChild.end())
        return fail("paired current P1 child row maps are incomplete, repeated or cross-material");
    next.childOffsets.assign(1u,0u);
    next.cellAdmissibleRegionOffsets.assign(1u,0u);
    for(std::size_t child=0;child<childCount;++child){
        auto& slots=slotsByChild[child];
        std::sort(slots.begin(),slots.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        if(slots.empty())return fail("paired current P1 leaves a current child without preference support");
        for(const auto& [group,fraction]:slots){
            next.slotGroupIndices.push_back(group);next.volumeFractions.push_back(fraction);
        }
        next.childOffsets.push_back(static_cast<std::uint32_t>(next.slotGroupIndices.size()));
        const auto& regions=regionsByChild[child];
        if(regions.empty()||!std::is_sorted(regions.begin(),regions.end())||
           std::adjacent_find(regions.begin(),regions.end())!=regions.end())
            return fail("paired current P1 child one-hop region row is empty or noncanonical");
        next.cellAdmissibleRegionIds.insert(next.cellAdmissibleRegionIds.end(),regions.begin(),regions.end());
        next.cellAdmissibleRegionOffsets.push_back(
            static_cast<std::uint32_t>(next.cellAdmissibleRegionIds.size()));
    }
    next.childAdmissibleGroupOffsets=next.childOffsets;
    next.childAdmissibleGroupIndices=next.slotGroupIndices;
    // Each role support binds a distinct base-P1 artifact. Preserve both
    // authenticated role hashes by deriving a domain-separated paired identity;
    // equality of the two role-local base hashes is not a source eligibility rule.
    spatial_mass_transport_detail::SHA256Builder admissibilityHash;
    admissibilityHash.string("numi.human.current-paired-source-region-support.v1");
    admissibilityHash.string(skin.admissibilitySourceSupportSHA256);
    admissibilityHash.string(fat.admissibilitySourceSupportSHA256);
    next.admissibilitySourceSupportSHA256=admissibilityHash.finish();
    next.admissibilityMaximumSourceRegionHops=skin.admissibilityMaximumSourceRegionHops;
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("numi.human.current-paired-p1-semantic-support.v2");
    hash.string(skin.typedSupportSHA256);hash.string(fat.typedSupportSHA256);
    hash.string(parent.parentMassPacketSHA256);
    for(const auto row:skinChildRows)hash.u32(row);
    for(const auto row:fatChildRows)hash.u32(row);
    next.typedSupportSHA256=hash.finish();
    output=std::move(next);error.clear();return true;
}

// Read and independently replay the bounded exact-projection CSV. The CSV
// producer projects in its current-source coordinate lane; native transport
// uses source330 reference/rest coordinates. Reconstruct both points from the
// same exact source tet row and barycentric witness, then enforce the unchanged
// transport radius in the reference/rest lane before any new slot is admitted.
[[nodiscard]] inline bool loadCurrentExactTupleProjectionCSV(
        const std::filesystem::path& path,const std::string_view expectedSHA256,
        const ConservativeSpatialMassTransportInput& input,
        const PhysicalSkinParentMassPacket& parent,
        const double maximumDistanceM,
        std::vector<CurrentExactTupleProjectionRow>& rows,std::string& error,
        std::vector<CurrentExactTupleProjectionReject>* restBasisRejects=nullptr){
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const std::size_t np=input.parentMassKg.size(),nc=input.childReferenceVolumeM3.size();
    if(!physical_skin_child_mass_detail::validSHA256(expectedSHA256)||
       !std::isfinite(maximumDistanceM)||maximumDistanceM<=0.0||
       input.parentMaterialIndices.size()!=np||input.parentDonorBodyIndices.size()!=np||
       input.parentAnatomicalRegionIndices.size()!=np||input.parentCentroidsWorld.size()!=np||
       input.childMaterialIndices.size()!=nc||input.childDonorBodyIndices.size()!=nc||
       input.childAnatomicalRegionIndices.size()!=nc||input.childCentroidsWorld.size()!=nc||
       parent.tetrahedra.size()!=np||parent.currentWorldPoints.size()!=parent.referenceWorldPoints.size()||
       parent.tetrahedronMassKg.size()!=np||parent.materialIndices.size()!=np||
       parent.donorBodyIndices.size()!=np||parent.anatomicalRegionIndices.size()!=np||
       parent.referenceTetVolumeM3.size()!=np||
       !physical_skin_child_mass_detail::validSHA256(parent.parentMassPacketSHA256))
        return fail("exact projection loader requires aligned pinned current source330 and child transport domains");
    std::string actualSHA;
    if(!physical_skin_child_mass_detail::sha256File(path,actualSHA,error))return false;
    if(actualSHA!=expectedSHA256)return fail("exact projection CSV SHA differs from its explicit input pin");
    std::ifstream inputFile(path);
    if(!inputFile)return fail("cannot open current exact projection CSV");
    std::string line;
    if(!std::getline(inputFile,line))return fail("current exact projection CSV is empty");
    const auto header=physical_skin_child_mass_detail::splitCSV(line);
    const std::map<std::string,std::size_t> column={
        {"source_material",0u},{"source_donor",1u},{"source_region",2u},
        {"source_projection_parent_tet_row",3u},{"source_projection_bary0",4u},
        {"source_projection_bary1",5u},{"source_projection_bary2",6u},
        {"source_projection_bary3",7u},{"projected_source_x_m",8u},
        {"projected_source_y_m",9u},{"projected_source_z_m",10u},
        {"exact_projection_distance_m",11u},{"receiver_child_tet_row",12u},
        {"receiver_current_material",13u},{"receiver_current_donor",14u},
        {"receiver_current_region",15u},{"receiver_centroid_x_m",16u},
        {"receiver_centroid_y_m",17u},{"receiver_centroid_z_m",18u},
        {"projection_status",39u}};
    for(const auto& [name,index]:column)
        if(index>=header.size()||header[index]!=name)
            return fail("current exact projection CSV header is not the pinned 40-column geometry schema");
    if(header.size()!=40u)return fail("current exact projection CSV has an unexpected column count");
    std::vector<CurrentExactTupleProjectionRow> next;
    next.reserve(60000u);
    std::vector<CurrentExactTupleProjectionReject> rejected;
    std::set<std::tuple<std::uint32_t,std::uint32_t,std::uint32_t,std::uint32_t>> unique;
    std::size_t lineNumber=1u;
    while(std::getline(inputFile,line)){
        ++lineNumber;if(line.empty())continue;
        const auto fields=physical_skin_child_mass_detail::splitCSV(line);
        if(fields.size()!=header.size())return fail("current exact projection CSV row has the wrong column count at row "+std::to_string(lineNumber));
        std::uint32_t material=0u,donor=0u,region=0u,sourceParent=0u,child=0u;
        std::uint32_t receiverMaterial=0u,receiverDonor=0u,receiverRegion=0u;
        std::array<double,4> bary{};
        std::array<double,3> reportedCurrentPoint{},reportedChildCentroid{};
        double reportedDistance=0.0;
        if(!physical_skin_child_mass_detail::parseNumber(fields[0],material)||
           !physical_skin_child_mass_detail::parseNumber(fields[1],donor)||
           !physical_skin_child_mass_detail::parseNumber(fields[2],region)||
           !physical_skin_child_mass_detail::parseNumber(fields[3],sourceParent)||
           !physical_skin_child_mass_detail::parseNumber(fields[4],bary[0])||
           !physical_skin_child_mass_detail::parseNumber(fields[5],bary[1])||
           !physical_skin_child_mass_detail::parseNumber(fields[6],bary[2])||
           !physical_skin_child_mass_detail::parseNumber(fields[7],bary[3])||
           !physical_skin_child_mass_detail::parseNumber(fields[8],reportedCurrentPoint[0])||
           !physical_skin_child_mass_detail::parseNumber(fields[9],reportedCurrentPoint[1])||
           !physical_skin_child_mass_detail::parseNumber(fields[10],reportedCurrentPoint[2])||
           !physical_skin_child_mass_detail::parseNumber(fields[11],reportedDistance)||
           !physical_skin_child_mass_detail::parseNumber(fields[12],child)||
           !physical_skin_child_mass_detail::parseNumber(fields[13],receiverMaterial)||
           !physical_skin_child_mass_detail::parseNumber(fields[14],receiverDonor)||
           !physical_skin_child_mass_detail::parseNumber(fields[15],receiverRegion)||
           !physical_skin_child_mass_detail::parseNumber(fields[16],reportedChildCentroid[0])||
           !physical_skin_child_mass_detail::parseNumber(fields[17],reportedChildCentroid[1])||
           !physical_skin_child_mass_detail::parseNumber(fields[18],reportedChildCentroid[2])||
           (material!=15u&&material!=16u)||sourceParent>=np||child>=nc||
           receiverMaterial!=material||input.childMaterialIndices[child]!=material||
           receiverDonor!=input.childDonorBodyIndices[child]||
           receiverRegion!=input.childAnatomicalRegionIndices[child]||
           !std::isfinite(reportedDistance)||reportedDistance<0.0||
           fields[39]!="PASS_EXACT_CURRENT_CENTROID_SOURCE_TUPLE_PROJECTION")
            return fail("current exact projection row has invalid identity, labels, numeric values or status at row "+std::to_string(lineNumber));
        if(input.parentMaterialIndices[sourceParent]!=material||
           input.parentDonorBodyIndices[sourceParent]!=donor||
           input.parentAnatomicalRegionIndices[sourceParent]!=region||
           parent.materialIndices[sourceParent]!=material||
           parent.donorBodyIndices[sourceParent]!=donor||
           parent.anatomicalRegionIndices[sourceParent]!=region)
            return fail("exact projection source tuple does not match its exact current parent-mass row");
        if(!std::all_of(bary.begin(),bary.end(),[](double x){return std::isfinite(x)&&x>=0.0&&x<=1.0;})||
           std::abs(std::accumulate(bary.begin(),bary.end(),0.0)-1.0)>2.0e-12)
            return fail("exact projection barycentric witness is outside the closed source tetrahedron");
        const auto tet=parent.tetrahedra[sourceParent];
        std::array<std::array<double,3>,4> currentTet{},referenceTet{};
        for(std::size_t k=0;k<4u;++k){
            if(tet[k]>=parent.currentWorldPoints.size()||
               tet[k]>=parent.referenceWorldPoints.size())
                return fail("exact projection parent tet node exceeds the pinned source330 point domain");
            currentTet[k]=parent.currentWorldPoints[tet[k]];
            referenceTet[k]=parent.referenceWorldPoints[tet[k]];
            if(!std::all_of(currentTet[k].begin(),currentTet[k].end(),[](double x){return std::isfinite(x);})||
               !std::all_of(referenceTet[k].begin(),referenceTet[k].end(),[](double x){return std::isfinite(x);}))
                return fail("exact projection source tet has a nonfinite coordinate in a pinned lane");
        }
        if(!std::isfinite(parent.referenceTetVolumeM3[sourceParent])||
           parent.referenceTetVolumeM3[sourceParent]<=0.0)
            return fail("exact projection source parent is not a positive reference tetrahedron");
        const auto currentPoint=source_carrier_transport_detail::barycentricPoint(currentTet,bary);
        const auto referencePoint=source_carrier_transport_detail::barycentricPoint(referenceTet,bary);
        const auto currentDistance=source_carrier_transport_detail::pointDistance(
            currentPoint,input.childCentroidsWorld[child]);
        const auto referenceDistance=source_carrier_transport_detail::pointDistance(
            referencePoint,input.childCentroidsWorld[child]);
        const auto currentPointError=source_carrier_transport_detail::pointDistance(
            currentPoint,reportedCurrentPoint);
        const auto childCentroidError=source_carrier_transport_detail::pointDistance(
            input.childCentroidsWorld[child],reportedChildCentroid);
        const double coordinateTol=5.0e-9;
        const double distanceTol=std::max(1.0e-12,64.0*std::numeric_limits<double>::epsilon()*
            std::max({1.0,currentDistance,reportedDistance,maximumDistanceM}));
        if(!std::isfinite(currentDistance)||!std::isfinite(referenceDistance)||
           currentPointError>coordinateTol||childCentroidError>coordinateTol||
           std::abs(currentDistance-reportedDistance)>coordinateTol||
           currentDistance>maximumDistanceM+distanceTol)
            return fail("exact projection current-coordinate point/child/distance does not replay against the pinned current geometry");
        if(!unique.emplace(child,material,donor,region).second)
            return fail("current exact projection CSV repeats an exact child/group slot");
        CurrentExactTupleProjectionRow row;
        row.childRow=child;row.material=material;row.geometricDonor=donor;row.anatomicalRegion=region;
        row.sourceParentRow=sourceParent;row.csvRow=static_cast<std::uint32_t>(next.size());
        row.sourceParentBarycentric=bary;row.projectedCurrentPointWorld=currentPoint;
        row.projectedReferencePointWorld=referencePoint;row.currentProjectionDistanceM=currentDistance;
        row.referenceTransportDistanceM=referenceDistance;row.sourceCSVSHA256=actualSHA;
        const double referenceTol=std::max(1.0e-12,64.0*std::numeric_limits<double>::epsilon()*
            std::max({1.0,referenceDistance,maximumDistanceM}));
        row.referenceBasisAllowed=referenceDistance<=maximumDistanceM+referenceTol;
        if(!row.referenceBasisAllowed)
            rejected.push_back({row.csvRow,sourceParent,child,currentDistance,referenceDistance,
                                "reference_rest_source_point_exceeds_transport_radius"});
        next.push_back(row);
    }
    if(!inputFile.eof()||next.empty())return fail("current exact projection CSV failed its complete nonempty read");
    rows=std::move(next);
    if(restBasisRejects)*restBasisRejects=std::move(rejected);
    error.clear();return true;
}

struct CurrentSourceCarrierQuadratureData {
    std::vector<std::uint32_t> parentOffsets,globalSourceTetRows,witnessChildRows;
    std::vector<std::array<std::uint32_t,4>> sourceTetNodeRows;
    std::vector<std::array<std::array<double,3>,4>> parentReferenceTetWorldVertices;
    std::vector<double> carrierWeights,witnessChildCentroidDistancesM;
    std::vector<std::array<double,4>> carrierBarycentric;
    std::vector<std::uint32_t> carrierRoleLocalRows,carrierRoleMaterialIndices,carrierRoleLocalChildRows,carrierRoleLocalParentRows;
    std::string sourceCarrierPacketSHA256,sourceTetTableSHA256,sourceNodeTableSHA256;

    [[nodiscard]] SourceCellCarrierQuadratureInput view() const {
        return {parentOffsets,globalSourceTetRows,sourceTetNodeRows,parentReferenceTetWorldVertices,
            carrierWeights,carrierBarycentric,witnessChildRows,witnessChildCentroidDistancesM,
            sourceCarrierPacketSHA256,sourceTetTableSHA256,sourceNodeTableSHA256};
    }
};

struct CurrentSourceCarrierPins {
    std::string skinPacketSHA256,fatPacketSHA256;
    std::string sourceTetTableSHA256,sourceNodeTableSHA256;
    std::uint32_t skinMaterialIndex=15u,fatMaterialIndex=16u;
};

namespace current_fractional_support_npz_detail {
inline bool readCarrierPacket(const std::filesystem::path& path,const std::string_view expectedSHA,
        const std::uint32_t roleMaterial,const PhysicalSkinParentMassPacket& parent,
        const std::span<const std::uint32_t> roleChildRows,
        std::vector<std::vector<std::array<double,10>>>& rowsByParent,
        std::string& error){
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    std::string actualSHA;
    if(!physical_skin_child_mass_detail::sha256File(path,actualSHA,error))return false;
    if(actualSHA!=expectedSHA||!physical_skin_child_mass_detail::validSHA256(expectedSHA))
        return fail("source carrier packet SHA differs from its explicit caller pin");
    const std::set<std::string> wanted{"source_parent_tet_rows","global_source330_tet_rows","parent_offsets",
        "carrier_weights","carrier_barycentric","carrier_target_child_rows","carrier_target_distance_m"};
    std::map<std::string,NpyArray> arrays;
    if(!readRequestedArrays(path,wanted,arrays,error))return false;
    std::vector<std::uint32_t> localRows,globalRows,offsets,targetRows;
    std::vector<double> weights,baryFlat,distances;
    const auto& localA=arrays.at("source_parent_tet_rows");
    const auto& globalA=arrays.at("global_source330_tet_rows");
    const auto& offsetA=arrays.at("parent_offsets");
    const auto& weightA=arrays.at("carrier_weights");
    const auto& baryA=arrays.at("carrier_barycentric");
    const auto& targetA=arrays.at("carrier_target_child_rows");
    const auto& distanceA=arrays.at("carrier_target_distance_m");
    if(localA.shape.size()!=1u||globalA.shape!=localA.shape||offsetA.shape!=std::vector<std::size_t>{localA.shape[0]+1u}||
       weightA.shape.size()!=1u||baryA.shape!=std::vector<std::size_t>{weightA.shape[0],4u}||
       targetA.shape!=weightA.shape||distanceA.shape!=weightA.shape)
        return fail("source carrier NPZ arrays have inconsistent role or carrier domains");
    if(!readU32(localA,localRows,error)||!readU32(globalA,globalRows,error)||
       !readU32(offsetA,offsets,error)||!readU32(targetA,targetRows,error)||
       !readF64(weightA,weights,error)||!readF64(baryA,baryFlat,error)||!readF64(distanceA,distances,error))return false;
    if(offsets.empty()||offsets.front()!=0u||offsets.back()!=weights.size()||
       localRows.size()+1u!=offsets.size()||globalRows.size()!=localRows.size()||
       targetRows.size()!=weights.size()||distances.size()!=weights.size()||baryFlat.size()!=weights.size()*4u)
        return fail("source carrier NPZ CSR terminals or aligned row lengths are invalid");
    std::vector<std::uint32_t> expectedGlobalRows;
    for(std::size_t p=0;p<parent.materialIndices.size();++p)
        if(parent.materialIndices[p]==roleMaterial)expectedGlobalRows.push_back(static_cast<std::uint32_t>(p));
    if(expectedGlobalRows.size()!=localRows.size())
        return fail("source carrier role source-parent domain differs from the current parent mass packet");
    for(std::size_t local=0;local<localRows.size();++local){
        const auto global=globalRows[local];
        if(localRows[local]!=local||global!=expectedGlobalRows[local]||global>=parent.tetrahedra.size()||
           parent.materialIndices[global]!=roleMaterial)
            return fail("source carrier local/global source-parent mapping differs from current role row order");
        if(offsets[local]>offsets[local+1u]||offsets[local+1u]>weights.size())
            return fail("source carrier role CSR offsets are reversed or out of range");
        for(std::uint32_t k=offsets[local];k<offsets[local+1u];++k){
            if(targetRows[k]>=roleChildRows.size()||!std::isfinite(distances[k])||
               roleChildRows[targetRows[k]]>=std::numeric_limits<std::uint32_t>::max())
                return fail("source carrier target child or stored distance is outside the current child domain");
            std::array<double,10> row{};
            row[0]=weights[k];row[1]=baryFlat[static_cast<std::size_t>(k)*4u];
            row[2]=baryFlat[static_cast<std::size_t>(k)*4u+1u];
            row[3]=baryFlat[static_cast<std::size_t>(k)*4u+2u];
            row[4]=baryFlat[static_cast<std::size_t>(k)*4u+3u];
            row[5]=static_cast<double>(roleChildRows[targetRows[k]]);
            row[6]=distances[k];row[7]=static_cast<double>(k);row[8]=static_cast<double>(targetRows[k]);row[9]=static_cast<double>(local);
            rowsByParent[global].push_back(row);
        }
    }
    return true;
}
} // namespace current_fractional_support_npz_detail

[[nodiscard]] inline bool loadCurrentPairedSourceCarrierQuadratureNPZ(
        const std::filesystem::path& skinPath,const std::filesystem::path& fatPath,
        const std::filesystem::path& sourceTetTablePath,const std::filesystem::path& sourceNodeTablePath,
        const CurrentSourceCarrierPins& pins,const PhysicalSkinParentMassPacket& parent,
        const std::span<const std::uint32_t> skinChildRows,
        const std::span<const std::uint32_t> fatChildRows,
        CurrentSourceCarrierQuadratureData& output,std::string& error){
    using namespace current_fractional_support_npz_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(parent.tetrahedra.empty()||parent.referenceWorldPoints.empty()||
       parent.materialIndices.size()!=parent.tetrahedra.size()||
       parent.tetrahedronMassKg.size()!=parent.tetrahedra.size()||
       parent.referenceTetVolumeM3.size()!=parent.tetrahedra.size()||
       !physical_skin_child_mass_detail::validSHA256(pins.sourceTetTableSHA256)||
       !physical_skin_child_mass_detail::validSHA256(pins.sourceNodeTableSHA256)||
       pins.skinMaterialIndex==pins.fatMaterialIndex||skinChildRows.empty()||fatChildRows.empty()||
       skinChildRows.size()+fatChildRows.size()>=std::numeric_limits<std::uint32_t>::max())
        return fail("current source carrier loading requires complete parent geometry and distinct pinned role materials");
    std::string actualTetsSHA,actualNodesSHA;
    if(!physical_skin_child_mass_detail::sha256File(sourceTetTablePath,actualTetsSHA,error)||
       !physical_skin_child_mass_detail::sha256File(sourceNodeTablePath,actualNodesSHA,error))return false;
    if(actualTetsSHA!=pins.sourceTetTableSHA256||actualNodesSHA!=pins.sourceNodeTableSHA256)
        return fail("current source carrier tet/node table hashes differ from the explicit source geometry pins");
    std::vector<std::vector<std::array<double,10>>> rowsByParent(parent.tetrahedra.size());
    if(!readCarrierPacket(skinPath,pins.skinPacketSHA256,pins.skinMaterialIndex,parent,
                          skinChildRows,rowsByParent,error)||
       !readCarrierPacket(fatPath,pins.fatPacketSHA256,pins.fatMaterialIndex,parent,
                          fatChildRows,rowsByParent,error))return false;
    CurrentSourceCarrierQuadratureData next;
    next.sourceTetTableSHA256=pins.sourceTetTableSHA256;next.sourceNodeTableSHA256=pins.sourceNodeTableSHA256;
    spatial_mass_transport_detail::SHA256Builder packetHash;
    packetHash.string("numi.human.current-paired-source-carrier-packets.v1");
    packetHash.string(pins.skinPacketSHA256);packetHash.string(pins.fatPacketSHA256);
    packetHash.string(pins.sourceTetTableSHA256);packetHash.string(pins.sourceNodeTableSHA256);
    next.sourceCarrierPacketSHA256=packetHash.finish();
    const std::size_t np=parent.tetrahedra.size();
    next.parentOffsets.assign(np+1u,0u);next.globalSourceTetRows.resize(np);
    next.sourceTetNodeRows=parent.tetrahedra;next.parentReferenceTetWorldVertices.resize(np);
    for(std::size_t p=0;p<np;++p){
        next.globalSourceTetRows[p]=static_cast<std::uint32_t>(p);
        const auto tet=parent.tetrahedra[p];
        for(std::size_t corner=0;corner<4u;++corner){
            if(tet[corner]>=parent.referenceWorldPoints.size())return fail("source parent tet node row exceeds the current source node domain");
            next.parentReferenceTetWorldVertices[p][corner]=parent.referenceWorldPoints[tet[corner]];
        }
        for(const auto& row:rowsByParent[p]){
            next.carrierWeights.push_back(row[0]);
            next.carrierBarycentric.push_back({row[1],row[2],row[3],row[4]});
            next.witnessChildRows.push_back(static_cast<std::uint32_t>(row[5]));
            next.witnessChildCentroidDistancesM.push_back(row[6]);
            next.carrierRoleLocalRows.push_back(static_cast<std::uint32_t>(row[7]));
            next.carrierRoleMaterialIndices.push_back(parent.materialIndices[p]);
            next.carrierRoleLocalChildRows.push_back(static_cast<std::uint32_t>(row[8]));
            next.carrierRoleLocalParentRows.push_back(static_cast<std::uint32_t>(row[9]));
        }
        if(next.carrierWeights.size()>=std::numeric_limits<std::uint32_t>::max())
            return fail("paired current source carrier packet exceeds the bounded row index domain");
        next.parentOffsets[p+1u]=static_cast<std::uint32_t>(next.carrierWeights.size());
    }
    if(next.carrierWeights.empty())return fail("paired current source carrier packets contain no witnessed carriers");
    output=std::move(next);error.clear();return true;
}



enum class CurrentCarrierWitnessMode : std::uint32_t {
    P1ExactTupleSlot=0u,
    AuthenticatedExactProjection=1u,
    CurrentCarrierPointProjection=2u
};

struct CurrentCarrierWitnessEvidenceRow {
    std::uint32_t roleMaterial=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t roleLocalCarrierRow=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t sourceParentLocalRow=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t sourceParentRow=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t roleLocalChildRow=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t childRow=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t material=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t donor=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t region=std::numeric_limits<std::uint32_t>::max();
    CurrentCarrierWitnessMode mode=CurrentCarrierWitnessMode::P1ExactTupleSlot;
    std::uint32_t projectionCSVRow=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t declaredP1GroupID=std::numeric_limits<std::uint32_t>::max();
    std::array<double,4> claimedBarycentric{};
    double claimedWeight=0.0;
    std::array<double,3> claimedSourcePointWorld{},claimedChildCentroidWorld{};
    double claimedDistanceM=0.0;
    std::string carrierConstruction,sourceCSVSHA256;
    bool hasCSVProof=false;
    bool extraChildSlot=false;
};

struct CurrentCarrierWitnessCSVSummary {
    std::string sourceCSVSHA256;
    std::uint32_t carrierRecordCount=0u;
    std::uint32_t additionalSupportRecordCount=0u;
    std::uint32_t centroidFallbackRecordCount=0u;
};

// The 1046c CSV is evidence, not the carrier data source. The NPZ remains
// authoritative for carrier arrays; this reader pins and parses only the two
// record kinds needed to bind each role-local carrier index to its sidecar row.
[[nodiscard]] inline bool loadCurrentCarrierWitnessCSV(
        const std::filesystem::path& path,const std::string_view expectedSHA256,
        const std::uint32_t expectedRoleMaterial,
        std::vector<CurrentCarrierWitnessEvidenceRow>& rows,
        CurrentCarrierWitnessCSVSummary& summary,std::string& error){
    using namespace current_fractional_support_npz_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if((expectedRoleMaterial!=15u&&expectedRoleMaterial!=16u)||
       !physical_skin_child_mass_detail::validSHA256(expectedSHA256))
        return fail("1046c carrier witness CSV requires an exact role and SHA pin");
    std::string actualSHA;
    if(!physical_skin_child_mass_detail::sha256File(path,actualSHA,error))return false;
    if(actualSHA!=expectedSHA256)return fail("1046c carrier witness CSV differs from its explicit SHA pin");
    std::ifstream input(path);
    if(!input)return fail("cannot open current 1046c carrier witness CSV");
    std::string line;
    if(!std::getline(input,line))return fail("current 1046c carrier witness CSV is empty");
    const auto header=physical_skin_child_mass_detail::splitCSV(line);
    static const std::vector<std::string> expectedHeader{
        "record_kind","role","carrier_construction","packet_carrier_row","local_parent_row",
        "global_source330_tet_row","child_local_row","child_global_row","material","donor","region",
        "provenance_mode","p1_group_id","projection_csv_row","bary0","bary1","bary2","bary3",
        "weight","source_x_m","source_y_m","source_z_m","child_centroid_x_m","child_centroid_y_m",
        "child_centroid_z_m","distance_m"};
    if(header!=expectedHeader)
        return fail("1046c carrier witness CSV header is not the pinned 26-column schema");
    const std::string expectedRole=expectedRoleMaterial==15u?"Skin":"Fat";
    std::vector<CurrentCarrierWitnessEvidenceRow> next;
    std::set<std::uint32_t> carrierRows;
    std::set<std::pair<std::uint32_t,std::uint32_t>> supportPairs;
    std::size_t fallbackCount=0u,edgeSupportCount=0u,physicalLine=1u;
    while(std::getline(input,line)){
        ++physicalLine;if(line.empty())continue;
        const auto f=physical_skin_child_mass_detail::splitCSV(line);
        if(f.size()!=expectedHeader.size()||f[1]!=expectedRole)
            return fail("1046c carrier witness row has an unexpected role or field count at line "+std::to_string(physicalLine));
        std::int64_t packetRow=-2,projectionRow=-2,p1Group=-2;
        CurrentCarrierWitnessEvidenceRow row;
        row.roleMaterial=expectedRoleMaterial;
        std::uint32_t csvMaterial=0u,csvDonor=0u,csvRegion=0u;
        if(!physical_skin_child_mass_detail::parseNumber(f[3],packetRow)||
           !physical_skin_child_mass_detail::parseNumber(f[4],row.sourceParentLocalRow)||
           !physical_skin_child_mass_detail::parseNumber(f[5],row.sourceParentRow)||
           !physical_skin_child_mass_detail::parseNumber(f[6],row.roleLocalChildRow)||
           !physical_skin_child_mass_detail::parseNumber(f[7],row.childRow)||
           !physical_skin_child_mass_detail::parseNumber(f[8],csvMaterial)||
           !physical_skin_child_mass_detail::parseNumber(f[9],csvDonor)||
           !physical_skin_child_mass_detail::parseNumber(f[10],csvRegion)||
           !physical_skin_child_mass_detail::parseNumber(f[12],p1Group)||
           !physical_skin_child_mass_detail::parseNumber(f[13],projectionRow)||
           !physical_skin_child_mass_detail::parseNumber(f[14],row.claimedBarycentric[0])||
           !physical_skin_child_mass_detail::parseNumber(f[15],row.claimedBarycentric[1])||
           !physical_skin_child_mass_detail::parseNumber(f[16],row.claimedBarycentric[2])||
           !physical_skin_child_mass_detail::parseNumber(f[17],row.claimedBarycentric[3])||
           !physical_skin_child_mass_detail::parseNumber(f[18],row.claimedWeight)||
           !physical_skin_child_mass_detail::parseNumber(f[19],row.claimedSourcePointWorld[0])||
           !physical_skin_child_mass_detail::parseNumber(f[20],row.claimedSourcePointWorld[1])||
           !physical_skin_child_mass_detail::parseNumber(f[21],row.claimedSourcePointWorld[2])||
           !physical_skin_child_mass_detail::parseNumber(f[22],row.claimedChildCentroidWorld[0])||
           !physical_skin_child_mass_detail::parseNumber(f[23],row.claimedChildCentroidWorld[1])||
           !physical_skin_child_mass_detail::parseNumber(f[24],row.claimedChildCentroidWorld[2])||
           !physical_skin_child_mass_detail::parseNumber(f[25],row.claimedDistanceM))
            return fail("1046c carrier witness row has malformed numeric fields at line "+std::to_string(physicalLine));
        row.material=csvMaterial;row.donor=csvDonor;row.region=csvRegion;
        row.carrierConstruction=f[2];row.sourceCSVSHA256=actualSHA;row.hasCSVProof=true;
        if(csvMaterial!=expectedRoleMaterial||row.carrierConstruction.empty()||
           !std::isfinite(row.claimedWeight)||!std::isfinite(row.claimedDistanceM)||
           row.claimedDistanceM<0.0||
           !std::all_of(row.claimedBarycentric.begin(),row.claimedBarycentric.end(),
                       [](double x){return std::isfinite(x);})||
           !std::all_of(row.claimedSourcePointWorld.begin(),row.claimedSourcePointWorld.end(),
                       [](double x){return std::isfinite(x);})||
           !std::all_of(row.claimedChildCentroidWorld.begin(),row.claimedChildCentroidWorld.end(),
                       [](double x){return std::isfinite(x);}))
            return fail("1046c witness has invalid role, construction, coordinates or measure at line "+std::to_string(physicalLine));
        if(f[0]=="centroid_route_fallback"){
            if(packetRow!=-1||p1Group<0||projectionRow!=-1||
               f[11]!="p1_exact_tuple_slot"||f[2]!="unchanged_source_centroid")
                return fail("1046c centroid fallback record is not the pinned diagnostic form");
            ++fallbackCount;
            continue;
        }
        if((f[0]!="carrier"&&f[0]!="carrier_child_slot")||packetRow<0||
           static_cast<std::uint64_t>(packetRow)>=std::numeric_limits<std::uint32_t>::max())
            return fail("1046c witness record kind or packet carrier row is invalid");
        row.roleLocalCarrierRow=static_cast<std::uint32_t>(packetRow);
        row.extraChildSlot=f[0]=="carrier_child_slot";
        if(!supportPairs.emplace(row.roleLocalCarrierRow,row.childRow).second)
            return fail("current carrier witness CSV repeats a carrier/child support pair");
        if(projectionRow< -1||projectionRow>=static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())||
           p1Group< -1||p1Group>=static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()))
            return fail("1046c carrier witness P1/projection row has an invalid sentinel or value");
        row.projectionCSVRow=projectionRow<0?std::numeric_limits<std::uint32_t>::max():
                             static_cast<std::uint32_t>(projectionRow);
        if(p1Group>=0)row.declaredP1GroupID=static_cast<std::uint32_t>(p1Group);
        if(row.extraChildSlot){
            if(f[11]!="CurrentCarrierPointProjection"||projectionRow!=-1||p1Group!=-1||
               f[2]!="current_source_carrier_edge_support")
                return fail("carrier child-slot row must use the exact carrier-point support mode and sentinels");
            row.mode=CurrentCarrierWitnessMode::CurrentCarrierPointProjection;
            ++edgeSupportCount;
        }else{
            if(!carrierRows.insert(row.roleLocalCarrierRow).second)
                return fail("1046c carrier witness CSV repeats a primary role-local packet row");
            if(f[11]=="p1_exact_tuple_slot")row.mode=CurrentCarrierWitnessMode::P1ExactTupleSlot;
            else if(f[11]=="authenticated_exact_projection")
                row.mode=CurrentCarrierWitnessMode::AuthenticatedExactProjection;
            else if(f[11]=="CurrentCarrierPointProjection")
                row.mode=CurrentCarrierWitnessMode::CurrentCarrierPointProjection;
            else return fail("1046c carrier witness uses an unknown provenance mode");
            if((row.mode==CurrentCarrierWitnessMode::AuthenticatedExactProjection)!=(projectionRow>=0))
                return fail("1046c carrier witness projection mode and projection row disagree");
            if(row.mode==CurrentCarrierWitnessMode::P1ExactTupleSlot&&p1Group<0)
                return fail("1046c P1 carrier witness omits its role-local P1 group identity");
            if(f[2]!="inherited_090_quadrature"&&
               f[2]!="current_exact_tuple_projection_degree1_lp"&&
               f[2]!="source_tetra_anchor_refinement_1046d")
                return fail("1046c carrier witness uses an unknown construction provenance");
        }
        next.push_back(std::move(row));
    }
    if(!input.eof()||next.empty()||carrierRows.empty())
        return fail("1046c carrier witness CSV failed complete nonempty parsing");
    for(std::uint32_t i=0;i<carrierRows.size();++i)
        if(!carrierRows.contains(i))
            return fail("1046c primary carrier rows are not a complete zero-based domain");
    std::sort(next.begin(),next.end(),[](const auto& a,const auto& b){
        if(a.roleLocalCarrierRow!=b.roleLocalCarrierRow)
            return a.roleLocalCarrierRow<b.roleLocalCarrierRow;
        if(a.extraChildSlot!=b.extraChildSlot)return a.extraChildSlot<b.extraChildSlot;
        return a.childRow<b.childRow;
    });
    rows=std::move(next);
    summary.sourceCSVSHA256=actualSHA;
    summary.carrierRecordCount=static_cast<std::uint32_t>(carrierRows.size());
    summary.additionalSupportRecordCount=static_cast<std::uint32_t>(edgeSupportCount);
    summary.centroidFallbackRecordCount=static_cast<std::uint32_t>(fallbackCount);
    error.clear();return true;
}

// Authenticate each 1046b witness as an exact typed tuple. A carrier-point
// witness may add an exact support slot only after replaying the barycentric
// source point, current child material, one-hop source region and unchanged
// source-point-to-current-child-centroid radius.
[[nodiscard]] inline bool bindCurrentCarrierWitnessAllowlist(
        ConservativeSpatialChildSemanticSupportData& support,
        const ConservativeSpatialMassTransportInput& input,
        const CurrentSourceCarrierQuadratureData& carrier,
        const std::span<const CurrentCarrierWitnessEvidenceRow> evidenceRows,
        const std::span<const CurrentExactTupleProjectionRow> exactProjectionRows,
        const std::string_view exactProjectionCSVSHA256,
        const std::string_view evidenceSHA256,std::string& error,
        const std::span<const std::uint32_t> skinChildGlobalRows={},
        const std::span<const std::uint32_t> fatChildGlobalRows={}){
    using Tuple=std::tuple<std::uint32_t,std::uint32_t,std::uint32_t>;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const std::size_t nc=input.childReferenceVolumeM3.size(),np=input.parentMassKg.size();
    const std::size_t nCarrier=carrier.carrierWeights.size();
    if(evidenceRows.size()<nCarrier||nCarrier==0u||
       carrier.parentOffsets.size()!=np+1u||carrier.witnessChildRows.size()!=nCarrier||
       carrier.carrierBarycentric.size()!=nCarrier||carrier.carrierRoleLocalRows.size()!=nCarrier||
       carrier.carrierRoleMaterialIndices.size()!=nCarrier||
       carrier.carrierRoleLocalChildRows.size()!=nCarrier||
       carrier.carrierRoleLocalParentRows.size()!=nCarrier||
       carrier.parentReferenceTetWorldVertices.size()!=np||
       support.childOffsets.size()!=nc+1u||support.childOffsets.back()!=support.slotGroupIndices.size()||
       support.cellAdmissibleRegionOffsets.size()!=nc+1u||
       support.admissibilityMaximumSourceRegionHops!=1u||
       !physical_skin_child_mass_detail::validSHA256(
           support.admissibilitySourceSupportSHA256)||
       !physical_skin_child_mass_detail::validSHA256(exactProjectionCSVSHA256)||
       !physical_skin_child_mass_detail::validSHA256(evidenceSHA256)||
       input.parentMaterialIndices.size()!=np||input.parentDonorBodyIndices.size()!=np||
       input.parentAnatomicalRegionIndices.size()!=np||
       input.childMaterialIndices.size()!=nc||input.childCentroidsWorld.size()!=nc)
        return fail("current carrier witness sidecar or paired source/child domains are incomplete");
    std::map<Tuple,std::uint32_t> groupByTuple;
    for(std::size_t g=0;g<support.groupMaterialIndices.size();++g){
        const Tuple key{support.groupMaterialIndices[g],support.donorBodyIndices[g],
                        support.anatomicalRegionIndices[g]};
        if(!groupByTuple.emplace(key,static_cast<std::uint32_t>(g)).second)
            return fail("current carrier witness support has duplicate exact tuple groups");
    }
    std::vector<std::vector<std::uint32_t>> allowed(nc);
    const bool hasExplicit=!support.childAdmissibleGroupOffsets.empty()||
                           !support.childAdmissibleGroupIndices.empty();
    if(hasExplicit){
        if(support.childAdmissibleGroupOffsets.size()!=nc+1u||
           support.childAdmissibleGroupOffsets.front()!=0u||
           support.childAdmissibleGroupOffsets.back()!=support.childAdmissibleGroupIndices.size())
            return fail("current carrier witness existing exact-group allowlist CSR is malformed");
        for(std::size_t c=0;c<nc;++c){
            const auto begin=support.childAdmissibleGroupOffsets[c];
            const auto end=support.childAdmissibleGroupOffsets[c+1u];
            if(begin>=end||end>support.childAdmissibleGroupIndices.size())
                return fail("current carrier witness existing exact-group allowlist has an empty row");
            allowed[c].assign(support.childAdmissibleGroupIndices.begin()+begin,
                              support.childAdmissibleGroupIndices.begin()+end);
        }
    }else{
        for(std::size_t c=0;c<nc;++c)
            for(std::uint32_t row=support.childOffsets[c];row<support.childOffsets[c+1u];++row)
                allowed[c].push_back(support.slotGroupIndices[row]);
    }
    for(std::size_t c=0;c<nc;++c){
        auto& row=allowed[c];
        std::sort(row.begin(),row.end());
        row.erase(std::unique(row.begin(),row.end()),row.end());
        for(std::uint32_t p1=support.childOffsets[c];p1<support.childOffsets[c+1u];++p1)
            if(!std::binary_search(row.begin(),row.end(),support.slotGroupIndices[p1]))
                return fail("current carrier witness allowlist omits an original P1 preference group");
    }
    std::vector<std::uint32_t> carrierParent(nCarrier,std::numeric_limits<std::uint32_t>::max());
    if(carrier.parentOffsets.empty()||carrier.parentOffsets.front()!=0u||
       carrier.parentOffsets.back()!=nCarrier)
        return fail("current carrier witness parent CSR is malformed");
    for(std::size_t p=0;p<np;++p)
        for(std::uint32_t k=carrier.parentOffsets[p];k<carrier.parentOffsets[p+1u];++k)
            carrierParent[k]=static_cast<std::uint32_t>(p);
    std::vector<bool> seen(nCarrier,false);
    std::map<std::pair<std::uint32_t,std::uint32_t>,std::uint32_t> carrierByRoleRow;
    for(std::size_t k=0;k<nCarrier;++k){
        const auto key=std::pair{carrier.carrierRoleMaterialIndices[k],carrier.carrierRoleLocalRows[k]};
        if(!carrierByRoleRow.emplace(key,static_cast<std::uint32_t>(k)).second)
            return fail("paired carrier NPZ repeats a role-local carrier identity");
    }
    std::set<std::pair<std::uint32_t,std::uint32_t>> seenSupportPairs;
    for(const auto& row:evidenceRows){
        if(row.roleMaterial!=15u&&row.roleMaterial!=16u)
            return fail("current carrier witness sidecar has an unknown role material");
        const auto carrierIt=carrierByRoleRow.find({row.roleMaterial,row.roleLocalCarrierRow});
        if(carrierIt==carrierByRoleRow.end())
            return fail("current carrier witness sidecar references a missing carrier row");
        const auto carrierRow=carrierIt->second;
        if(!row.extraChildSlot&&seen[carrierRow])
            return fail("current carrier witness sidecar repeats a primary carrier row");
        if(!row.extraChildSlot)seen[carrierRow]=true;
        const auto parent=carrierParent[carrierRow];
        const auto child=row.childRow;
        if(!seenSupportPairs.emplace(carrierRow,child).second)
            return fail("current carrier witness repeats an exact carrier/child pair");
        const auto primaryChild=carrier.witnessChildRows[carrierRow];
        const auto constructionValid=row.extraChildSlot?
            row.carrierConstruction=="current_source_carrier_edge_support":
            (row.carrierConstruction=="inherited_090_quadrature"||
             row.carrierConstruction=="current_exact_tuple_projection_degree1_lp"||
             row.carrierConstruction=="source_tetra_anchor_refinement_1046d");
        if(!row.hasCSVProof||!physical_skin_child_mass_detail::validSHA256(row.sourceCSVSHA256)||
           !constructionValid||parent>=np||child>=nc||
           row.sourceParentRow!=carrier.globalSourceTetRows[parent]||
           row.material!=input.parentMaterialIndices[parent]||
           row.donor!=input.parentDonorBodyIndices[parent]||
           row.region!=input.parentAnatomicalRegionIndices[parent])
            return fail("current carrier witness sidecar does not match its exact current source parent and child rows");
        if(!row.extraChildSlot){
            if(child!=primaryChild||row.roleLocalChildRow!=carrier.carrierRoleLocalChildRows[carrierRow])
                return fail("primary carrier witness no longer matches its pinned NPZ target child");
        }else{
            if(row.mode!=CurrentCarrierWitnessMode::CurrentCarrierPointProjection||
               child==primaryChild||row.projectionCSVRow!=std::numeric_limits<std::uint32_t>::max()||
               row.declaredP1GroupID!=std::numeric_limits<std::uint32_t>::max())
                return fail("additional carrier child slot duplicates or changes the primary carrier target/provenance");
            const auto childRows=row.roleMaterial==15u?skinChildGlobalRows:fatChildGlobalRows;
            if(childRows.empty()||row.roleLocalChildRow>=childRows.size()||
               childRows[row.roleLocalChildRow]!=child)
                return fail("additional carrier child slot does not match the authenticated role-local child crosswalk");
        }
        if(!row.extraChildSlot){
            const auto childRows=row.roleMaterial==15u?skinChildGlobalRows:fatChildGlobalRows;
            if(!childRows.empty()&&(row.roleLocalChildRow>=childRows.size()||
               childRows[row.roleLocalChildRow]!=child))
                return fail("primary carrier target differs from the authenticated role-local child crosswalk");
        }
        if(row.material!=input.parentMaterialIndices[parent]||
           row.donor!=input.parentDonorBodyIndices[parent]||
           row.region!=input.parentAnatomicalRegionIndices[parent]||
           input.childMaterialIndices[child]!=row.material)
            return fail("current carrier witness has a material or exact parent-tuple mismatch");
        if(row.sourceParentLocalRow!=carrier.carrierRoleLocalParentRows[carrierRow])
            return fail("1046c carrier witness role-local parent row differs from the pinned role packet");
        const Tuple tuple{row.material,row.donor,row.region};
        const auto groupIt=groupByTuple.find(tuple);
        if(groupIt==groupByTuple.end()||input.childMaterialIndices[child]!=row.material)
            return fail("current carrier witness has no exact current material/donor/region source group");
        const auto group=groupIt->second;
        if(row.sourceParentLocalRow!=carrier.carrierRoleLocalParentRows[carrierRow])
            return fail("1046c carrier witness role-local parent row differs from the pinned role packet");
        const auto regionBegin=support.cellAdmissibleRegionOffsets[child];
        const auto regionEnd=support.cellAdmissibleRegionOffsets[child+1u];
        if(regionBegin>=regionEnd||regionEnd>support.cellAdmissibleRegionIds.size()||
           !std::binary_search(support.cellAdmissibleRegionIds.begin()+regionBegin,
                               support.cellAdmissibleRegionIds.begin()+regionEnd,row.region))
            return fail("current carrier witness source region is outside the authenticated one-hop child region support");
        if(row.mode==CurrentCarrierWitnessMode::P1ExactTupleSlot){
            bool found=false;
            for(std::uint32_t slot=support.childOffsets[child];slot<support.childOffsets[child+1u];++slot)
                found|=support.slotGroupIndices[slot]==group;
            if(!found)return fail("carrier sidecar labels a non-P1 tuple as a P1 exact-slot witness");
        }else if(row.mode==CurrentCarrierWitnessMode::AuthenticatedExactProjection){
            if(row.projectionCSVRow>=exactProjectionRows.size())
                return fail("carrier sidecar projection row index is outside the pinned 1044 exact projection CSV");
            const auto& projection=exactProjectionRows[row.projectionCSVRow];
            if(projection.csvRow!=row.projectionCSVRow||
               projection.sourceCSVSHA256!=exactProjectionCSVSHA256||
               !projection.referenceBasisAllowed||
               projection.sourceParentRow!=row.sourceParentRow||
               projection.childRow!=child||projection.material!=row.material||
               projection.geometricDonor!=row.donor||projection.anatomicalRegion!=row.region)
                return fail("carrier sidecar projection row lacks exact current/reference source geometry and tuple binding");
        }else if(row.mode!=CurrentCarrierWitnessMode::CurrentCarrierPointProjection){
            return fail("carrier sidecar uses an unknown tuple-support evidence mode");
        }
        if(row.extraChildSlot&&row.mode!=CurrentCarrierWitnessMode::CurrentCarrierPointProjection)
            return fail("additional carrier child slot is not an authenticated carrier-point projection");
        const auto& packetBary=carrier.carrierBarycentric[carrierRow];
        const auto& packetTet=carrier.parentReferenceTetWorldVertices[parent];
        const double packetWeight=carrier.carrierWeights[carrierRow];
        if(!std::isfinite(packetWeight)||packetWeight<=0.0||
           std::abs(packetWeight-row.claimedWeight)>std::max(1.0e-14,64.0*std::numeric_limits<double>::epsilon()*packetWeight))
            return fail("1046c carrier witness weight differs from the pinned NPZ carrier");
        for(std::size_t axis=0;axis<4u;++axis)
            if(!std::isfinite(row.claimedBarycentric[axis])||
               std::abs(packetBary[axis]-row.claimedBarycentric[axis])>2.0e-12)
                return fail("1046c carrier witness barycentric row differs from the pinned NPZ carrier");
        const auto claimedSource=source_carrier_transport_detail::barycentricPoint(packetTet,packetBary);
        const auto claimedChild=input.childCentroidsWorld[child];
        const auto claimedDistance=source_carrier_transport_detail::pointDistance(claimedSource,claimedChild);
        if(source_carrier_transport_detail::pointDistance(claimedSource,row.claimedSourcePointWorld)>5.0e-9||
           source_carrier_transport_detail::pointDistance(claimedChild,row.claimedChildCentroidWorld)>5.0e-9||
           !std::isfinite(row.claimedDistanceM)||row.claimedDistanceM<0.0||
           std::abs(claimedDistance-row.claimedDistanceM)>5.0e-9)
            return fail("1046c carrier witness source/child coordinate or distance does not replay");
        if(row.mode==CurrentCarrierWitnessMode::CurrentCarrierPointProjection){
            const double weight=carrier.carrierWeights[carrierRow];
            const auto& bary=carrier.carrierBarycentric[carrierRow];
            if(!std::isfinite(weight)||weight<=0.0||
               !std::all_of(bary.begin(),bary.end(),[](double x){
                   return std::isfinite(x)&&x>=-2.0e-12&&x<=1.0+2.0e-12;
               }))
                return fail("current carrier point witness has a nonpositive weight or invalid barycentric coordinate");
            const double barySum=std::accumulate(bary.begin(),bary.end(),0.0);
            if(std::abs(barySum-1.0)>2.0e-12)
                return fail("current carrier point witness barycentric coordinates do not sum to one");
            const auto point=source_carrier_transport_detail::barycentricPoint(
                carrier.parentReferenceTetWorldVertices[parent],bary);
            const auto distance=source_carrier_transport_detail::pointDistance(
                point,input.childCentroidsWorld[child]);
            const double tolerance=std::max(1.0e-12,64.0*std::numeric_limits<double>::epsilon()*
                std::max({1.0,distance,input.maximumTransportDistanceM}));
            if(!std::isfinite(distance)||distance>input.maximumTransportDistanceM+tolerance)
                return fail("current carrier point-to-child-centroid witness exceeds the unchanged transport radius");
            auto& childGroups=allowed[child];
            if(!std::binary_search(childGroups.begin(),childGroups.end(),group)){
                childGroups.insert(std::lower_bound(childGroups.begin(),childGroups.end(),group),group);
            }
        }
    }
    if(std::find(seen.begin(),seen.end(),false)!=seen.end())
        return fail("current carrier witness sidecar does not cover every primary paired carrier row");
    support.childAdmissibleGroupOffsets.assign(nc+1u,0u);
    support.childAdmissibleGroupIndices.clear();
    for(std::size_t c=0;c<nc;++c){
        auto& row=allowed[c];
        if(row.empty())return fail("current carrier witness exact allowlist leaves a child without source tuple support");
        support.childAdmissibleGroupIndices.insert(support.childAdmissibleGroupIndices.end(),
                                                   row.begin(),row.end());
        support.childAdmissibleGroupOffsets[c+1u]=
            static_cast<std::uint32_t>(support.childAdmissibleGroupIndices.size());
    }
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("numi.human.current-fractional-exact-group-carrier-evidence.v1");
    hash.string(support.admissibleGroupEvidenceSHA256);
    hash.string(evidenceSHA256);
    hash.string(carrier.sourceCarrierPacketSHA256);
    support.admissibleGroupEvidenceSHA256=hash.finish();
    error.clear();return true;
}


struct CurrentSourceCentroidSlotWitnessRow {
    // Both rows are in the current, pinned source-330 parent and combined
    // current-433 child domains. The parent point is recomputed as the positive
    // (1/4,1/4,1/4,1/4) barycentre of its exact source tetrahedron.
    std::uint32_t sourceParentRow=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t receiverChildRow=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t material=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t donor=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t region=std::numeric_limits<std::uint32_t>::max();
    std::array<double,4> barycentric{};
    std::array<double,3> claimedSourceCentroidWorld{};
    std::array<double,3> claimedChildCentroidWorld{};
    double claimedDistanceM=0.0;
    std::uint32_t oneHopSourceRegion=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t oneHopMaxHops=0u;
    std::uint32_t childMaterial=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t childDonor=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t childRegion=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t csvRow=std::numeric_limits<std::uint32_t>::max();
    std::string sourceCSV_SHA256;
    bool hasCSVProof=false;
};

[[nodiscard]] inline bool loadCurrentSourceCentroidSlotWitnessCSV(
        const std::filesystem::path& path,const std::string_view expectedSHA256,
        std::vector<CurrentSourceCentroidSlotWitnessRow>& rows,std::string& error){
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(!physical_skin_child_mass_detail::validSHA256(expectedSHA256))
        return fail("source-centroid witness loader requires a valid explicit CSV SHA");
    std::string actualSHA;
    if(!physical_skin_child_mass_detail::sha256File(path,actualSHA,error))return false;
    if(actualSHA!=expectedSHA256)return fail("source-centroid witness CSV differs from its explicit SHA pin");
    std::ifstream input(path);
    if(!input)return fail("cannot open current source-centroid witness CSV");
    std::string line;
    if(!std::getline(input,line))return fail("current source-centroid witness CSV is empty");
    const auto header=physical_skin_child_mass_detail::splitCSV(line);
    static const std::vector<std::string> expectedHeader{
        "source_parent_row","receiver_child_tet_row","source_material","source_donor","source_region",
        "source_barycentric0","source_barycentric1","source_barycentric2","source_barycentric3",
        "source_centroid_x_m","source_centroid_y_m","source_centroid_z_m",
        "receiver_child_centroid_x_m","receiver_child_centroid_y_m","receiver_child_centroid_z_m",
        "source_to_current_child_centroid_distance_m","one_hop_source_region",
        "one_hop_max_source_region_hops","receiver_current_material","receiver_current_donor",
        "receiver_current_region","capacity_reason_bits"};
    if(header!=expectedHeader)return fail("current source-centroid witness CSV header is not the pinned 22-column schema");
    std::vector<CurrentSourceCentroidSlotWitnessRow> next;
    next.reserve(500000u);
    std::set<std::pair<std::uint32_t,std::uint32_t>> unique;
    std::size_t physicalLine=1u;
    while(std::getline(input,line)){
        ++physicalLine;if(line.empty())continue;
        const auto f=physical_skin_child_mass_detail::splitCSV(line);
        if(f.size()!=expectedHeader.size())
            return fail("source-centroid witness row has an unexpected column count at line "+std::to_string(physicalLine));
        CurrentSourceCentroidSlotWitnessRow row;
        std::uint32_t ignoredReasonBits=0u;
        if(!physical_skin_child_mass_detail::parseNumber(f[0],row.sourceParentRow)||
           !physical_skin_child_mass_detail::parseNumber(f[1],row.receiverChildRow)||
           !physical_skin_child_mass_detail::parseNumber(f[2],row.material)||
           !physical_skin_child_mass_detail::parseNumber(f[3],row.donor)||
           !physical_skin_child_mass_detail::parseNumber(f[4],row.region)||
           !physical_skin_child_mass_detail::parseNumber(f[5],row.barycentric[0])||
           !physical_skin_child_mass_detail::parseNumber(f[6],row.barycentric[1])||
           !physical_skin_child_mass_detail::parseNumber(f[7],row.barycentric[2])||
           !physical_skin_child_mass_detail::parseNumber(f[8],row.barycentric[3])||
           !physical_skin_child_mass_detail::parseNumber(f[9],row.claimedSourceCentroidWorld[0])||
           !physical_skin_child_mass_detail::parseNumber(f[10],row.claimedSourceCentroidWorld[1])||
           !physical_skin_child_mass_detail::parseNumber(f[11],row.claimedSourceCentroidWorld[2])||
           !physical_skin_child_mass_detail::parseNumber(f[12],row.claimedChildCentroidWorld[0])||
           !physical_skin_child_mass_detail::parseNumber(f[13],row.claimedChildCentroidWorld[1])||
           !physical_skin_child_mass_detail::parseNumber(f[14],row.claimedChildCentroidWorld[2])||
           !physical_skin_child_mass_detail::parseNumber(f[15],row.claimedDistanceM)||
           !physical_skin_child_mass_detail::parseNumber(f[16],row.oneHopSourceRegion)||
           !physical_skin_child_mass_detail::parseNumber(f[17],row.oneHopMaxHops)||
           !physical_skin_child_mass_detail::parseNumber(f[18],row.childMaterial)||
           !physical_skin_child_mass_detail::parseNumber(f[19],row.childDonor)||
           !physical_skin_child_mass_detail::parseNumber(f[20],row.childRegion)||
           !physical_skin_child_mass_detail::parseNumber(f[21],ignoredReasonBits)||
           !unique.emplace(row.sourceParentRow,row.receiverChildRow).second||
           !std::isfinite(row.claimedDistanceM)||row.claimedDistanceM<0.0||
           !std::all_of(row.barycentric.begin(),row.barycentric.end(),
                        [](double x){return std::isfinite(x);})||
           !std::all_of(row.claimedSourceCentroidWorld.begin(),row.claimedSourceCentroidWorld.end(),
                        [](double x){return std::isfinite(x);})||
           !std::all_of(row.claimedChildCentroidWorld.begin(),row.claimedChildCentroidWorld.end(),
                        [](double x){return std::isfinite(x);}))
            return fail("current source-centroid witness row has invalid numeric data or duplicate row pair at line "+std::to_string(physicalLine));
        row.csvRow=static_cast<std::uint32_t>(next.size());row.sourceCSV_SHA256=actualSHA;row.hasCSVProof=true;
        next.push_back(std::move(row));
    }
    if(!input.eof()||next.empty())return fail("current source-centroid witness CSV failed its complete nonempty read");
    rows=std::move(next);error.clear();return true;
}

// Bind a narrowly enumerated support extension from current source-cell
// centroids. This changes only the exact tuple allowlist; it leaves P1 fractions,
// semantic labels, source mass and graph distance gates unchanged.
[[nodiscard]] inline bool bindCurrentSourceCentroidSlotWitnesses(
        ConservativeSpatialChildSemanticSupportData& support,
        const ConservativeSpatialMassTransportInput& input,
        const PhysicalSkinParentMassPacket& parent,
        const std::span<const CurrentSourceCentroidSlotWitnessRow> witnesses,
        const std::string_view evidenceSHA256,std::string& error){
    using Tuple=std::tuple<std::uint32_t,std::uint32_t,std::uint32_t>;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const std::size_t np=input.parentMassKg.size(),nc=input.childReferenceVolumeM3.size();
    const double maxDistance=input.maximumTransportDistanceM;
    if(witnesses.empty()||!physical_skin_child_mass_detail::validSHA256(evidenceSHA256)||
       support.groupMaterialIndices.empty()||
       support.groupMaterialIndices.size()!=support.donorBodyIndices.size()||
       support.groupMaterialIndices.size()!=support.anatomicalRegionIndices.size()||
       support.sourceParentTetOffsets.size()!=support.groupMaterialIndices.size()+1u||
       support.sourceParentTetOffsets.empty()||support.sourceParentTetOffsets.front()!=0u||
       support.sourceParentTetOffsets.back()!=support.sourceParentTetRows.size()||
       support.childOffsets.size()!=nc+1u||support.slotGroupIndices.size()!=support.volumeFractions.size()||
       support.childOffsets.empty()||support.childOffsets.front()!=0u||
       support.childOffsets.back()!=support.slotGroupIndices.size()||
       support.cellAdmissibleRegionOffsets.size()!=nc+1u||
       support.cellAdmissibleRegionOffsets.empty()||support.cellAdmissibleRegionOffsets.front()!=0u||
       support.cellAdmissibleRegionOffsets.back()!=support.cellAdmissibleRegionIds.size()||
       support.admissibilityMaximumSourceRegionHops!=1u||
       !physical_skin_child_mass_detail::validSHA256(support.admissibilitySourceSupportSHA256)||
       !physical_skin_child_mass_detail::validSHA256(parent.parentMassPacketSHA256)||
       input.parentCentroidsWorld.size()!=np||input.parentMaterialIndices.size()!=np||
       input.parentDonorBodyIndices.size()!=np||input.parentAnatomicalRegionIndices.size()!=np||
       input.childCentroidsWorld.size()!=nc||input.childMaterialIndices.size()!=nc||
       input.childDonorBodyIndices.size()!=nc||input.childAnatomicalRegionIndices.size()!=nc||
       parent.tetrahedra.size()!=np||parent.referenceWorldPoints.empty()||
       !std::isfinite(maxDistance)||maxDistance<=0.0)
        return fail("source-centroid allowlist requires aligned current 330/433 domains and authenticated one-hop support");
    std::map<Tuple,std::uint32_t> groupByTuple;
    std::vector<std::uint32_t> groupForParent(np,std::numeric_limits<std::uint32_t>::max());
    for(std::size_t g=0;g<support.groupMaterialIndices.size();++g){
        const Tuple key{support.groupMaterialIndices[g],support.donorBodyIndices[g],
                        support.anatomicalRegionIndices[g]};
        if(!groupByTuple.emplace(key,static_cast<std::uint32_t>(g)).second)
            return fail("source-centroid allowlist has duplicate exact semantic groups");
        const auto begin=support.sourceParentTetOffsets[g],end=support.sourceParentTetOffsets[g+1u];
        if(begin>=end||end>support.sourceParentTetRows.size())
            return fail("source-centroid allowlist group has an empty or invalid current parent CSR row");
        for(std::uint32_t k=begin;k<end;++k){
            const auto p=support.sourceParentTetRows[k];
            if(p>=np||groupForParent[p]!=std::numeric_limits<std::uint32_t>::max()||
               input.parentMaterialIndices[p]!=std::get<0>(key)||
               input.parentDonorBodyIndices[p]!=std::get<1>(key)||
               input.parentAnatomicalRegionIndices[p]!=std::get<2>(key))
                return fail("source-centroid allowlist source-parent CSR disagrees with current exact tuple ownership");
            groupForParent[p]=static_cast<std::uint32_t>(g);
        }
    }
    std::vector<std::vector<std::uint32_t>> allowed(nc);
    const bool explicitAllowlist=!support.childAdmissibleGroupOffsets.empty()||
                                 !support.childAdmissibleGroupIndices.empty();
    if(explicitAllowlist){
        if(support.childAdmissibleGroupOffsets.size()!=nc+1u||
           support.childAdmissibleGroupOffsets.front()!=0u||
           support.childAdmissibleGroupOffsets.back()!=support.childAdmissibleGroupIndices.size())
            return fail("source-centroid allowlist existing exact-group CSR is malformed");
        for(std::size_t c=0;c<nc;++c){
            const auto begin=support.childAdmissibleGroupOffsets[c];
            const auto end=support.childAdmissibleGroupOffsets[c+1u];
            if(begin>=end||end>support.childAdmissibleGroupIndices.size())
                return fail("source-centroid allowlist existing exact-group CSR contains an empty row");
            allowed[c].assign(support.childAdmissibleGroupIndices.begin()+begin,
                              support.childAdmissibleGroupIndices.begin()+end);
        }
    }else{
        for(std::size_t c=0;c<nc;++c)
            for(std::uint32_t k=support.childOffsets[c];k<support.childOffsets[c+1u];++k)
                allowed[c].push_back(support.slotGroupIndices[k]);
    }
    for(std::size_t c=0;c<nc;++c){
        auto& row=allowed[c];
        std::sort(row.begin(),row.end());
        if(std::adjacent_find(row.begin(),row.end())!=row.end())
            return fail("source-centroid allowlist existing exact-group row repeats a group");
        for(std::uint32_t k=support.childOffsets[c];k<support.childOffsets[c+1u];++k){
            const auto g=support.slotGroupIndices[k];
            if(g>=support.groupMaterialIndices.size()||!std::isfinite(support.volumeFractions[k])||
               support.volumeFractions[k]<=0.0||support.volumeFractions[k]>1.0||
               !std::binary_search(row.begin(),row.end(),g))
                return fail("source-centroid allowlist does not preserve every positive P1 preference slot");
        }
    }
    std::set<std::pair<std::uint32_t,std::uint32_t>> seen;
    spatial_mass_transport_detail::SHA256Builder rowDigest;
    rowDigest.string("numi.human.current-source-centroid-slot-witness-rows.v1");
    rowDigest.string(evidenceSHA256);
    for(const auto& witness:witnesses){
        const auto& row=witness;
        const auto p=row.sourceParentRow,c=row.receiverChildRow;
        if(p>=np||c>=nc||!seen.emplace(p,c).second||
           !row.hasCSVProof||row.csvRow>=witnesses.size()||
           row.sourceCSV_SHA256!=evidenceSHA256)
            return fail("source-centroid witness has an out-of-range, repeated or unpinned source/receiver pair");
        const auto group=groupForParent[p];
        if(group>=support.groupMaterialIndices.size()||
           input.parentMaterialIndices[p]!=input.childMaterialIndices[c])
            return fail("source-centroid witness crosses source/child material or lacks exact parent-group ownership");
        const auto regionBegin=support.cellAdmissibleRegionOffsets[c];
        const auto regionEnd=support.cellAdmissibleRegionOffsets[c+1u];
        if(regionBegin>=regionEnd||regionEnd>support.cellAdmissibleRegionIds.size()||
           !std::is_sorted(support.cellAdmissibleRegionIds.begin()+regionBegin,
                           support.cellAdmissibleRegionIds.begin()+regionEnd)||
           std::adjacent_find(support.cellAdmissibleRegionIds.begin()+regionBegin,
                              support.cellAdmissibleRegionIds.begin()+regionEnd)!=
               support.cellAdmissibleRegionIds.begin()+regionEnd||
           !std::binary_search(support.cellAdmissibleRegionIds.begin()+regionBegin,
                               support.cellAdmissibleRegionIds.begin()+regionEnd,
                               input.parentAnatomicalRegionIndices[p]))
            return fail("source-centroid witness source region is outside the authenticated one-hop child support");
        if(row.material!=input.parentMaterialIndices[p]||
           row.donor!=input.parentDonorBodyIndices[p]||
           row.region!=input.parentAnatomicalRegionIndices[p]||
           row.childMaterial!=input.childMaterialIndices[c]||
           row.childDonor!=input.childDonorBodyIndices[c]||
           row.childRegion!=input.childAnatomicalRegionIndices[c]||
           row.oneHopSourceRegion!=row.region||row.oneHopMaxHops!=1u||
           row.barycentric!=std::array<double,4>{0.25,0.25,0.25,0.25})
            return fail("source-centroid witness CSV tuple, barycentre or child label assertion differs from current rows");
        const auto tet=parent.tetrahedra[p];
        std::array<std::array<double,3>,4> vertices{};
        for(std::size_t corner=0;corner<4u;++corner){
            if(tet[corner]>=parent.referenceWorldPoints.size())
                return fail("source-centroid witness source tet exceeds the pinned current source330 point domain");
            vertices[corner]=parent.referenceWorldPoints[tet[corner]];
            if(!std::all_of(vertices[corner].begin(),vertices[corner].end(),
                             [](double x){return std::isfinite(x);}))
                return fail("source-centroid witness source tet has a nonfinite reference coordinate");
        }
        constexpr std::array<double,4> quarter{0.25,0.25,0.25,0.25};
        const auto sourceCentroid=source_carrier_transport_detail::barycentricPoint(vertices,quarter);
        const auto parentCentroidError=source_carrier_transport_detail::pointDistance(
            sourceCentroid,input.parentCentroidsWorld[p]);
        if(!std::isfinite(parentCentroidError)||parentCentroidError>1.0e-11)
            return fail("source-centroid witness differs from the current parent transport centroid");
        const auto distance=source_carrier_transport_detail::pointDistance(
            sourceCentroid,input.childCentroidsWorld[c]);
        const double tolerance=std::max(1.0e-12,64.0*std::numeric_limits<double>::epsilon()*
            std::max({1.0,distance,maxDistance}));
        if(!std::isfinite(distance)||distance>maxDistance+tolerance||
           source_carrier_transport_detail::pointDistance(sourceCentroid,row.claimedSourceCentroidWorld)>5.0e-9||
           source_carrier_transport_detail::pointDistance(input.childCentroidsWorld[c],row.claimedChildCentroidWorld)>5.0e-9||
           std::abs(distance-row.claimedDistanceM)>5.0e-9)
            return fail("source-centroid witness does not replay its claimed source/child coordinates or unchanged radius");
        auto& childGroups=allowed[c];
        if(!std::binary_search(childGroups.begin(),childGroups.end(),group))
            childGroups.insert(std::lower_bound(childGroups.begin(),childGroups.end(),group),group);
        rowDigest.u32(p);rowDigest.u32(c);rowDigest.u32(group);
        rowDigest.real(distance);
    }
    support.childAdmissibleGroupOffsets.assign(nc+1u,0u);
    support.childAdmissibleGroupIndices.clear();
    for(std::size_t c=0;c<nc;++c){
        auto& row=allowed[c];
        if(row.empty())return fail("source-centroid exact-group allowlist leaves a child without source support");
        support.childAdmissibleGroupIndices.insert(support.childAdmissibleGroupIndices.end(),
                                                   row.begin(),row.end());
        support.childAdmissibleGroupOffsets[c+1u]=
            static_cast<std::uint32_t>(support.childAdmissibleGroupIndices.size());
    }
    spatial_mass_transport_detail::SHA256Builder digest;
    digest.string("numi.human.current-fractional-exact-group-centroid-evidence.v1");
    digest.string(support.admissibleGroupEvidenceSHA256);
    digest.string(evidenceSHA256);
    digest.string(parent.parentMassPacketSHA256);
    digest.string(rowDigest.finish());
    support.admissibleGroupEvidenceSHA256=digest.finish();
    error.clear();return true;
}


// Caller-owned evidence paths for the one-solve native mapping route. P1 NPZs
// and the paired carrier NPZ are loaded separately with their full current pins;
// this callback binds the two current CSV proofs to the owner-built transport
// input before either solve or deterministic replay begins.
struct CurrentFractionalSemanticProofFiles {
    std::filesystem::path exactProjectionCSVPath,sourceCentroidSlotsCSVPath;
    std::filesystem::path skinCarrierWitnessCSVPath,fatCarrierWitnessCSVPath;
    std::string exactProjectionCSV_SHA256,sourceCentroidSlotsCSV_SHA256;
    std::string skinCarrierWitnessCSV_SHA256,fatCarrierWitnessCSV_SHA256;
    const CurrentSourceCarrierQuadratureData* carriers=nullptr;
    std::vector<CurrentExactTupleProjectionRow> exactProjectionRows;
    std::vector<CurrentExactTupleProjectionReject> referenceBasisRejects;
    std::vector<CurrentSourceCentroidSlotWitnessRow> sourceCentroidRows;
    std::vector<CurrentCarrierWitnessEvidenceRow> carrierWitnessRows;
    std::vector<std::uint32_t> skinChildGlobalRows,fatChildGlobalRows;
    CurrentCarrierWitnessCSVSummary skinCarrierSummary,fatCarrierSummary;
    std::string combinedCarrierWitnessSHA256;
};

[[nodiscard]] inline bool bindCurrentFractionalSemanticSupportProofs(
        void* opaqueContext,ConservativeSpatialChildSemanticSupportData& support,
        const ConservativeSpatialMassTransportInput& input,
        const PhysicalSkinParentMassPacket& parent,std::string& error){
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    auto* context=static_cast<CurrentFractionalSemanticProofFiles*>(opaqueContext);
    if(context==nullptr||context->carriers==nullptr||
       context->exactProjectionCSVPath.empty()||context->sourceCentroidSlotsCSVPath.empty()||
       context->skinCarrierWitnessCSVPath.empty()||context->fatCarrierWitnessCSVPath.empty()||
       !physical_skin_child_mass_detail::validSHA256(context->exactProjectionCSV_SHA256)||
       !physical_skin_child_mass_detail::validSHA256(context->sourceCentroidSlotsCSV_SHA256)||
       !physical_skin_child_mass_detail::validSHA256(context->skinCarrierWitnessCSV_SHA256)||
       !physical_skin_child_mass_detail::validSHA256(context->fatCarrierWitnessCSV_SHA256))
        return fail("current fractional proof callback lacks a complete explicit path/SHA/carrier input set");
    std::vector<CurrentExactTupleProjectionRow> projections;
    std::vector<CurrentExactTupleProjectionReject> restRejects;
    if(!loadCurrentExactTupleProjectionCSV(context->exactProjectionCSVPath,
            context->exactProjectionCSV_SHA256,input,parent,input.maximumTransportDistanceM,
            projections,error,&restRejects))
        return fail("current exact-projection CSV replay failed: "+error);
    std::vector<CurrentSourceCentroidSlotWitnessRow> centroidRows;
    if(!loadCurrentSourceCentroidSlotWitnessCSV(context->sourceCentroidSlotsCSVPath,
            context->sourceCentroidSlotsCSV_SHA256,centroidRows,error))
        return fail("current source-centroid slot CSV failed its pinned read: "+error);
    std::vector<CurrentCarrierWitnessEvidenceRow> carrierRows;
    CurrentCarrierWitnessCSVSummary skinSummary,fatSummary;
    if(!loadCurrentCarrierWitnessCSV(context->skinCarrierWitnessCSVPath,
            context->skinCarrierWitnessCSV_SHA256,15u,carrierRows,skinSummary,error))
        return fail("current Skin 1046c carrier witness CSV failed its pinned read: "+error);
    std::vector<CurrentCarrierWitnessEvidenceRow> fatRows;
    if(!loadCurrentCarrierWitnessCSV(context->fatCarrierWitnessCSVPath,
            context->fatCarrierWitnessCSV_SHA256,16u,fatRows,fatSummary,error))
        return fail("current Fat 1046c carrier witness CSV failed its pinned read: "+error);
    if(skinSummary.carrierRecordCount==0u||fatSummary.carrierRecordCount==0u||
       skinSummary.centroidFallbackRecordCount>skinSummary.carrierRecordCount||
       fatSummary.centroidFallbackRecordCount>fatSummary.carrierRecordCount)
        return fail("current 1046c role witness counts are empty or inconsistent");
    carrierRows.insert(carrierRows.end(),fatRows.begin(),fatRows.end());
    spatial_mass_transport_detail::SHA256Builder carrierEvidenceHash;
    carrierEvidenceHash.string("numi.human.current-paired-1046c-carrier-witness-csvs.v1");
    carrierEvidenceHash.string(skinSummary.sourceCSVSHA256);
    carrierEvidenceHash.string(fatSummary.sourceCSVSHA256);
    const std::string combinedCarrierSHA=carrierEvidenceHash.finish();

    if(!bindCurrentExactTupleProjectionRows(support,input,parent,projections,
                                             context->exactProjectionCSV_SHA256,error))
        return fail("current exact-projection proof binding failed: "+error);
    if(!bindCurrentSourceCentroidSlotWitnesses(support,input,parent,centroidRows,
                                                context->sourceCentroidSlotsCSV_SHA256,error))
        return fail("current 008 source-centroid slot proof binding failed: "+error);
    // Every sidecar row is independently pinned by role. This avoids treating
    // the combined digest as a substitute for either source file's SHA.
    for(const auto& row:carrierRows){
        const auto& expected=row.roleMaterial==15u?skinSummary.sourceCSVSHA256:fatSummary.sourceCSVSHA256;
        if(row.sourceCSVSHA256!=expected)
            return fail("current 1046c carrier witness role does not match its explicit sidecar file pin");
    }
    if(!bindCurrentCarrierWitnessAllowlist(support,input,*context->carriers,
            carrierRows,projections,context->exactProjectionCSV_SHA256,
            combinedCarrierSHA,error,context->skinChildGlobalRows,
            context->fatChildGlobalRows))
        return fail("current 1046c carrier tuple witnesses failed exact support binding: "+error);
    context->exactProjectionRows=std::move(projections);
    context->referenceBasisRejects=std::move(restRejects);
    context->sourceCentroidRows=std::move(centroidRows);
    context->carrierWitnessRows=std::move(carrierRows);
    context->skinCarrierSummary=std::move(skinSummary);
    context->fatCarrierSummary=std::move(fatSummary);
    context->combinedCarrierWitnessSHA256=combinedCarrierSHA;
    error.clear();return true;
}
} // namespace metalrobo::human::coarse_tissue
