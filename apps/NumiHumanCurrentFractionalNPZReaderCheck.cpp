#include "NumiHumanCurrentFractionalSupportNPZ.hpp"
#include <cassert>
#include <iostream>

using namespace metalrobo::human::coarse_tissue::current_fractional_support_npz_detail;
namespace {
void put16(std::vector<std::uint8_t>& b,std::uint16_t v) {
    b.push_back(v&255u);b.push_back((v>>8u)&255u);
}
void put32(std::vector<std::uint8_t>& b,std::uint32_t v) {
    put16(b,v&65535u);put16(b,v>>16u);
}
std::vector<std::uint8_t> npy() {
    std::string header="{'descr': '<u4', 'fortran_order': False, 'shape': (3,), }";
    while((10u+header.size()+1u)%64u)header+=' ';
    header+='\n';
    std::vector<std::uint8_t> b{0x93u,'N','U','M','P','Y',1u,0u};
    put16(b,static_cast<std::uint16_t>(header.size()));
    b.insert(b.end(),header.begin(),header.end());
    for(auto v:{1u,2u,3u})put32(b,v);
    return b;
}
std::vector<std::uint8_t> zip(const std::vector<std::uint8_t>& raw,bool compressed,bool badCRC=false) {
    std::vector<std::uint8_t> data=raw;
    if(compressed) {
        data.resize(compressBound(raw.size()));z_stream z{};
        assert(deflateInit2(&z,Z_DEFAULT_COMPRESSION,Z_DEFLATED,-MAX_WBITS,8,Z_DEFAULT_STRATEGY)==Z_OK);
        z.next_in=const_cast<Bytef*>(raw.data());z.avail_in=raw.size();
        z.next_out=data.data();z.avail_out=data.size();
        assert(deflate(&z,Z_FINISH)==Z_STREAM_END);data.resize(z.total_out);deflateEnd(&z);
    }
    auto crc=static_cast<std::uint32_t>(crc32(0L,raw.data(),raw.size()))+(badCRC?1u:0u);
    const std::string name="rows.npy";std::vector<std::uint8_t>b;
    put32(b,0x04034b50u);put16(b,20);put16(b,0);put16(b,compressed?8:0);
    put32(b,0);put32(b,crc);put32(b,data.size());put32(b,raw.size());
    put16(b,name.size());put16(b,0);b.insert(b.end(),name.begin(),name.end());b.insert(b.end(),data.begin(),data.end());
    const auto offset=b.size();
    put32(b,0x02014b50u);put16(b,20);put16(b,20);put16(b,0);put16(b,compressed?8:0);
    put32(b,0);put32(b,crc);put32(b,data.size());put32(b,raw.size());put16(b,name.size());
    put16(b,0);put16(b,0);put16(b,0);put16(b,0);put32(b,0);put32(b,0);
    b.insert(b.end(),name.begin(),name.end());const auto directorySize=b.size()-offset;
    put32(b,0x06054b50u);put16(b,0);put16(b,0);put16(b,1);put16(b,1);
    put32(b,directorySize);put32(b,offset);put16(b,0);return b;
}
}
int main() {
    const auto path=std::filesystem::temp_directory_path()/
        ("numi-fractional-reader-"+std::to_string(getpid())+".npz");
    const auto write=[&](const auto& bytes) {
        std::ofstream f(path,std::ios::binary|std::ios::trunc);
        f.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());assert(f.good());
    };
    std::string error;auto raw=npy();const auto* allocation=raw.data();NpyArray direct;
    assert(parseNpy(std::move(raw),direct,error)&&direct.bytes.data()==allocation);
    std::vector<std::uint32_t> values;
    assert(readU32(direct,values,error)&&(values==std::vector<std::uint32_t>{1,2,3}));
    for(bool compressed:{false,true}) {
        std::map<std::string,NpyArray> arrays;write(zip(npy(),compressed));
        assert(readRequestedArrays(path,{"rows"},arrays,error));
        assert(arrays.at("rows").bytes==direct.bytes&&arrays.at("rows").shape==direct.shape);
        arrays.clear();assert(!readRequestedArrays(path,{"missing"},arrays,error));
        arrays.clear();write(zip(npy(),compressed,true));
        assert(!readRequestedArrays(path,{"rows"},arrays,error)&&error.find("CRC")!=std::string::npos);
    }
    auto truncated=zip(npy(),true);truncated.resize(12);write(truncated);
    std::map<std::string,NpyArray> arrays;assert(!readRequestedArrays(path,{"rows"},arrays,error));
    raw=npy();raw.pop_back();assert(!parseNpy(std::move(raw),direct,error));
    std::filesystem::remove(path);
    std::cout<<"fractional_npz_reader PASS stored deflated exact_values moved_allocation crc missing truncated shape\n";
}
