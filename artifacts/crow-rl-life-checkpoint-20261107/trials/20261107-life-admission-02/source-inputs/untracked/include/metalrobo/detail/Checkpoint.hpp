#pragma once
// Publication-boundary serialization only; never called from a device hot loop.
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <array>
#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unistd.h>
#include <vector>
namespace metalrobo::checkpoint {
using Bytes=std::vector<std::byte>;
using Digest=std::array<unsigned char,CC_SHA256_DIGEST_LENGTH>;
inline void require(bool value,const char* message) {if (!value) throw std::runtime_error(message);}
inline Digest digest(std::span<const std::byte> bytes) {
    CC_SHA256_CTX context;CC_SHA256_Init(&context);
    for (std::size_t offset=0;offset<bytes.size();) {
        const auto count=std::min<std::size_t>(bytes.size()-offset,1u<<20);
        CC_SHA256_Update(&context,bytes.data()+offset,static_cast<CC_LONG>(count));offset+=count;
    }
    Digest result;CC_SHA256_Final(result.data(),&context);return result;
}
inline Digest fileDigest(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);require(bool(input),"Checkpoint identity file unavailable");
    CC_SHA256_CTX context;CC_SHA256_Init(&context);std::array<char,65536> bytes;
    while (input) {input.read(bytes.data(),bytes.size());CC_SHA256_Update(&context,bytes.data(),static_cast<CC_LONG>(input.gcount()));}
    require(input.eof(),"Checkpoint identity file read failed");Digest result;CC_SHA256_Final(result.data(),&context);return result;
}
template<class T> inline void append(Bytes& bytes,const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto data=std::as_bytes(std::span(&value,1));bytes.insert(bytes.end(),data.begin(),data.end());
}
inline void appendBlob(Bytes& bytes,std::span<const std::byte> value) {
    append(bytes,std::uint64_t(value.size()));bytes.insert(bytes.end(),value.begin(),value.end());
}
struct Reader {
    std::span<const std::byte> remaining;
    std::span<const std::byte> take(std::size_t count) {
        require(count<=remaining.size(),"Truncated checkpoint section");auto value=remaining.first(count);remaining=remaining.subspan(count);return value;
    }
    template<class T> T get() {static_assert(std::is_trivially_copyable_v<T>);T value;const auto data=take(sizeof(T));std::memcpy(&value,data.data(),sizeof(T));return value;}
    std::span<const std::byte> blob() {const auto count=get<std::uint64_t>();require(count<=remaining.size(),"Invalid checkpoint section length");return take(static_cast<std::size_t>(count));}
    void end() const {require(remaining.empty(),"Trailing checkpoint data");}
};
struct Header {
    std::array<char,8> kind{};
    std::uint32_t abi=1,reserved=0;
    std::uint64_t identity=0,bytes=0;
    Digest sha{};
};
static_assert(sizeof(Header)==64 && std::endian::native==std::endian::little);
inline Bytes seal(std::array<char,8> kind,std::uint64_t identity,std::span<const std::byte> payload) {
    Header h;h.kind=kind;h.identity=identity;h.bytes=payload.size();h.sha=digest(payload);
    Bytes result;append(result,h);result.insert(result.end(),payload.begin(),payload.end());return result;
}
inline std::span<const std::byte> open(std::span<const std::byte> bytes,std::array<char,8> kind,std::uint64_t identity) {
    Reader r{bytes};const auto h=r.get<Header>();require(h.kind==kind && h.abi==1 && !h.reserved && h.identity==identity && h.bytes==r.remaining.size(),"Checkpoint schema or contract mismatch");
    require(h.sha==digest(r.remaining),"Checkpoint checksum mismatch");return r.remaining;
}
inline void writeExclusive(const std::filesystem::path& path,std::span<const std::byte> bytes) {
    const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);require(fd>=0,"Cannot exclusively create checkpoint");
    bool okay=true;std::size_t offset=0;
    while (offset<bytes.size()) {const auto count=::write(fd,bytes.data()+offset,bytes.size()-offset);if (count<=0) {okay=false;break;}offset+=count;}
    if (okay) okay=::fsync(fd)==0;const auto closed=::close(fd);require(okay && closed==0,"Checkpoint write or fsync failed");
    const auto parent=path.parent_path().empty()?std::filesystem::path("."):path.parent_path();
    const int directory=::open(parent.c_str(),O_RDONLY|O_DIRECTORY);require(directory>=0,"Checkpoint directory open failed");
    const bool directorySynced=::fsync(directory)==0;const int directoryClosed=::close(directory);
    require(directorySynced && directoryClosed==0,"Checkpoint directory fsync failed");
}
inline Bytes readBounded(const std::filesystem::path& path,std::uint64_t maximumBytes) {
    const auto count=std::filesystem::file_size(path);require(count<=maximumBytes,"Checkpoint exceeds admitted size");
    Bytes bytes(static_cast<std::size_t>(count));std::ifstream input(path,std::ios::binary);
    require(bool(input.read(reinterpret_cast<char*>(bytes.data()),bytes.size())) && input.peek()==std::char_traits<char>::eof(),"Checkpoint read failed or length changed");return bytes;
}
}
