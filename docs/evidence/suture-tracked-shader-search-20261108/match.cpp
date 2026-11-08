#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <array>
#include <vector>
constexpr std::uint64_t prime=1099511628211ull;
std::uint64_t bytes(const void* v,std::size_t n,std::uint64_t h=1469598103934665603ull) {
 auto *p=static_cast<const unsigned char*>(v);
 for(std::size_t i=0;i<n;++i)h=(h^p[i])*prime;
 return h?h:1;
}
std::uint64_t mix(std::uint64_t h,std::uint64_t v) {
 for(unsigned i=0;i<8;++i)h=(h^((v>>(i*8))&255))*prime;
 return h;
}
int main() {
 std::string path;unsigned count=0,fail=0;
 while(std::getline(std::cin,path)) {
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  if(!f){++fail;continue;}auto size=f.tellg();
  if(size<=0 || size>64*1024*1024){++fail;continue;}
  std::vector<char> blob(static_cast<std::size_t>(size));f.seekg(0);f.read(blob.data(),size);
  if(!f){++fail;continue;}
  std::array<unsigned char,16> zeros{};auto z=bytes(zeros.data(),zeros.size()),e=bytes(nullptr,0);
  std::uint64_t h=14695981039346656037ull;
  for(auto value:std::array<std::uint64_t,12>{10068538913772727962ull,25,bytes(blob.data(),blob.size()),blob.size(),1,1,0,0,e,e,z,z}) h=mix(h,value);
  ++count;
  std::cout<<h<<"\t"<<path<<"\n";
 }
 std::cerr<<"read="<<count<<" failed="<<fail<<"\n";
}
