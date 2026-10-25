#include "numi/matter/matter.hpp"
#include <iostream>
#include <string>
using namespace numi::matter;
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    CompiledWorld world;
    std::string error;
    if(readPackage(argv[1],world,nullptr,&error)) {
        std::cerr<<"old v19 reader unexpectedly accepted version-20 metadata\n";
        return 3;
    }
    std::cout<<error<<'\n';
    return error.find("ABI mismatch: recook required")!=std::string::npos?0:4;
}
