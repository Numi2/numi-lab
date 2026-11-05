#include "cardboard_expression_cache_fixture.hpp"
#include HEADER_PATH
#include <filesystem>
#include <iostream>
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    std::filesystem::create_directories(argv[1]);
    for(unsigned mode=0;mode<2;++mode){
        const auto world=numi_cardboard_expression_cache::makeWorld(
            numi_cardboard_expression_cache::makeMaterial(true,mode!=0));
        const auto result=numi::matter::compileWorld(world);
        if(!result.succeeded()){std::cerr<<"fixture compile failed\n";return 1;}
        std::string error;
        const auto path=std::filesystem::path(argv[1])/("generated-"+std::to_string(mode)+".metalinc");
        if(!nmWriteNativeTangentSource(result.world,path.c_str(),error)){
            std::cerr<<error<<"\n";return 1;
        }
    }
    std::cout<<"PASS native scalar generator cached and invalid-inactive-branch fixtures\n";
}
