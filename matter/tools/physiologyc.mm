#include "numi/matter/human_physiology.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, const char* argv[]) {
    try {
        if(argc!=7 || std::string(argv[3])!="--timestep" || std::string(argv[5])!="--environments")
            throw std::runtime_error("usage: numi-matter-physiologyc INPUT.json OUTPUT.nmatterpack --timestep SECONDS --environments COUNT");
        std::size_t used=0;
        double dt=std::stod(argv[4],&used);
        if(used!=std::string(argv[4]).size() || !std::isfinite(dt) || dt<=0) throw std::runtime_error("invalid timestep");
        auto environments=std::stoul(argv[6],&used);
        if(used!=std::string(argv[6]).size() || environments==0 || environments>=NM_INVALID_INDEX) throw std::runtime_error("invalid environment count");
        numi::matter::WorldSource source;
        source.frameTimestep=dt; source.environmentCount=static_cast<std::uint32_t>(environments);
        source.gravity={0,0,0};
        source.mixedSolver.relativeResidual=1e-7;
        source.mixedSolver.newtonIterations=12;
        source.mixedSolver.fgmresIterations=64;
        std::string error;
        if(!numi::matter::readHumanPhysiologyNetwork(argv[1],source.vascular,&error)) throw std::runtime_error(error);
        auto compiled=numi::matter::compileWorld(source,{.maximumRateExponent=0});
        if(!compiled.succeeded()) {
            for(const auto& d:compiled.diagnostics) std::cerr<<d.message<<'\n';
            throw std::runtime_error("native physiology compilation rejected");
        }
        if(!numi::matter::writePackage(compiled,argv[2],&error))throw std::runtime_error(error);
        numi::matter::CompiledWorld readback;
        if(!numi::matter::readPackage(argv[2],readback,nullptr,&error) || readback.fingerprint!=compiled.world.fingerprint)throw std::runtime_error("package readback failed: "+error);
        std::cout<<"physiology_compile=pass compartments="<<readback.vascular.compartments.size()
                 <<" species="<<readback.vascular.species.size()<<" tissue_reservoirs="<<readback.vascular.tissues.size()
                 <<" native_fingerprint=0x"<<std::hex<<readback.fingerprint<<std::dec
                 <<" biological_qualification=unqualified\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"physiology_compile=failed reason="<<e.what()<<'\n';return 1;}
}
