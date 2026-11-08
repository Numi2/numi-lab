#include "metalrobo/CrowManipulation.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace metalrobo;
int main(){try{
    CompiledRun run;const auto result=compileRun(makeAmericanCrowObjectPushFixtureManifest(),run);
    if(!result.succeeded())throw std::runtime_error(result.message);
    if(run.world().fingerprint()!=6600400359852534321ull)throw std::runtime_error("Fixture physical identity changed");
    const auto pairs=run.world().eligiblePairs();
    if(pairs.size()<=106)throw std::runtime_error("Missing reported pair");
    const auto pair=pairs[106];const auto& model=run.model();
    std::cout<<"{\"world_fingerprint\":"<<run.world().fingerprint()<<",\"simulation_steps\":0,\"pair_index\":106,\"eligible_pair_count\":"<<pairs.size()<<",\"colliders\":[";
    for(unsigned i=0;i<2;++i){const auto index=i?pair.colliderB:pair.colliderA;const auto& shape=model.shapes.at(index);if(i)std::cout<<',';
      std::cout<<"{\"index\":"<<index<<",\"name\":"<<std::quoted(model.shapeNames.at(index))
       <<",\"body\":"<<std::quoted(model.bodyNames.at(shape.bodyIndex))<<",\"shape_type\":"<<shape.shapeType
       <<",\"flags\":"<<shape.flags<<",\"contact_rest_m\":"<<shape.contactRestAndBoundingRadius.x<<'}';}
    std::cout<<"]}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
