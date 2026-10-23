#include "NumiHumanD36Ring0RemeshAdapter.hpp"
using namespace metalrobo::human::coarse_tissue;
bool compileAdapter(numi::matter::ObjectSource& o,PreparedOrganicFEM& s,PreparedOrganicFEM& f,RegisteredSkinFatReductionSourceLabels& l,std::span<const std::uint32_t> p,std::string& e) {
 LocalSkinFatTetSpliceReport r;
 return applyD36Ring0Remesh(o,s,f,l,p,r,e,[](const void*,std::size_t){return std::array<unsigned char,32>{};});
}
int main(){return 0;}
