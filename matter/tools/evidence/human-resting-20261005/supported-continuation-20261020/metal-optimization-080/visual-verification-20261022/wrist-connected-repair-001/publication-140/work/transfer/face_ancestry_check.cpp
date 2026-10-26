#include "NumiHumanCoarseTissuePreparation.hpp"
#include <iostream>
using namespace metalrobo::human::coarse_tissue;
int main() {
    RegisteredSurface s;
    s.faces.resize(2);
    s.sourceFaceAncestryParentCount=4;
    s.sourceFaceAncestryOffsets={0,2,3};
    s.sourceFaceAncestryRows={1,3,2};
    std::string error;
    if(!validateRegisteredFaceAncestry(s,error)) { std::cerr<<error<<"\n"; return 1; }
    s.sourceFaceAncestryRows[1]=1;
    if(validateRegisteredFaceAncestry(s,error)||error.find("unsorted")==std::string::npos) return 2;
    s.sourceFaceAncestryRows={1,4,2}; s.sourceFaceAncestryParentCount=4;
    if(validateRegisteredFaceAncestry(s,error)||error.find("invalid")==std::string::npos) return 3;
    s.sourceFaceAncestryRows={1,3,2}; s.sourceFaceAncestryParentCount=0;
    if(validateRegisteredFaceAncestry(s,error)||error.find("incomplete")==std::string::npos) return 4;
    std::cout<<"face ancestry CSR sorted/range/parent-count checks: PASS\n";
    return 0;
}
