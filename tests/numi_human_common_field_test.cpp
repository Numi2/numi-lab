#include "metalrobo/numi_human_resting_common_field_math.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
}
int main() {
    using namespace numi_human_resting_common_field;
    check(kMonomialExponents.front()==Exponent{0,0,0,0,0,0,0},"constant term is not first");
    check(kMonomialExponents[1]==Exponent{0,0,0,0,0,0,1},"last coordinate lexicographic term mismatch");
    check(kMonomialExponents.back()==Exponent{3,0,0,0,0,0,0},"first coordinate cubic term is not last");
    check(termIndex(Exponent{0,0,0,0,0,0,0})==0,"constant index mismatch");
    check(termIndex(Exponent{1,1,0,0,0,0,0})<120,"cross term missing");
    check(termIndex(Exponent{1,1,1,0,0,0,0})<120,"cubic cross term missing");
    check(termIndex(Exponent{1,1,1,1,0,0,0})==120,"degree-four term must be rejected");
    check(sizeof(MRHumanRestingCommonFieldVertexGPU)==112,"common map ABI size mismatch");
    check(sizeof(MRHumanRestingCommonCoordinateBoxGPU)==64,"coordinate-box ABI size mismatch");
    check(sizeof(MRHumanRestingCommonFieldGPU)==3504,"common params ABI size mismatch");
    check(sizeof(MRHumanRestingCommonCoordinatesGPU)==64,"common coordinate result ABI size mismatch");
    std::cout<<"common_field_cubic_terms=120 coordinate_order=lexicographic layout=pass\n";
    return 0;
}
