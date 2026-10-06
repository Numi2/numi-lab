#include "metalrobo/numi_human_resting_common_field_math.hpp"
#include "metalrobo/numi_human_resting_common_attachment.hpp"
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
    using PassiveRange=PassiveAttachmentRangeContract;
    std::array<PassiveRange,3> passive{{
        {6,51011,20,2,100,3},
        {11,51011,20,2,103,2,true,true,true,7,7,20,1,-0.10f,-0.055f},
        {125,51021,20,5,105,4}
    }};
    check(validPassiveAttachmentRanges(passive,100,109),
        "valid typed passive ranges or IVC anchor fixture was rejected");
    check(!validPassiveAttachmentRanges(passive,101,109),
        "passive range first-offset mismatch was accepted");
    check(!validPassiveAttachmentRanges(passive,100,108),
        "passive map trailing/missing row was accepted");
    const std::array<PassiveRange,0> legacyNoAttachments{};
    check(validPassiveAttachmentRanges(legacyNoAttachments,100,100),
        "legacy common map without optional passive attachments was rejected");
    check(!validPassiveAttachmentRanges(legacyNoAttachments,100,101),
        "unowned trailing rows were accepted without passive ranges");
    check(quinticSmoothstep(-.10f,-.055f,-.11f)==0.0f&&
        quinticSmoothstep(-.10f,-.055f,-.10f)==0.0f&&
        quinticSmoothstep(-.10f,-.055f,-.0775f)==.5f&&
        quinticSmoothstep(-.10f,-.055f,-.055f)==1.0f&&
        quinticSmoothstep(-.10f,-.055f,-.04f)==1.0f,
        "IVC quintic transition endpoints or midpoint are invalid");
    auto bad=passive;bad[1].semantic=51021;
    check(!validPassiveAttachmentRanges(bad,100,109),
        "IVC with a pulmonary-artery semantic was accepted");
    bad=passive;bad[1].anchorBodyIndex=20;
    check(!validPassiveAttachmentRanges(bad,100,109),
        "IVC with a wrong anchor body was accepted");
    bad=passive;bad[2].hasAttachment=true;
    check(!validPassiveAttachmentRanges(bad,100,109),
        "non-IVC passive range with an attachment blend was accepted");
    bad=passive;bad[2].stableId=6;
    check(!validPassiveAttachmentRanges(bad,100,109),
        "duplicate passive source IDs were accepted");
    bad=passive;bad[0].bodyIndex=7;
    check(!validPassiveAttachmentRanges(bad,100,109),
        "passive map range outside torso registration was accepted");
    std::cout<<"common_field_cubic_terms=120 coordinate_order=lexicographic layout=pass attachments=pass\n";
    return 0;
}
