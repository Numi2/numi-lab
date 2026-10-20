#include "NumiHumanRestingAorticWall.hpp"

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace {
void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}

bool rejects(bool wallEnabled,std::optional<std::string_view> setting) {
    try {
        (void)numi::human::aortic_wall::resolveRestingOuterLateralContact(
            wallEnabled,setting);
    } catch(const std::runtime_error&) {
        return true;
    }
    return false;
}
} // namespace

int main() {
    using numi::human::aortic_wall::resolveRestingOuterLateralContact;
    require(!resolveRestingOuterLateralContact(false,std::nullopt),
            "disabled aortic wall unexpectedly enabled outer contact");
    require(resolveRestingOuterLateralContact(true,std::nullopt),
            "resting aortic wall did not default to its authored outer contact");
    require(resolveRestingOuterLateralContact(true,std::string_view("1")),
            "explicit outer-contact enable was ignored");
    require(!resolveRestingOuterLateralContact(true,std::string_view("0")),
            "explicit contact-free diagnostic override was ignored");
    require(!resolveRestingOuterLateralContact(false,std::string_view("0")),
            "disabled wall with contact opt-out returned contact-enabled");
    require(rejects(false,std::string_view("1")),
            "outer contact was allowed without the registered aortic wall");
    require(rejects(true,std::string_view("true"))&&rejects(true,std::string_view("2")),
            "outer-contact parser accepted a nonbinary environment value");
    const auto preload=numi::human::aortic_wall::solveLoadedReference(
        0.009,0.0102,11400.0,0.0,130000.0,1300000.0,1.0,512u);
    require(preload.axialStretch>0.8&&preload.axialStretch<1.2,
            "closed-end aortic preload escaped its declared stretch sensitivity range");
    require(std::abs(preload.innerBoundaryResidualPa)<1.0e-5&&
            std::abs(preload.outerBoundaryResidualPa)<1.0e-5,
            "aortic preload failed radial pressure traction balance");
    require(std::abs(preload.axialEquilibriumResidualN)<
                preload.axialPressureResultantN*1.1e-6,
            "aortic preload failed closed-end pressure resultant balance");
    require(preload.axialPressureResultantN>0.0&&
                preload.axialWallResultantN>0.0,
            "aortic preload removed its pressure-loaded wall force");
    std::cout<<"aortic_outer_contact_policy=passed closed_end_loaded_preload=passed"
             <<" axial_stretch="<<preload.axialStretch
             <<" axial_pressure_N="<<preload.axialPressureResultantN
             <<" axial_wall_N="<<preload.axialWallResultantN
             <<" axial_residual_N="<<preload.axialEquilibriumResidualN<<"\n";
    return 0;
}
