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
    std::cout<<"aortic_outer_contact_policy=passed default=wall_enabled opt_out=explicit_zero\n";
    return 0;
}
