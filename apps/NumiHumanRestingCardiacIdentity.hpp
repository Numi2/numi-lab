#pragma once
#include <string_view>

// Source-map concepts are canonicalized to chamber walls. Earlier receipts
// used the broader FMA7088 heart concept for the same exact atrial members;
// retain that narrow compatibility while rejecting it for the ventricular wall.
inline constexpr bool numiHumanRestingCardiacConceptAccepted(unsigned stableID,std::string_view conceptID) noexcept {
    switch(stableID) {
        case 1: return conceptID=="FMA9457"||conceptID=="FMA7088";
        case 23: return conceptID=="FMA13884";
        case 24: return conceptID=="FMA9531"||conceptID=="FMA7088";
        default: return false;
    }
}
