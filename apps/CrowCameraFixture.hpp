#pragma once
#include "metalrobo/CrowForaging.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
namespace crowCameraFixture {
using namespace metalrobo;
constexpr unsigned N=2,W=crowForagingCameraWidth,H=crowForagingCameraHeight,T=128;
constexpr double nearDepth=.01,farDepth=5;
inline void require(bool value,const std::string& reason) {if(!value)throw std::runtime_error(reason);}
template<class R> void check(const R& r,const char* reason) {require(r.succeeded(),std::string(reason)+": "+r.message);}
using Visual=CrowForagingVisual;
inline RunManifest makeManifest(bool blocked) {return makeAmericanCrowForagingFixtureManifest(blocked);}
inline Visual makeVisual(const CompiledRun& run,const std::filesystem::path& output,bool foodPortIndicators=false) {
    return authorAmericanCrowForagingVisual(run,output,foodPortIndicators);
}
}
