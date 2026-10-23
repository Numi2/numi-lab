#pragma once
#include "NumiHumanAcceptedFEMSkinTransfer.hpp"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace numi::human {
// Generated from D16's immutable physical midpoint ancestry. No geometry,
// muscle binding, donor mass, contact or force data are changed.
inline std::vector<AcceptedFEMSkinTransfer::BaselineProlongationRow>
makeElbowPhysicalBaselineProlongation065(std::size_t count) {
    if(count!=2411u)throw std::runtime_error("elbow baseline ancestry row count mismatch");
    std::vector<AcceptedFEMSkinTransfer::BaselineProlongationRow> result(count);
    for(std::uint32_t i=0;i<count;++i)result[i]={{i,i,i},{1.0,0.0,0.0}};
    constexpr std::array<std::array<std::uint32_t,3>,79> edges{{
        {{2332u,358u,374u}},
        {{2333u,358u,375u}},
        {{2334u,374u,375u}},
        {{2335u,374u,660u}},
        {{2336u,466u,664u}},
        {{2337u,576u,630u}},
        {{2338u,576u,677u}},
        {{2339u,630u,677u}},
        {{2340u,648u,669u}},
        {{2341u,648u,684u}},
        {{2342u,661u,668u}},
        {{2343u,661u,682u}},
        {{2344u,662u,669u}},
        {{2345u,662u,677u}},
        {{2346u,662u,684u}},
        {{2347u,664u,675u}},
        {{2348u,665u,675u}},
        {{2349u,665u,2320u}},
        {{2350u,668u,682u}},
        {{2351u,669u,684u}},
        {{2352u,673u,707u}},
        {{2353u,677u,681u}},
        {{2354u,677u,684u}},
        {{2355u,677u,688u}},
        {{2356u,681u,688u}},
        {{2357u,684u,688u}},
        {{2358u,707u,773u}},
        {{2359u,707u,2322u}},
        {{2360u,734u,746u}},
        {{2361u,734u,748u}},
        {{2362u,734u,759u}},
        {{2363u,734u,765u}},
        {{2364u,734u,781u}},
        {{2365u,734u,2320u}},
        {{2366u,746u,765u}},
        {{2367u,746u,2320u}},
        {{2368u,748u,765u}},
        {{2369u,748u,781u}},
        {{2370u,759u,781u}},
        {{2371u,759u,782u}},
        {{2372u,759u,2321u}},
        {{2373u,773u,782u}},
        {{2374u,773u,820u}},
        {{2375u,773u,830u}},
        {{2376u,773u,2321u}},
        {{2377u,773u,2322u}},
        {{2378u,782u,820u}},
        {{2379u,782u,2321u}},
        {{2380u,820u,830u}},
        {{2381u,830u,2322u}},
        {{2382u,1813u,1836u}},
        {{2383u,1813u,2323u}},
        {{2384u,1813u,2324u}},
        {{2385u,1836u,2324u}},
        {{2386u,1836u,2325u}},
        {{2387u,1836u,2329u}},
        {{2388u,1988u,2011u}},
        {{2389u,1988u,2012u}},
        {{2390u,2007u,2012u}},
        {{2391u,2007u,2021u}},
        {{2392u,2007u,2022u}},
        {{2393u,2007u,2033u}},
        {{2394u,2007u,2040u}},
        {{2395u,2007u,2079u}},
        {{2396u,2011u,2012u}},
        {{2397u,2011u,2029u}},
        {{2398u,2011u,2142u}},
        {{2399u,2012u,2021u}},
        {{2400u,2012u,2142u}},
        {{2401u,2021u,2022u}},
        {{2402u,2021u,2142u}},
        {{2403u,2022u,2079u}},
        {{2404u,2029u,2142u}},
        {{2405u,2033u,2040u}},
        {{2406u,2040u,2079u}},
        {{2407u,2323u,2324u}},
        {{2408u,2324u,2325u}},
        {{2409u,2325u,2329u}},
        {{2410u,2329u,2331u}},
    }};
    for(const auto& row:edges)result[row[0]]={{row[1],row[2],row[1]},{0.5,0.5,0.0}};
    return result;
}
} // namespace numi::human
