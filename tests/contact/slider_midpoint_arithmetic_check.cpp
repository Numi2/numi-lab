#include "metalrobo/slider_midpoint_state_gpu.h"
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
static float f(std::uint32_t x) { float y; std::memcpy(&y,&x,sizeof(y)); return y; }
static std::uint32_t b(float x) { std::uint32_t y; std::memcpy(&y,&x,sizeof(y)); return y; }
int main() {
  std::uint32_t qh,ql,vh,vl,dh,dl,h;
  while (std::cin >> std::hex >> qh >> ql >> vh >> vl >> dh >> dl >> h) {
    MRSliderMidpointState s{{f(qh),f(ql)},{f(vh),f(vl)}};
    auto m=mrSliderMidpointKinematicTarget(s,{f(dh),f(dl)},f(h),false);
    auto e=mrSliderMidpointKinematicTarget(s,{f(dh),f(dl)},f(h),true);
    std::cout << std::hex << b(m.position.high) << " " << b(m.position.low) << " "
              << b(m.velocity.high) << " " << b(m.velocity.low) << " "
              << b(e.position.high) << " " << b(e.position.low) << " "
              << b(e.velocity.high) << " " << b(e.velocity.low) << " "
              << mrSliderMidpointTargetCanonical(m) << " " << mrSliderMidpointTargetCanonical(e) << "\n";
  }
}
