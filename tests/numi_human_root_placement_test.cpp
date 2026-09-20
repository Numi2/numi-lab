#include "metalrobo/NumiHumanInitialState.hpp"
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace {
std::size_t checks = 0;
void require(bool ok, const char* label="root placement regression") {
    ++checks;
    if (!ok) throw std::runtime_error(label);
}
}
int main() {
    try {
        MRCompensatedRootTranslationGPU placed{};
        std::string placementError;
        const std::array<std::array<double, 3u>, 6u> placements{{
            {-0.0, 0.0, -0.0},
            {-0.05000000074505806, 0.20000000298023224, 1.94682457843},
            {1.0 + 0x1p-26, -1.0 - 0x1p-26, 100000001.0},
            {1.0 + 0x1p-24 + 0x1p-52, 1.0 + 0x1p-24 - 0x1p-52, -1.0 - 0x1p-24},
            {0.001234567890123, -12.34567890123, 9876.543210123},
            {double(std::numeric_limits<float>::denorm_min()), 1e-20, -1e-20}
        }};
        for (const auto& position : placements) {
            require(metalrobo::makeNumiHumanInitialRootTranslation(position, placed, placementError),
                    "offline root placement must round-trip through its authoritative expansion");
            require(placementError.empty() && mrCompensatedTranslationValid(placed));
            const auto high = mrCompensatedTranslationProjection(placed);
            const std::array<float,3u> highValues{high.x,high.y,high.z};
            const std::array<float,3u> r{placed.reference.x,placed.reference.y,placed.reference.z};
            const std::array<float,3u> d{placed.displacement.x,placed.displacement.y,placed.displacement.z};
            const std::array<float,3u> c{placed.correction.x,placed.correction.y,placed.correction.z};
            for (std::size_t axis=0;axis<3u;++axis) {
                require((double(r[axis])+d[axis])+c[axis]==position[axis], "FP64 placement lost low words");
                require(std::bit_cast<std::uint32_t>(highValues[axis])==
                        std::bit_cast<std::uint32_t>(float(position[axis])), "q projection changed");
            }
        }
        const auto acceptedPlacement=placed;
        for (const double bad : {std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::max(), 1e-100}) {
            require(!metalrobo::makeNumiHumanInitialRootTranslation({0.0,bad,1.0},placed,placementError));
            require(!placementError.empty() && std::memcmp(&placed,&acceptedPlacement,sizeof(placed))==0,
                    "invalid placement mutated accepted root state");
        }
        std::cout << "Human root placement: " << checks << " checks passed; no GPU execution\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
