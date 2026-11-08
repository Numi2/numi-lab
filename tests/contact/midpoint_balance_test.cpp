#include "midpoint_balance.hpp"
#include <cmath>
#include <limits>
int main() {
    MidpointBalance x;
    x.add({1., 2., 3.}, {0., 2., 3.});
    x.add({-1., -2., -3.}, {0., -2., -3.});
    if (x.maxLocal != 1. || x.signedSum[0] != 0. ||
        x.signedSum[1] != 0. || x.signedSum[2] != 0.) return 1;
    bool rejected = false;
    try { x.add({std::numeric_limits<double>::infinity(), 0., 0.}, {0., 0., 0.}); }
    catch (const std::runtime_error&) { rejected = true; }
    return rejected ? 0 : 1;
}
