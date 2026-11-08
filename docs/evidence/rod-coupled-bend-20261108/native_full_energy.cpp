#include "metalrobo/DiscreteElasticRod.hpp"
#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>
using namespace metalrobo;
DiscreteElasticRodEnergy energy(const DiscreteElasticRodModel& m, const DiscreteElasticRodState& s) {
    DiscreteElasticRodEnergy out;
    if (evaluateDiscreteElasticRodEnergy(m, s, out).status != DiscreteElasticRodStatus::success) {
        std::cerr << "native energy evaluation failed\n"; std::exit(2);
    }
    return out;
}
int main() {
    std::cout << std::setprecision(17);
    int n;
    while (std::cin >> n) {
        if (n < 3 || n > 9) return 3;
        constexpr double rest=0.11758755785232791, ea=2.3818718058345718, radius=0.000020;
        DiscreteRodMaterial mat; mat.radius=radius; mat.density=1100;
        mat.youngModulus=ea/(std::numbers::pi*radius*radius);
        auto model=makeStraightSutureRod(n,rest,mat);
        auto state=makeDiscreteElasticRodDefaultState(model);
        for (auto& x:state.positions) for (double& v:x) std::cin >> v;
        for (double& v:state.twists) std::cin >> v;
        if (!std::cin) return 4;
        auto base=energy(model,state);
        std::cout << "{\"n\":" << n << ",\"bend\":" << base.bend
                  << ",\"twist\":" << base.twist << ",\"stretch\":" << base.stretch
                  << ",\"EI\":" << model.bendStiffness[0] << ",\"GJ\":" << model.twistStiffness[0]
                  << ",\"restLength\":" << model.restLengths[0];
        for (int quality=0;quality<2;quality++) {
            double hx=quality==0?2e-7:1e-7, ht=quality==0?2e-5:1e-5;
            std::cout << ",\"bendGradient" << quality << "\":[";
            bool first=true;
            for (int j=0;j<3*n+n-1;j++) {
                auto a=state,b=state; double h=j<3*n?hx:ht;
                if (j<3*n) {a.positions[j/3][j%3]+=h;b.positions[j/3][j%3]-=h;}
                else {a.twists[j-3*n]+=h;b.twists[j-3*n]-=h;}
                auto p=energy(model,a), q=energy(model,b);
                if (!first) std::cout << ","; first=false;
                std::cout << (p.bend-q.bend)/(2*h);
            }
            std::cout << "],\"stretchGradient" << quality << "\":[";
            first=true;
            for (int j=0;j<3*n;j++) {
                auto a=state,b=state;
                a.positions[j/3][j%3]+=hx; b.positions[j/3][j%3]-=hx;
                auto plus=energy(model,a),minus=energy(model,b);
                if (!first) std::cout << ","; first=false;
                std::cout << (plus.stretch-minus.stretch)/(2*hx);
            }
            std::cout << "],\"twistGradient" << quality << "\":[";
            first=true;
            for (int j=0;j<n-1;j++) {
                auto a=state,b=state;a.twists[j]+=ht;b.twists[j]-=ht;
                auto p=energy(model,a),q=energy(model,b);
                if (!first) std::cout << ","; first=false;
                std::cout << (p.twist-q.twist)/(2*ht);
            }
            std::cout << "]";
        }
        std::cout << "}\n";
    }
}
