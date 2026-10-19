#include "NumiHumanWholeBodyMassReference.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace metalrobo;
using namespace metalrobo::human;
using namespace metalrobo::human::whole_body_mass_reference_detail;

namespace {
struct Parsed {
    std::vector<MRBodyPropertiesGPU> bodies;
    std::vector<NumiHumanTissueMassNode> nodes;
};

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

Parsed readDump(const char* path) {
    std::ifstream in(path);
    require(static_cast<bool>(in), std::string("cannot open input dump: ") + path);
    Parsed out;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream row(line);
        std::string tag;
        row >> tag;
        if (tag == "body") {
            std::uint32_t index = 0;
            int motion = 0;
            double mass = 0.0;
            row >> index >> motion >> mass;
            M bodyInertia{};
            for (double& value : bodyInertia) row >> value;
            require(static_cast<bool>(row), "malformed body row");
            if (out.bodies.size() <= index) out.bodies.resize(std::size_t(index) + 1);
            auto& body = out.bodies[index];
            body = {};
            body.motionType = static_cast<std::uint8_t>(motion);
            const float packedMass = static_cast<float>(mass);
            body.massAndInverseMass = {packedMass, packedMass > 0.0f ? 1.0f / packedMass : 0.0f, 0.0f, 0.0f};
            body.inertiaRow0 = {float(bodyInertia[0]), float(bodyInertia[1]), float(bodyInertia[2]), 0.0f};
            body.inertiaRow1 = {float(bodyInertia[3]), float(bodyInertia[4]), float(bodyInertia[5]), 0.0f};
            body.inertiaRow2 = {float(bodyInertia[6]), float(bodyInertia[7]), float(bodyInertia[8]), 0.0f};
        } else if (tag == "node") {
            NumiHumanTissueMassNode node{};
            row >> node.nodeIndex >> node.donorBody >> node.massKg >>
                node.localPosition[0] >> node.localPosition[1] >> node.localPosition[2];
            require(static_cast<bool>(row), "malformed node row");
            out.nodes.push_back(node);
        } else {
            throw std::runtime_error("unknown input dump row");
        }
    }
    require(in.eof(), "failed while reading input dump");
    return out;
}

void composeExactReference(Parsed& parsed, double& originalMass, double& tissueMass) {
    originalMass = 0.0;
    tissueMass = 0.0;
    for (const auto& body : parsed.bodies)
        if (body.motionType == MR_MOTION_DYNAMIC)
            originalMass += double(body.massAndInverseMass.x);

    std::map<std::uint32_t, NumiHumanTissueMassPartition> donors;
    for (const auto& node : parsed.nodes) {
        require(node.donorBody < parsed.bodies.size(), "node donor out of range");
        auto& part = donors[node.donorBody];
        part.donorBody = node.donorBody;
        ++part.nodeCount;
        part.tissueMassKg += node.massKg;
        tissueMass += node.massKg;
        for (std::size_t i = 0; i < 3; ++i) {
            part.tissueFirstMomentKgM[i] += node.massKg * node.localPosition[i];
            for (std::size_t j = 0; j < 3; ++j)
                part.tissueSecondMomentKgM2[3 * i + j] +=
                    node.massKg * node.localPosition[i] * node.localPosition[j];
        }
    }
    require(originalMass > 0.0 && tissueMass < originalMass, "invalid global mass totals");
    const double scale = (originalMass - tissueMass) / originalMass;
    std::vector<V> offsets(parsed.bodies.size());
    for (std::size_t i = 0; i < parsed.bodies.size(); ++i) {
        auto& body = parsed.bodies[i];
        if (body.motionType != MR_MOTION_DYNAMIC) continue;
        auto it = donors.find(static_cast<std::uint32_t>(i));
        const NumiHumanTissueMassPartition empty{};
        const auto& part = it == donors.end() ? empty : it->second;
        M sourceSecond{};
        double floorCorrection = 0.0;
        std::string error;
        require(floorSource(second(inertia(body)), sourceSecond, floorCorrection, error),
                "source moment floor failed for body " + std::to_string(i) + ": " + error);
        const double residual = scale * double(body.massAndInverseMass.x);
        const double composed = residual + part.tissueMassKg;
        V c{};
        if (part.tissueMassKg > 0.0)
            for (std::size_t k = 0; k < 3; ++k)
                c[k] = part.tissueFirstMomentKgM[k] / composed;
        M composedSecond = add(mul(sourceSecond, scale), part.tissueSecondMomentKgM2);
        composedSecond = sub(composedSecond, mul(outer(c, c), composed));
        require(pd(composedSecond), "inferred composed second moment invalid for body " + std::to_string(i));
        body.centerOfMass.x = float(double(body.centerOfMass.x) + c[0]);
        body.centerOfMass.y = float(double(body.centerOfMass.y) + c[1]);
        body.centerOfMass.z = float(double(body.centerOfMass.z) + c[2]);
        require(packBody(body, composed, composedSecond, error),
                "inferred reference pack failed for body " + std::to_string(i) + ": " + error);
        offsets[i] = c;
    }
    for (auto& node : parsed.nodes)
        for (std::size_t k = 0; k < 3; ++k)
            node.localPosition[k] -= offsets[node.donorBody][k];
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: whole_body_stock_dump_replay <native tissue-mass-input.txt>");
        auto parsed = readDump(argv[1]);
        require(parsed.bodies.size() == 157, "unexpected body count in retained native dump");
        require(parsed.nodes.size() == 13100, "unexpected node count in retained native dump");
        double originalMass = 0.0, tissueMass = 0.0;
        composeExactReference(parsed, originalMass, tissueMass);

        const auto stock = compileNumiHumanTissueMassPartition(parsed.bodies, parsed.nodes);
        require(stock.succeeded(), "stock partition failed: " + stock.error);
        require(stock.partitions.size() == 33, "stock partition did not cover all 33 cooked donors");
        for (const auto& partition : stock.partitions) {
            require(partition.packedMomentRelativeError <=
                        16.0 * std::numeric_limits<float>::epsilon(),
                    "stock packed moment closure exceeded the existing Float32 bound");
        }
        std::cout << "native_dump_stock_replay=PASS bodies=" << parsed.bodies.size()
                  << " nodes=" << parsed.nodes.size()
                  << " donors=" << stock.partitions.size()
                  << " original_dynamic_mass_kg=" << originalMass
                  << " cooked_tissue_mass_kg=" << tissueMass
                  << " residual_mass_kg=" << originalMass - tissueMass
                  << " stock_symmetric_accumulation=PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
