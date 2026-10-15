#include <numi/matter/shared.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

static_assert(sizeof(NMFEMStatus5CaptureGPU) == 3392);
static_assert(offsetof(NMFEMStatus5CaptureGPU, status) == 80);
static_assert(offsetof(NMFEMStatus5CaptureGPU, microstep) == 128);

static const char* stageName(std::uint32_t stage) {
    switch (stage) {
        case 10: return "kinematic_target";
        case 20: return "physical_internal_force";
        case 30: return "residual_armijo_private_geometry";
        case 31: return "residual_armijo_private_internal_force";
        case 40: return "capsule_box_private_geometry";
        case 41: return "capsule_box_private_internal_force";
        case 50: return "later_physical_internal_force";
        case 60: return "global_fem_validation";
        default: return "unknown";
    }
}

static const char* indexKind(std::uint32_t kind) {
    switch (kind) {
        case NM_STATUS5_INDEX_NONE: return "none";
        case NM_STATUS5_INDEX_NODE: return "node";
        case NM_STATUS5_INDEX_TETRAHEDRON: return "tetrahedron";
        case NM_STATUS5_INDEX_SOLVER_ITERATION: return "solver_iteration";
        default: return "unknown";
    }
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream stream(argv[1], std::ios::binary);
    if (!stream) return 3;
    std::vector<char> bytes((std::istreambuf_iterator<char>(stream)), {});
    if (bytes.size() != 4u * sizeof(NMFEMStatus5CaptureGPU)) return 4;
    const auto* records = reinterpret_cast<const NMFEMStatus5CaptureGPU*>(bytes.data());
    std::uint32_t present = 0;
    for (std::uint32_t i = 0; i < 4; ++i) present += records[i].capture.x != 0;
    std::cout << std::setprecision(10)
              << "{\"record_bytes\":" << sizeof(NMFEMStatus5CaptureGPU)
              << ",\"record_count\":4,\"records_present\":" << present
              << ",\"records\":[";
    for (std::uint32_t i = 0; i < 4; ++i) {
        if (i) std::cout << ',';
        const auto& r = records[i];
        std::cout << "{\"slot\":" << i
                  << ",\"capture\":[" << r.capture.x << ',' << r.capture.y << ','
                  << r.capture.z << ',' << r.capture.w << ']'
                  << ",\"stage_name\":\"" << stageName(r.capture.y) << '\"'
                  << ",\"domain_name\":\"" << (r.capture.z == 0 ? "global" : "private_armijo") << '\"'
                  << ",\"index_kind_name\":\"" << indexKind(r.capture.w) << '\"'
                  << ",\"context\":[" << r.context.x << ',' << r.context.y << ','
                  << r.context.z << ',' << r.context.w << ']'
                  << ",\"identity\":[" << r.identity.x << ',' << r.identity.y << ','
                  << r.identity.z << ',' << r.identity.w << ']'
                  << ",\"payload\":[" << r.payload.x << ',' << r.payload.y << ','
                  << r.payload.z << ',' << r.payload.w << ']'
                  << ",\"scalars\":[" << r.scalars.x << ',' << r.scalars.y << ','
                  << r.scalars.z << ',' << r.scalars.w << ']'
                  << ",\"status\":[" << r.status.code << ',' << r.status.environment << ','
                  << r.status.objectIndex << ',' << r.status.failingIndex << ','
                  << r.status.completedMicrosteps << ',' << r.status.fgmresIterations << ','
                  << r.status.contactCount << ',' << r.status.eventCount << ','
                  << r.status.diagnostics.x << ',' << r.status.diagnostics.y << ','
                  << r.status.diagnostics.z << ',' << r.status.diagnostics.w << ']'
                  << ",\"microstep\":[" << r.microstep.controlStep << ',' << r.microstep.microtick << ','
                  << r.microstep.microtickCount << ',' << r.microstep.solverIteration << ','
                  << r.microstep.seedLo << ',' << r.microstep.seedHi << ',' << r.microstep.flags << ','
                  << r.microstep.reserved << ',' << r.microstep.time.x << ',' << r.microstep.time.y << ','
                  << r.microstep.time.z << ',' << r.microstep.time.w << ']'
                  << ",\"material_words\":[" << r.material.constitutiveKind << ',' << r.material.parameterOffset
                  << ',' << r.material.parameterCount << ',' << r.material.stateCount << ']'
                  << ",\"material_validity\":[" << r.material.validity.x << ','
                  << r.material.validity.y << ',' << r.material.validity.z << ','
                  << r.material.validity.w << ']'
                  << ",\"tetrahedron\":[" << r.tetrahedron.nodes.x << ',' << r.tetrahedron.nodes.y << ','
                  << r.tetrahedron.nodes.z << ',' << r.tetrahedron.nodes.w << ','
                  << r.tetrahedron.identity.x << ',' << r.tetrahedron.identity.y << ','
                  << r.tetrahedron.identity.z << ',' << r.tetrahedron.identity.w << ']'
                  << '}';
    }
    std::cout << "]}\n";
}
