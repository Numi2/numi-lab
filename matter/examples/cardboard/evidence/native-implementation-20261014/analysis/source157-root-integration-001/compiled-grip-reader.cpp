
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cmath>
struct nm_float4 { float x=0, y=0, z=0, w=0; };
struct MeshSource { std::vector<nm_float4> restNodes; std::vector<std::uint32_t> rightGripNodes; };
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
bool commandedRightGripConstrained(const std::vector<nm_float4>& targets,
                                   const MeshSource& mesh,
                                   const std::uint32_t environment) {
    const std::size_t nodeCount = mesh.restNodes.size();
    require(nodeCount > 0u, "right-grip target telemetry has no FEM nodes");
    require(environment <= std::numeric_limits<std::size_t>::max() / nodeCount,
            "right-grip target telemetry environment offset overflows");
    const std::size_t base = static_cast<std::size_t>(environment) * nodeCount;
    require(base <= targets.size() && nodeCount <= targets.size() - base,
            "right-grip target telemetry range is truncated");
    require(!mesh.rightGripNodes.empty(),
            "right-grip target telemetry has no commanded grip nodes");
    const std::uint32_t firstNode = mesh.rightGripNodes.front();
    require(firstNode < nodeCount,
            "right-grip target telemetry node is outside the FEM range");
    const float mode = targets[base + firstNode].w;
    require(mode == 1.0f || mode == 2.0f,
            "right-grip target telemetry has an unsupported constraint mode");
    for (const std::uint32_t node : mesh.rightGripNodes) {
        require(node < nodeCount,
                "right-grip target telemetry node is outside the FEM range");
        require(targets[base + node].w == mode,
                "right-grip target telemetry cannot encode mixed node modes as one boolean");
    }
    return mode == 1.0f;
}
int main() {
    int checks = 0;
    auto check = [&](bool value) { if (!value) throw std::runtime_error("assertion"); ++checks; };
    MeshSource mesh{{{}, {}, {}}, {0, 2}};
    std::vector<nm_float4> targets(6);
    for (int n=0; n<3; ++n) targets[n].w=1;
    for (int n=3; n<6; ++n) targets[n].w=2;
    check(commandedRightGripConstrained(targets, mesh, 0));
    check(!commandedRightGripConstrained(targets, mesh, 1));
    auto rejected = [&](auto work) {
        bool threw = false;
        try { work(); } catch (const std::runtime_error&) { threw = true; }
        check(threw);
    };
    rejected([&] { commandedRightGripConstrained(targets, mesh, 2); });
    targets[2].w=2;
    rejected([&] { commandedRightGripConstrained(targets, mesh, 0); });
    targets[2].w=1;
    targets[0].w=0;
    rejected([&] { commandedRightGripConstrained(targets, mesh, 0); });
    targets[0].w=std::numeric_limits<float>::quiet_NaN();
    rejected([&] { commandedRightGripConstrained(targets, mesh, 0); });
    targets[0].w=1;
    mesh.rightGripNodes={0, 3};
    rejected([&] { commandedRightGripConstrained(targets, mesh, 0); });
    mesh.rightGripNodes={};
    rejected([&] { commandedRightGripConstrained(targets, mesh, 0); });
    mesh.rightGripNodes={0};
    mesh.restNodes.clear();
    rejected([&] { commandedRightGripConstrained(targets, mesh, 0); });
    check(checks == 9);
}
