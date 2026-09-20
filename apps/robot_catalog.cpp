#include "metalrobo/RunProgram.hpp"

#include <iostream>
#include <string>
#include <string_view>

namespace {
std::string escaped(const std::string_view value) {
    std::string result;
    result.reserve(value.size());
    for (const char character : value) {
        switch (character) {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\n': result += "\\n"; break;
        default: result.push_back(character); break;
        }
    }
    return result;
}

template <typename Values>
void strings(const Values& values) {
    std::cout << '[';
    bool first = true;
    for (const auto& value : values) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << '"' << escaped(value) << '"';
    }
    std::cout << ']';
}

void robot(const metalrobo::RobotPack& pack) {
    std::cout << "{\"id\":\"" << escaped(pack.id)
              << "\",\"revision\":" << pack.revision
              << ",\"source_repository\":\""
              << escaped(pack.sourceRepository)
              << "\",\"source_revision\":\""
              << escaped(pack.sourceRevision)
              << "\",\"license\":\"" << escaped(pack.license)
              << "\",\"model_fingerprint\":"
              << metalrobo::engineModelFingerprint(pack.mechanics)
              << ",\"articulations\":" << pack.mechanics.articulations.size()
              << ",\"bodies\":" << pack.mechanics.bodies.size()
              << ",\"joints\":" << pack.mechanics.joints.size()
              << ",\"dofs\":" << pack.mechanics.dofs.size()
              << ",\"shapes\":" << pack.mechanics.shapes.size()
              << ",\"capabilities\":";
    strings(pack.capabilities);
    std::cout << ",\"roles\":[";
    for (std::size_t index = 0u; index < pack.roles.size(); ++index) {
        if (index != 0u) std::cout << ',';
        const auto& role = pack.roles[index];
        std::cout << "{\"id\":\"" << escaped(role.id)
                  << "\",\"kind\":" << static_cast<std::uint32_t>(role.kind)
                  << ",\"members\":";
        strings(role.members);
        std::cout << '}';
    }
    std::cout << "]}";
}

bool selfCheck() {
    const auto ids = metalrobo::builtinRobotIds();
    if (ids.empty()) {
        std::cerr << "robot catalog is empty\n";
        return false;
    }
    for (std::size_t index = 0u; index < ids.size(); ++index) {
        const auto& id = ids[index];
        if (id.empty()) {
            std::cerr << "robot catalog contains an empty id\n";
            return false;
        }
        for (std::size_t previous = 0u; previous < index; ++previous) {
            if (ids[previous] == id) {
                std::cerr << "robot catalog contains duplicate id: " << id
                          << '\n';
                return false;
            }
        }
        const auto pack = metalrobo::builtinRobotPack(id);
        if (!pack) {
            std::cerr << "robot catalog cannot inspect listed id: " << id
                      << '\n';
            return false;
        }
        if (pack->id != id) {
            std::cerr << "robot catalog id mismatch: listed " << id
                      << " but inspected " << pack->id << '\n';
            return false;
        }
        if (pack->capabilities.empty() || pack->roles.empty()) {
            std::cerr << "robot catalog metadata is incomplete: " << id
                      << '\n';
            return false;
        }
    }
    std::cout << "{\"schema\":\"numi.robot-catalog-check.v1\",\"robots\":"
              << ids.size() << ",\"status\":\"ok\"}\n";
    return true;
}
}

int main(const int argc, const char* const* argv) {
    if (argc == 2 && std::string_view{argv[1]} == "--self-check") {
        return selfCheck() ? 0 : 1;
    }
    if (argc == 1) {
        std::cout << "{\"schema\":\"numi.robot-catalog.v1\",\"robots\":[";
        const auto ids = metalrobo::builtinRobotIds();
        for (std::size_t index = 0u; index < ids.size(); ++index) {
            if (index != 0u) std::cout << ',';
            robot(*metalrobo::builtinRobotPack(ids[index]));
        }
        std::cout << "]}\n";
        return 0;
    }
    if (argc == 2) {
        const auto pack = metalrobo::builtinRobotPack(argv[1]);
        if (!pack) {
            std::cerr << "unknown robot: " << argv[1] << '\n';
            return 2;
        }
        std::cout << "{\"schema\":\"numi.robot-pack.v1\",\"robot\":";
        robot(*pack);
        std::cout << "}\n";
        return 0;
    }
    std::cerr << "usage: metalrobo_robot_catalog [robot-id]\n";
    return 2;
}
