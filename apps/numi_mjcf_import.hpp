#pragma once

#include <libxml/tree.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace numi {

struct MJCFMotor {
    std::string name;
    std::string joint;
    double effortLimit;
};

struct MJCFImport {
    std::unique_ptr<xmlDoc, decltype(&xmlFreeDoc)> urdf{nullptr, xmlFreeDoc};
    std::vector<MJCFMotor> motors;
    float gravity = 9.81f;
    unsigned physicsSubsteps = 10;
    bool ground = false;
};

MJCFImport importMJCF(const std::filesystem::path& path);

} // namespace numi
