#pragma once

#include <libxml/tree.h>

#include <filesystem>
#include <memory>

namespace numi {

struct MJCFImport {
    std::unique_ptr<xmlDoc, decltype(&xmlFreeDoc)> urdf{nullptr, xmlFreeDoc};
    float gravity = 9.81f;
    unsigned physicsSubsteps = 10;
    bool ground = false;
};

MJCFImport importMJCF(const std::filesystem::path& path);

} // namespace numi
