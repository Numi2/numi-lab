#pragma once
#include "numi/matter/matter.hpp"
namespace numi::matter::detail {
bool compileVascular(const WorldSource&, CompiledWorld&, std::vector<Diagnostic>&);
bool validateVascularLayout(const CompiledWorld&, std::string* error);
}
