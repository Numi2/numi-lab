
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
struct Parsed { bool valid; std::uint32_t root, iteration; bool allNewton; };
Parsed parse(const char* baseCauchySelector, std::uint32_t control,
             std::uint32_t femBaseCGLSIterations, bool femBaseCGLSPrimaryPolicyRequested) {
    struct { std::uint32_t controlStep; } request{control};
    bool valid = true;
    std::uint32_t femBaseCauchyFallbackRoot=0, femBaseCauchyFallbackIteration=0;
    bool femBaseCauchyFallbackAllNewtonIterations = false;
    auto rejectBaseCauchy = [&](const char*) { valid=false; };
                const char* separator = std::strchr(baseCauchySelector, ':');
                const char* end = baseCauchySelector +
                    std::strlen(baseCauchySelector);
                const auto digitsOnly = [](const char* begin,
                                           const char* finish) {
                    if (begin >= finish) return false;
                    for (const char* cursor = begin; cursor != finish;
                         ++cursor) {
                        if (*cursor < '0' || *cursor > '9') return false;
                    }
                    return true;
                };
                char* rootEnd = nullptr;
                char* iterationEnd = nullptr;
                const bool everyControlStep = separator != nullptr &&
                    separator - baseCauchySelector == 3 &&
                    std::strncmp(baseCauchySelector, "all", 3) == 0;
                const unsigned long root = everyControlStep
                    ? request.controlStep : std::strtoul(
                        baseCauchySelector, &rootEnd, 10);
                const bool allNewtonIterations = separator != nullptr &&
                    std::strcmp(separator + 1, "all") == 0;
                const unsigned long iteration = separator == nullptr ||
                    allNewtonIterations ? 0ul :
                    std::strtoul(separator + 1, &iterationEnd, 10);
                if (separator == nullptr ||
                    (!everyControlStep && rootEnd != separator) ||
                    (!allNewtonIterations && iterationEnd != end) ||
                    (!everyControlStep &&
                     !digitsOnly(baseCauchySelector, separator)) ||
                    (!allNewtonIterations &&
                     !digitsOnly(separator + 1, end)) ||
                    root > std::numeric_limits<std::uint32_t>::max() ||
                    iteration > 256ul) {
                    rejectBaseCauchy(
                        "selector_must_be_control_step_colon_all_or_newton_0_to_256_or_exact_all_colon_all");
                } else if (everyControlStep &&
                           (!allNewtonIterations ||
                            femBaseCGLSIterations == 0u ||
                            !femBaseCGLSPrimaryPolicyRequested)) {
                    // All-control scope is admitted only for the existing
                    // bounded CGLS primary policy. Every selected control still
                    // passes the complete original capability/certificate gate.
                    rejectBaseCauchy(
                        "all_control_steps_requires_cgls_primary_policy_and_all_newton_suffix");
                } else {
                    femBaseCauchyFallbackRoot =
                        static_cast<std::uint32_t>(root);
                    femBaseCauchyFallbackIteration =
                        static_cast<std::uint32_t>(iteration);
                    femBaseCauchyFallbackAllNewtonIterations =
                        allNewtonIterations;
                }

    return {valid, femBaseCauchyFallbackRoot, femBaseCauchyFallbackIteration,
            femBaseCauchyFallbackAllNewtonIterations};
}
bool trueRoot(const char* value, std::uint32_t control) {
    setenv("NM_FEM_TRUE_RESIDUAL_ROOT", value, 1);
    struct { std::uint32_t controlStep; } request{control};
                if (const char* selector = std::getenv("NM_FEM_TRUE_RESIDUAL_ROOT")) {
                const bool everyControlStep = std::strcmp(selector, "all") == 0;
                char* end = nullptr;
                const unsigned long parsed = everyControlStep
                    ? request.controlStep : std::strtoul(selector, &end, 10);
                if ((!everyControlStep && (end == selector || *end != '\0')) ||
                    parsed > std::numeric_limits<std::uint32_t>::max()) {

        return false;
    } else { return parsed == control; }
  }
  return false;
}
int main() {
    unsigned checks=0;
    auto check=[&](bool value) { ++checks; assert(value); };
    for (auto control: {0u, 1u, 29u, 4294967295u}) {
        auto p=parse("all:all",control,64,true);
        check(p.valid && p.root==control && p.iteration==0 && p.allNewton);
        check(trueRoot("all",control));
    }
    for (auto value: {"all:0", "all:48", "all:256", "all:257", "all:",
                      "All:all", "alll:all", "all:All", "all:all:all",
                      " all:all", "all:all ", "all", ":all", "", "-1:all",
                      "4294967296:all", "1:257", "1:-1", "1:48:2"})
        check(!parse(value,1,64,true).valid);
    check(!parse("all:all",1,0,true).valid);
    check(!parse("all:all",1,64,false).valid);
    check(!parse("all:all",1,0,false).valid);
    for (auto value: {"0:all", "1:all", "4294967295:all", "1:0", "1:256", "01:048"}) {
        auto p=parse(value,29,0,false);
        check(p.valid);
    }
    auto legacy=parse("19:3",42,0,false);
    check(legacy.valid && legacy.root==19 && legacy.iteration==3 && !legacy.allNewton);
    check(trueRoot("19",19)); check(!trueRoot("19",20));
    for (auto value: {"All", "alll", "all:all", "", "4294967296"})
        check(!trueRoot(value,1));
    std::cout<<"selector_checks="<<checks<<" passed\n";
}
