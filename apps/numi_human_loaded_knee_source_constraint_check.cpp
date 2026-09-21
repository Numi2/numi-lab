#include "metalrobo/NumiHumanSourceConstraints.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: " << argv[0]
                  << " HumanPack.loaded-anatomy-knee.v1.json"
                  << " HumanPack.loaded-anatomy-knee.binding.v1.json"
                  << " HumanPack.ownership.v1.json"
                  << " HumanPack.loaded-anatomy-knee.source-compliance.v1.json"
                  << " myosim-fullbody-joint-equalities-source-compliance.nheq"
                  << " myosim-fullbody-joint-limits.nhlim\n";
        return 64;
    }
    metalrobo::NumiHumanLoadedKneeBindingAdmissionV1 base;
    std::string error;
    if (!metalrobo::loadNumiHumanLoadedKneeBindingV1(
            argv[1], argv[2], argv[3], base, error)) {
        std::cerr << "loaded_knee_source_constraint_config=rejected reason=\""
                  << error << "\"\n";
        return 1;
    }
    metalrobo::NumiHumanLoadedKneeSourceComplianceAdmissionV1 source;
    if (!metalrobo::loadNumiHumanLoadedKneeSourceComplianceV1(
            argv[4], argv[5], argv[6], base, source, error)) {
        std::cerr << "loaded_knee_source_constraint_config=rejected reason=\""
                  << error << "\"\n";
        return 1;
    }
    metalrobo::NumiHumanSourceConstraintProgramV1 program;
    if (!metalrobo::loadNumiHumanLoadedKneeSourceConstraintProgramV1(
            argv[5], argv[6], base, source, program, error)) {
        std::cerr << "loaded_knee_source_constraint_config=rejected reason=\""
                  << error << "\"\n";
        return 1;
    }
    numi::matter::RuntimeConfiguration configuration;
    if (!metalrobo::configureNumiHumanSourceConstraintsV1(
            program, configuration, error)) {
        std::cerr << "loaded_knee_source_constraint_config=rejected reason=\""
                  << error << "\"\n";
        return 1;
    }
    std::cout
        << "loaded_knee_source_constraint_config=admitted"
        << " candidate_only=true"
        << " runtime_initialized=false"
        << " runtime_executed=false"
        << " prepared_state_bound=false"
        << " nq=" << configuration.humanEqualityDispatch.qCount
        << " nv=" << configuration.humanEqualityDispatch.dofCount
        << " equality_rows=" << configuration.humanJointEqualities.size()
        << " limit_rows=" << configuration.humanJointLimits.size()
        << " equality_fingerprint="
        << configuration.humanEqualitySourceFingerprint
        << " limit_fingerprint="
        << configuration.humanLimitSourceFingerprint
        << "\n";
    return 0;
}
