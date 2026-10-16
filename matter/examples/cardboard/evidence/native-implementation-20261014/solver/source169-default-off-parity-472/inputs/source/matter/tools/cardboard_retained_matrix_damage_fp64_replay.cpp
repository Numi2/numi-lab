// Independent CPU/FP64 replay of the exact FP32 inputs emitted by the native
// retained-matrix damage tangent diagnostic. The included source supplies the
// bytecode interpreter, FP64 local Newton solve, and compiler-backed tangent
// chain already used by the source-bound CPU checks.
#define main cardboardRetainedMatrixDamageFullCheckMain
#include "cardboard_retained_matrix_damage_check.cpp"
#undef main

#include <fstream>

namespace {

struct NativeResponse {
    bool valid = false;
    bool projectionValid = false;
    double residual = 0.0;
    LocalState state;
    std::array<double, 9> stress{};
    std::array<double, 9> tangent{};
};

double readFloatRoundTrip(std::istream& input) {
    double parsed = 0.0;
    if (!(input >> parsed)) throw std::runtime_error("malformed replay input");
    return static_cast<double>(static_cast<float>(parsed));
}

template <std::size_t N>
std::array<double, N> readFloatArray(std::istream& input) {
    std::array<double, N> values{};
    for (auto& value : values) value = readFloatRoundTrip(input);
    return values;
}

NativeResponse readNativeResponse(std::istream& input, const std::size_t stateCount) {
    NativeResponse result;
    unsigned valid = 0u;
    unsigned projectionValid = 0u;
    if (!(input >> valid >> projectionValid >> result.residual))
        throw std::runtime_error("malformed native response header");
    result.valid = valid != 0u;
    result.projectionValid = projectionValid != 0u;
    result.stress = readFloatArray<9u>(input);
    result.tangent = readFloatArray<9u>(input);
    result.state.resize(stateCount);
    for (auto& value : result.state) value = readFloatRoundTrip(input);
    return result;
}

ExplicitDamageSet nativeInstrumentSet() {
    auto set = syntheticSet();
    set.setId = "retained_matrix_native_instrument_synthetic_v1";
    set.sha256 = std::string(64u, 'a');
    set.evidenceClass = "synthetic_instrument_only_not_calibration";
    for (auto& mode : set.fractureModes) mode = {1.0e8, 1.0e6, 0.04};
    set.fractureModes[0] = {2.0e6, 0.4, 0.004};
    set.damageCap = 1.0;
    set.smoothStrainWidth = 1.0e-7;
    return set;
}

double maxStateDifference(const LocalState& first, const LocalState& second) {
    if (first.size() != second.size())
        throw std::runtime_error("state count mismatch in replay");
    double maximum = 0.0;
    for (std::size_t i = 0u; i < first.size(); ++i)
        maximum = std::max(maximum, std::abs(first[i] - second[i]));
    return maximum;
}

double projectionResidualInf(const ConstitutiveProgram& program,
                             const Matrix9& f, const LocalState& accepted,
                             const LocalState& candidate, const double dt) {
    const Matrix9 zero{};
    double result = 0.0;
    for (const auto& residual : program.implicitResiduals)
        result = std::max(result, std::abs(evaluate(residual, program.material,
            f, zero, accepted, candidate, dt)));
    return result;
}

std::vector<double> asVector(const std::array<double, 9>& values) {
    return {values.begin(), values.end()};
}

double relativeError9(const std::array<double, 9>& reference,
                      const std::array<double, 9>& actual) {
    return relativeError(asVector(reference), asVector(actual));
}

void runReplay(const std::filesystem::path& materialPath,
               const std::filesystem::path& replayPath) {
    std::ifstream input(replayPath);
    require(input.good(), "cannot open prepared replay input: " + replayPath.string());
    double dtInput = 0.0;
    double epsilon = 0.0;
    if (!(input >> dtInput >> epsilon))
        throw std::runtime_error("replay input lacks dt and epsilon");
    const double dt = static_cast<double>(static_cast<float>(dtInput));
    const Matrix9 f = readFloatArray<9u>(input);
    const Matrix9 direction = readFloatArray<9u>(input);
    const Matrix9 plusF = readFloatArray<9u>(input);
    const Matrix9 minusF = readFloatArray<9u>(input);
    std::size_t stateCount = 0u;
    if (!(input >> stateCount) || stateCount == 0u ||
        stateCount > NM_MAX_MATERIAL_STATE)
        throw std::runtime_error("replay input has an invalid accepted-state count");
    LocalState accepted(stateCount);
    for (auto& value : accepted) value = readFloatRoundTrip(input);
    const auto nativeCenter = readNativeResponse(input, stateCount);
    const auto nativePlus = readNativeResponse(input, stateCount);
    const auto nativeMinus = readNativeResponse(input, stateCount);
    require(nativeCenter.valid && nativePlus.valid && nativeMinus.valid &&
            nativeCenter.projectionValid && nativePlus.projectionValid &&
            nativeMinus.projectionValid,
            "native diagnostic includes an invalid response; FP64 comparison is descriptive only");

    const auto source = parse(materialPath);
    const auto region = source.name == "hajali2009_liner_hill_ideal" ?
        std::string("liner") : std::string("medium");
    auto element = request(source, region, 0u, {0.0, 0.0, 0.0}, 0.001);
    element.explicitSet = nativeInstrumentSet();
    const auto built = buildElementProgram(element);
    const auto compiled = numi::matter::detail::compileConstitutive(
        built.program, NM_EXPRESSION_STACK_CAPACITY);
    require(compiled.succeeded(), "FP64 replay material compilation failed");
    require(compiled.program.material.internalState.size() == stateCount,
            "captured state count does not match rebuilt source-bound material");

    const auto fp64Center = project(compiled.program, f, accepted, dt, 1.0e-12);
    const auto fp64Plus = project(compiled.program, plusF, accepted, dt, 1.0e-12);
    const auto fp64Minus = project(compiled.program, minusF, accepted, dt, 1.0e-12);
    const auto fp64CenterStress = stress(compiled.program, f, accepted,
                                         fp64Center.state, dt);
    const auto nativeCandidateStress = stress(compiled.program, f, accepted,
                                              nativeCenter.state, dt);
    Matrix9 energyPlusF = f;
    Matrix9 energyMinusF = f;
    constexpr double energyStep = 1.0e-7;
    energyPlusF[0] += energyStep;
    energyMinusF[0] -= energyStep;
    const double energyGradientP11 =
        (evaluateEnergy(compiled.program.material, energyPlusF, accepted,
                        fp64Center.state, dt) -
         evaluateEnergy(compiled.program.material, energyMinusF, accepted,
                        fp64Center.state, dt)) / (2.0 * energyStep);
    const auto fp64PlusStress = stress(compiled.program, plusF, accepted,
                                       fp64Plus.state, dt);
    const auto fp64MinusStress = stress(compiled.program, minusF, accepted,
                                        fp64Minus.state, dt);
    const auto tangentAtFp64Candidate = consistentTangentAction(
        compiled.program, f, accepted, fp64Center.state, direction, dt);
    const auto tangentAtNativeCandidate = consistentTangentAction(
        compiled.program, f, accepted, nativeCenter.state, direction, dt);
    std::array<double, 9> fdAtCapturedInputs{};
    std::array<double, 9> fdNativeAtCapturedInputs{};
    std::array<double, 9> tangentFp64Array{};
    std::array<double, 9> tangentNativeCandidateArray{};
    const double hDirectional = [&] {
        double displacement2 = 0.0;
        double direction2 = 0.0;
        for (std::size_t i = 0u; i < 9u; ++i) {
            const double displacement = plusF[i] - minusF[i];
            displacement2 += displacement * displacement;
            direction2 += direction[i] * direction[i];
        }
        return 0.5 * std::sqrt(displacement2 / std::max(direction2, 1.0e-300));
    }();
    (void)epsilon;
    for (std::size_t i = 0u; i < 9u; ++i) {
        fdAtCapturedInputs[i] =
            (fp64PlusStress[i] - fp64MinusStress[i]) / (2.0 * hDirectional);
        fdNativeAtCapturedInputs[i] =
            (nativePlus.stress[i] - nativeMinus.stress[i]) / (2.0 * hDirectional);
        tangentFp64Array[i] = tangentAtFp64Candidate[i];
        tangentNativeCandidateArray[i] = tangentAtNativeCandidate[i];
    }
    const auto nativeTangentArray = nativeCenter.tangent;
    std::cout << std::setprecision(17)
              << "evidence=exact_fp32_capture_independent_fp64_bytecode_ad_replay"
              << " calibration=false physical_validation=false\n"
              << "material=" << source.name
              << " state_count=" << stateCount
              << " dt_f32=" << dt
              << " requested_epsilon=" << epsilon
              << " actual_directional_half_step=" << hDirectional << '\n'
              << "reference_volume_m3=" << built.mapping.referenceVolumeM3 << '\n';
    for (std::size_t mode = 0u; mode < built.mapping.modeLengthM.size(); ++mode) {
        std::cout << "mode_geometry index=" << mode
                  << " length_m=" << built.mapping.modeLengthM[mode]
                  << " normal_onset_strain=" << built.mapping.modeOnsetStrain[2u * mode]
                  << " normal_softening_strain=" << built.mapping.modeSofteningStrain[2u * mode]
                  << " shear_onset_strain=" << built.mapping.modeOnsetStrain[2u * mode + 1u]
                  << " shear_softening_strain=" << built.mapping.modeSofteningStrain[2u * mode + 1u]
                  << '\n';
    }
    std::cout
              << "native_center_projection_residual=" << nativeCenter.residual
              << " fp64_center_projection_residual=" << fp64Center.residual
              << "native_center_state_vs_fp64_inf="
              << maxStateDifference(nativeCenter.state, fp64Center.state) << '\n'
              << "native_center_implicit_residual_inf="
              << projectionResidualInf(compiled.program, f, accepted,
                                       nativeCenter.state, dt) << '\n'
              << "native_plus_implicit_residual_inf="
              << projectionResidualInf(compiled.program, plusF, accepted,
                                       nativePlus.state, dt) << '\n'
              << "native_minus_implicit_residual_inf="
              << projectionResidualInf(compiled.program, minusF, accepted,
                                       nativeMinus.state, dt) << '\n'
              << "native_stress_vs_fp64_projected_stress_relative="
              << relativeError9(fp64CenterStress, nativeCenter.stress) << '\n'
              << "native_stress_vs_fp64_native_candidate_stress_relative="
              << relativeError9(nativeCandidateStress, nativeCenter.stress) << '\n'
              << "fp64_bytecode_stress_P11_vs_energy_FD_relative="
              << std::abs(fp64CenterStress[0] - energyGradientP11) /
                 std::max(1.0, std::abs(energyGradientP11)) << '\n'
              << "fp64_energy_fd_P11_Pa=" << energyGradientP11 << '\n'
              << "native_tangent_vs_fp64_ad_at_fp64_candidate_relative="
              << relativeError9(tangentFp64Array, nativeTangentArray) << '\n'
              << "native_tangent_vs_fp64_ad_at_native_candidate_relative="
              << relativeError9(tangentNativeCandidateArray, nativeTangentArray) << '\n'
              << "fp64_ad_vs_fp64_fd_at_captured_F_inputs_relative="
              << relativeError9(fdAtCapturedInputs, tangentFp64Array) << '\n'
              << "native_fd_vs_fp64_fd_at_captured_F_inputs_relative="
              << relativeError9(fdAtCapturedInputs, fdNativeAtCapturedInputs) << '\n';
    const auto printNine = [](const char* label,
                              const std::array<double, 9>& values) {
        std::cout << label << '=';
        for (std::size_t i = 0u; i < values.size(); ++i)
            std::cout << (i == 0u ? "" : ",") << values[i];
        std::cout << '\n';
    };
    printNine("native_center_stress_Pa", nativeCenter.stress);
    printNine("fp64_center_stress_Pa", fp64CenterStress);
    printNine("fp64_stress_at_native_candidate_Pa", nativeCandidateStress);
    printNine("native_center_tangent_action_Pa", nativeCenter.tangent);
    printNine("fp64_ad_tangent_action_Pa", tangentFp64Array);
    printNine("fp64_fd_tangent_action_Pa", fdAtCapturedInputs);
    printNine("native_fd_tangent_action_Pa", fdNativeAtCapturedInputs);
    for (std::size_t i = 0u; i < stateCount; ++i) {
        if (std::abs(nativeCenter.state[i] - fp64Center.state[i]) > 1.0e-12) {
            std::cout << "state_difference name="
                      << compiled.program.material.internalState[i].name
                      << " native=" << nativeCenter.state[i]
                      << " fp64=" << fp64Center.state[i]
                      << " delta=" << nativeCenter.state[i] - fp64Center.state[i]
                      << '\n';
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error(
                "usage: fp64-replay MATERIAL.nmatter PREPARED_NATIVE_INPUT.txt");
        }
        runReplay(argv[1], argv[2]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "retained matrix FP64 replay failure: " << error.what() << '\n';
        return 1;
    }
}
