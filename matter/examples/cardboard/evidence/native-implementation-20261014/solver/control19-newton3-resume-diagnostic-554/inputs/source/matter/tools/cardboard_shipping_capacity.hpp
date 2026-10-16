#pragma once

// A narrow, source-bound admission record for using a measured local
// compression peak to size shipping loads. Version 2 also records the shared
// middle comparison and requires all three observed differences to decrease
// from coarse/middle to middle/fine. This is numerical capacity qualification
// only; it does not qualify a material grade or physical box.

#include <cmath>
#include <array>
#include <cstdint>
#include <istream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace numi::cardboard {

inline constexpr const char* kShippingCapacityQualificationSchema =
    "numi.cardboard.shipping-capacity-qualification.v2";
inline constexpr double kShippingCapacityPeakTolerance = 0.02;
inline constexpr double kShippingCapacityCurveTolerance = 0.05;
inline constexpr double kShippingCapacityWorkTolerance = 0.02;
inline constexpr double kShippingCapacityEndpointTolerance = 1.0e-9;

struct ShippingCapacityQualification {
    std::string schema;
    std::string status;
    std::string boardRecipeId;
    std::uint64_t sourcePhysicsFingerprint = 0u;
    std::uint64_t deviceProgramFingerprint = 0u;
    std::uint64_t compiledWorldFingerprint = 0u;
    std::string sourceBindingSha256;
    std::string assembledMatterCheckpointSha256;
    std::string assembledRigidSidecarSha256;
    std::string nominalCurveSha256;
    std::uint64_t nominalPeakSampleIndex = 0u;
    std::uint64_t nominalAcceptedSampleCount = 0u;
    double nominalTerminalStrokeFraction = 0.0;
    bool nominalFullStrokeCompleted = false;
    bool nominalRateSensitivityPassed = false;

    std::string meshStatus;
    std::string meshReportFile;
    std::string meshReportSha256;
    std::uint32_t meshLevelCount = 0u;
    bool meshAllCasesComplete = false;
    bool meshFinestPairWithinTolerance = false;
    double meshFinestPeakDifference = 0.0;
    double meshFinestCurveRms = 0.0;
    double meshFinestWorkDifference = 0.0;
    double meshCoarseMiddlePeakDifference = 0.0;
    double meshCoarseMiddleCurveRms = 0.0;
    double meshCoarseMiddleWorkDifference = 0.0;

    std::string timeStatus;
    std::string timeReportFile;
    std::string timeReportSha256;
    std::uint32_t timeLevelCount = 0u;
    bool timeAllCasesComplete = false;
    bool timeFinestPairWithinTolerance = false;
    double timeFinestPeakDifference = 0.0;
    double timeFinestCurveRms = 0.0;
    double timeFinestWorkDifference = 0.0;
    double timeCoarseMiddlePeakDifference = 0.0;
    double timeCoarseMiddleCurveRms = 0.0;
    double timeCoarseMiddleWorkDifference = 0.0;
};

struct ShippingCapacityIdentity {
    std::string boardRecipeId;
    std::uint64_t sourcePhysicsFingerprint = 0u;
    std::uint64_t deviceProgramFingerprint = 0u;
    std::uint64_t compiledWorldFingerprint = 0u;
    std::string sourceBindingSha256;
    std::string assembledMatterCheckpointSha256;
    std::string assembledRigidSidecarSha256;
    std::string nominalCurveSha256;
    std::uint64_t nominalPeakSampleIndex = 0u;
    std::uint64_t nominalAcceptedSampleCount = 0u;
    double nominalTerminalStrokeFraction = 0.0;
    bool nominalFullStrokeCompleted = false;
    bool nominalRateSensitivityPassed = false;
};

struct ShippingCapacityQualificationResult {
    bool qualified = false;
    std::vector<std::string> reasons;
};

inline bool shippingCapacityDigestValid(const std::string& value) {
    if (value.size() != 64u) return false;
    for (const unsigned char c : value)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return false;
    return true;
}

inline ShippingCapacityQualificationResult validateShippingCapacityQualification(
    const ShippingCapacityQualification& q,
    const ShippingCapacityIdentity& actual) {
    ShippingCapacityQualificationResult result;
    const auto reject = [&](const char* reason) {
        result.reasons.emplace_back(reason);
    };
    if (q.schema != kShippingCapacityQualificationSchema)
        reject("unsupported_schema");
    if (q.status != "qualified_for_numerical_capacity_only")
        reject("study_not_numerically_qualified");
    if (q.boardRecipeId.empty() || q.boardRecipeId != actual.boardRecipeId)
        reject("board_recipe_identity_mismatch");
    if (q.sourcePhysicsFingerprint != actual.sourcePhysicsFingerprint ||
        q.deviceProgramFingerprint != actual.deviceProgramFingerprint ||
        q.compiledWorldFingerprint != actual.compiledWorldFingerprint)
        reject("compiled_source_or_program_fingerprint_mismatch");
    if (!shippingCapacityDigestValid(q.sourceBindingSha256) ||
        q.sourceBindingSha256 != actual.sourceBindingSha256)
        reject("source_material_binding_mismatch");
    if (!shippingCapacityDigestValid(q.assembledMatterCheckpointSha256) ||
        q.assembledMatterCheckpointSha256 != actual.assembledMatterCheckpointSha256 ||
        !shippingCapacityDigestValid(q.assembledRigidSidecarSha256) ||
        q.assembledRigidSidecarSha256 != actual.assembledRigidSidecarSha256)
        reject("assembled_checkpoint_identity_mismatch");
    if (!shippingCapacityDigestValid(q.nominalCurveSha256) ||
        q.nominalCurveSha256 != actual.nominalCurveSha256 ||
        q.nominalPeakSampleIndex != actual.nominalPeakSampleIndex ||
        q.nominalAcceptedSampleCount != actual.nominalAcceptedSampleCount)
        reject("nominal_compression_curve_or_peak_mismatch");
    if (!q.nominalFullStrokeCompleted || !actual.nominalFullStrokeCompleted ||
        !std::isfinite(q.nominalTerminalStrokeFraction) ||
        !std::isfinite(actual.nominalTerminalStrokeFraction) ||
        std::abs(q.nominalTerminalStrokeFraction - 0.70) >
            kShippingCapacityEndpointTolerance ||
        std::abs(actual.nominalTerminalStrokeFraction - 0.70) >
            kShippingCapacityEndpointTolerance ||
        std::abs(q.nominalTerminalStrokeFraction -
                 actual.nominalTerminalStrokeFraction) >
            kShippingCapacityEndpointTolerance)
        reject("nominal_loading_postpeak_endpoint_incomplete");
    if (!q.nominalRateSensitivityPassed || !actual.nominalRateSensitivityPassed)
        reject("nominal_half_speed_sensitivity_not_passed");

    const auto validateStudy = [&](const std::string& status,
                                   const std::string& reportFile,
                                   const std::string& reportSha,
                                   const std::uint32_t levelCount,
                                   const bool allCasesComplete,
                                   const bool finestPairWithinTolerance,
                                   const double peakDifference,
                                   const double curveRms,
                                   const double workDifference,
                                   const double coarseMiddlePeakDifference,
                                   const double coarseMiddleCurveRms,
                                   const double coarseMiddleWorkDifference,
                                   const char* prefix) {
        const auto add = [&](const char* suffix) {
            result.reasons.emplace_back(std::string(prefix) + suffix);
        };
        if (status != "qualified") add("_study_not_qualified");
        if (reportFile.empty() || reportFile.find("..") != std::string::npos ||
            reportFile.front() == '/' || reportFile.find('\\') != std::string::npos)
            add("_report_path_invalid");
        if (!shippingCapacityDigestValid(reportSha)) add("_report_digest_invalid");
        if (levelCount < 3u || !allCasesComplete || !finestPairWithinTolerance)
            add("_levels_incomplete_or_finest_pair_failed");
        if (!(std::isfinite(peakDifference) && peakDifference >= 0.0 &&
              peakDifference <= kShippingCapacityPeakTolerance))
            add("_peak_difference_outside_tolerance");
        if (!(std::isfinite(curveRms) && curveRms >= 0.0 &&
              curveRms <= kShippingCapacityCurveTolerance))
            add("_full_curve_difference_outside_tolerance");
        if (!(std::isfinite(workDifference) && workDifference >= 0.0 &&
              workDifference <= kShippingCapacityWorkTolerance))
            add("_work_difference_outside_tolerance");
        const std::array<double, 3> coarseMiddle{
            coarseMiddlePeakDifference, coarseMiddleCurveRms,
            coarseMiddleWorkDifference};
        const std::array<double, 3> middleFine{
            peakDifference, curveRms, workDifference};
        bool everyMetricDecreases = true;
        for (std::size_t i = 0u; i < coarseMiddle.size(); ++i) {
            if (!(std::isfinite(coarseMiddle[i]) && coarseMiddle[i] >= 0.0 &&
                  std::isfinite(middleFine[i]) && middleFine[i] >= 0.0 &&
                  middleFine[i] < coarseMiddle[i]))
                everyMetricDecreases = false;
        }
        if (!everyMetricDecreases)
            add("_adjacent_differences_do_not_decrease");
    };
    validateStudy(q.meshStatus, q.meshReportFile, q.meshReportSha256,
        q.meshLevelCount, q.meshAllCasesComplete, q.meshFinestPairWithinTolerance,
        q.meshFinestPeakDifference, q.meshFinestCurveRms,
        q.meshFinestWorkDifference, q.meshCoarseMiddlePeakDifference,
        q.meshCoarseMiddleCurveRms, q.meshCoarseMiddleWorkDifference, "mesh");
    validateStudy(q.timeStatus, q.timeReportFile, q.timeReportSha256,
        q.timeLevelCount, q.timeAllCasesComplete, q.timeFinestPairWithinTolerance,
        q.timeFinestPeakDifference, q.timeFinestCurveRms,
        q.timeFinestWorkDifference, q.timeCoarseMiddlePeakDifference,
        q.timeCoarseMiddleCurveRms, q.timeCoarseMiddleWorkDifference, "time");
    result.qualified = result.reasons.empty();
    return result;
}

inline std::map<std::string, std::string> parseShippingCapacityProperties(
    std::istream& input, const std::string& sourceName = "shipping capacity evidence") {
    static const std::set<std::string> allowed{
        "schema", "status", "board_recipe_id", "source_physics_fingerprint",
        "device_program_fingerprint", "compiled_world_fingerprint",
        "source_binding_sha256", "assembled_matter_checkpoint_sha256",
        "assembled_rigid_sidecar_sha256", "nominal_curve_sha256",
        "nominal_peak_sample_index", "nominal_accepted_sample_count",
        "nominal_terminal_stroke_fraction", "nominal_full_stroke_completed",
        "nominal_rate_sensitivity_passed", "mesh_status", "mesh_report_file",
        "mesh_report_sha256", "mesh_level_count", "mesh_all_cases_complete",
        "mesh_finest_pair_within_tolerance", "mesh_finest_peak_difference",
        "mesh_finest_curve_rms", "mesh_finest_work_difference",
        "mesh_coarse_middle_peak_difference", "mesh_coarse_middle_curve_rms",
        "mesh_coarse_middle_work_difference", "time_status",
        "time_report_file", "time_report_sha256", "time_level_count",
        "time_all_cases_complete", "time_finest_pair_within_tolerance",
        "time_finest_peak_difference", "time_finest_curve_rms",
        "time_finest_work_difference", "time_coarse_middle_peak_difference",
        "time_coarse_middle_curve_rms", "time_coarse_middle_work_difference"};
    std::map<std::string, std::string> values;
    std::string line;
    std::uint32_t lineNumber = 0u;
    while (std::getline(input, line)) {
        ++lineNumber;
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        const auto last = line.find_last_not_of(" \t\r\n");
        line = line.substr(first, last - first + 1u);
        if (line.front() == '#') continue;
        const auto separator = line.find('=');
        if (separator == std::string::npos)
            throw std::invalid_argument(sourceName + ":" +
                std::to_string(lineNumber) + " must use key=value syntax");
        const std::string key = line.substr(0u, separator);
        const std::string value = line.substr(separator + 1u);
        if (!allowed.contains(key) || value.empty() ||
            key.find_first_of(" \t") != std::string::npos)
            throw std::invalid_argument(sourceName + ":" +
                std::to_string(lineNumber) + " has an unsupported or empty property");
        if (!values.emplace(key, value).second)
            throw std::invalid_argument(sourceName + ":" +
                std::to_string(lineNumber) + " repeats property " + key);
    }
    if (!input.eof()) throw std::invalid_argument(sourceName + " read failed");
    if (values.size() != allowed.size())
        throw std::invalid_argument(sourceName + " is missing required properties");
    return values;
}

inline ShippingCapacityQualification parseShippingCapacityQualification(
    std::istream& input, const std::string& sourceName = "shipping capacity evidence") {
    const auto values = parseShippingCapacityProperties(input, sourceName);
    const auto get = [&](const char* key) -> const std::string& {
        return values.at(key);
    };
    const auto parseU64 = [&](const char* key) {
        const auto& raw = get(key);
        std::size_t used = 0u;
        unsigned long long value = 0u;
        try { value = std::stoull(raw, &used); }
        catch (...) { throw std::invalid_argument(sourceName + " has invalid " + key); }
        if (used != raw.size()) throw std::invalid_argument(sourceName + " has invalid " + key);
        return static_cast<std::uint64_t>(value);
    };
    const auto parseU32 = [&](const char* key) {
        const auto value = parseU64(key);
        if (value > UINT32_MAX) throw std::invalid_argument(sourceName + " has out-of-range " + key);
        return static_cast<std::uint32_t>(value);
    };
    const auto parseDouble = [&](const char* key) {
        const auto& raw = get(key);
        std::size_t used = 0u;
        double value = 0.0;
        try { value = std::stod(raw, &used); }
        catch (...) { throw std::invalid_argument(sourceName + " has invalid " + key); }
        if (used != raw.size() || !std::isfinite(value))
            throw std::invalid_argument(sourceName + " has invalid " + key);
        return value;
    };
    const auto parseBool = [&](const char* key) {
        const auto& value = get(key);
        if (value == "true") return true;
        if (value == "false") return false;
        throw std::invalid_argument(sourceName + " has invalid boolean " + key);
    };
    ShippingCapacityQualification q;
    q.schema = get("schema"); q.status = get("status");
    q.boardRecipeId = get("board_recipe_id");
    q.sourcePhysicsFingerprint = parseU64("source_physics_fingerprint");
    q.deviceProgramFingerprint = parseU64("device_program_fingerprint");
    q.compiledWorldFingerprint = parseU64("compiled_world_fingerprint");
    q.sourceBindingSha256 = get("source_binding_sha256");
    q.assembledMatterCheckpointSha256 = get("assembled_matter_checkpoint_sha256");
    q.assembledRigidSidecarSha256 = get("assembled_rigid_sidecar_sha256");
    q.nominalCurveSha256 = get("nominal_curve_sha256");
    q.nominalPeakSampleIndex = parseU64("nominal_peak_sample_index");
    q.nominalAcceptedSampleCount = parseU64("nominal_accepted_sample_count");
    q.nominalTerminalStrokeFraction = parseDouble("nominal_terminal_stroke_fraction");
    q.nominalFullStrokeCompleted = parseBool("nominal_full_stroke_completed");
    q.nominalRateSensitivityPassed = parseBool("nominal_rate_sensitivity_passed");
    q.meshStatus = get("mesh_status"); q.meshReportFile = get("mesh_report_file");
    q.meshReportSha256 = get("mesh_report_sha256");
    q.meshLevelCount = parseU32("mesh_level_count");
    q.meshAllCasesComplete = parseBool("mesh_all_cases_complete");
    q.meshFinestPairWithinTolerance = parseBool("mesh_finest_pair_within_tolerance");
    q.meshFinestPeakDifference = parseDouble("mesh_finest_peak_difference");
    q.meshFinestCurveRms = parseDouble("mesh_finest_curve_rms");
    q.meshFinestWorkDifference = parseDouble("mesh_finest_work_difference");
    q.meshCoarseMiddlePeakDifference = parseDouble("mesh_coarse_middle_peak_difference");
    q.meshCoarseMiddleCurveRms = parseDouble("mesh_coarse_middle_curve_rms");
    q.meshCoarseMiddleWorkDifference = parseDouble("mesh_coarse_middle_work_difference");
    q.timeStatus = get("time_status"); q.timeReportFile = get("time_report_file");
    q.timeReportSha256 = get("time_report_sha256");
    q.timeLevelCount = parseU32("time_level_count");
    q.timeAllCasesComplete = parseBool("time_all_cases_complete");
    q.timeFinestPairWithinTolerance = parseBool("time_finest_pair_within_tolerance");
    q.timeFinestPeakDifference = parseDouble("time_finest_peak_difference");
    q.timeFinestCurveRms = parseDouble("time_finest_curve_rms");
    q.timeFinestWorkDifference = parseDouble("time_finest_work_difference");
    q.timeCoarseMiddlePeakDifference = parseDouble("time_coarse_middle_peak_difference");
    q.timeCoarseMiddleCurveRms = parseDouble("time_coarse_middle_curve_rms");
    q.timeCoarseMiddleWorkDifference = parseDouble("time_coarse_middle_work_difference");
    return q;
}

} // namespace numi::cardboard
