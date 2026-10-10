#include "../apps/NumiHumanAcceptedQAuditWindow.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string_view>

namespace {
using numiHumanAcceptedQAuditWindow::ParseError;
using numiHumanAcceptedQAuditWindow::parse;
using numiHumanAcceptedQAuditWindow::Window;

void expect(const bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void expectError(const ParseError actual, const ParseError expected,
                 const char* message) {
    expect(actual == expected, message);
}
}  // namespace

int main() {
    const auto disabled = parse(false, std::nullopt, std::nullopt, 155000u);
    expect(disabled.ok() && !disabled.window.enabled,
           "default-off remains disabled with no bounds");
    expect(disabled.window.segmentSteps(0u, 155000u, 8u) == 8u &&
               disabled.window.segmentSteps(154992u, 8u, 8u) == 8u,
           "default-off preserves ordinary cap-eight segmentation");
    expect(disabled.window.segmentSteps(154998u, 2u, 8u) == 2u,
           "default-off preserves a short terminal segment");

    const auto unbounded = parse(true, std::nullopt, std::nullopt, 100u);
    expect(unbounded.ok() && unbounded.window.enabled &&
               !unbounded.window.bounded && unbounded.window.firstAcceptedStep == 1u &&
               unbounded.window.lastAcceptedStep == 100u &&
               unbounded.window.rowCount() == 100u,
           "existing unbounded audit retains the full accepted horizon");
    expect(unbounded.window.segmentSteps(0u, 100u, 1u) == 1u &&
               unbounded.window.segmentSteps(40u, 60u, 1u) == 1u,
           "existing unbounded audit remains one submission per accepted step");

    const auto bounded = parse(true, std::string_view{"10"},
                                std::string_view{"12"}, 20u);
    expect(bounded.ok() && bounded.window.bounded &&
               bounded.window.rowCount() == 3u,
           "bounded accepted-step range parses with inclusive row count");
    const Window& w = bounded.window;
    expect(w.segmentSteps(0u, 20u, 8u) == 8u,
           "normal cap is preserved before the window");
    expect(w.segmentSteps(8u, 12u, 8u) == 1u,
           "normal segment splits before the first accepted sample");
    expect(w.segmentSteps(9u, 11u, 8u) == 1u &&
               w.segmentSteps(10u, 10u, 8u) == 1u &&
               w.segmentSteps(11u, 9u, 8u) == 1u,
           "each step inside the inclusive window has a one-step submission");
    expect(w.segmentSteps(12u, 8u, 8u) == 4u &&
               w.segmentSteps(16u, 4u, 8u) == 4u,
           "normal cadence realigns after an unaligned window end, then resumes cap-sized segments");
    expect(w.contains(10u) && w.contains(12u) &&
               !w.contains(9u) && !w.contains(13u),
           "sample membership uses one-based inclusive endpoints");

    const auto boundaryAligned = parse(true, std::string_view{"9"},
                                       std::string_view{"16"}, 20u);
    expect(boundaryAligned.ok() &&
               boundaryAligned.window.segmentSteps(0u, 20u, 8u) == 8u &&
               boundaryAligned.window.segmentSteps(8u, 12u, 8u) == 1u,
           "a window beginning on a normal segment boundary still becomes cap one");
    expect(w.segmentSteps(18u, 2u, 8u) == 2u,
           "normal segmentation stays bounded by remaining horizon");
    using numiHumanAcceptedQAuditWindow::shouldPresentAcceptedPose;
    expect(shouldPresentAcceptedPose(true, 32u, 8u, false, true, 3u, 512u),
           "legacy COM observer keeps the existing cap-at-least-render shortcut");
    expect(!shouldPresentAcceptedPose(true, 32u, 8u, true, false, 32u, 512u) &&
               shouldPresentAcceptedPose(true, 32u, 8u, true, true, 32u, 512u) &&
               shouldPresentAcceptedPose(true, 32u, 8u, true, true, 512u, 512u),
           "bounded Q audit preserves both the normal observer grid and render cadence");
    expect(!shouldPresentAcceptedPose(true, 8u, 32u, true, true, 8u, 512u) &&
               shouldPresentAcceptedPose(true, 8u, 32u, true, true, 32u, 512u),
           "cap-eight bounded mode keeps the same every-32-step presentation as control");
    expect(!shouldPresentAcceptedPose(true, 7u, 32u, true, false, 32u, 512u) &&
               shouldPresentAcceptedPose(true, 7u, 32u, true, true, 224u, 512u),
           "non-divisor cap seven suppresses off-grid frame 32 and resumes at common step 224");
    expect(shouldPresentAcceptedPose(false, 32u, 8u, true, true, 3u, 512u),
           "without COM auditing the normal presentation behavior is unchanged");
    const auto unalignedEnd = parse(true, std::string_view{"9"},
                                    std::string_view{"11"}, 24u);
    expect(unalignedEnd.ok() &&
               unalignedEnd.window.segmentSteps(11u, 13u, 8u) == 5u &&
               unalignedEnd.window.segmentSteps(16u, 8u, 8u) == 8u,
           "post-window observer endpoints return to the original global stride");

    expectError(parse(false, std::string_view{"2"}, std::string_view{"3"},
                      10u).error,
                ParseError::boundsWithoutAudit,
                "bounds cannot silently enable or affect the audit");
    expectError(parse(true, std::string_view{"2"}, std::nullopt, 10u).error,
                ParseError::incompleteBounds,
                "both bounds are required together");
    expectError(parse(true, std::string_view{"x"}, std::string_view{"3"},
                      10u).error,
                ParseError::invalidInteger,
                "non-integer bound is rejected");
    expectError(parse(true, std::string_view{""}, std::string_view{"3"},
                      10u).error,
                ParseError::invalidInteger,
                "explicit empty bound is rejected rather than treated as absent");
    expectError(parse(true, std::string_view{"2 "}, std::string_view{"3"},
                      10u).error,
                ParseError::invalidInteger,
                "trailing characters in a bound are rejected");
    expectError(parse(true, std::string_view{"0"}, std::string_view{"3"},
                      10u).error,
                ParseError::outOfRange,
                "accepted steps are one-based");
    expectError(parse(true, std::string_view{"2"}, std::string_view{"11"},
                      10u).error,
                ParseError::outOfRange,
                "bound beyond the accepted horizon is rejected");
    expectError(parse(true, std::string_view{"8"}, std::string_view{"7"},
                      10u).error,
                ParseError::reversedRange,
                "reversed range is rejected");
    expectError(parse(true, std::nullopt, std::nullopt, 0u).error,
                ParseError::emptyHorizon,
                "empty horizon cannot enable Q sampling");
    return 0;
}
