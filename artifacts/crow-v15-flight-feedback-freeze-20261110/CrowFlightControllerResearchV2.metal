// v15 flight controller V1; see artifacts/crow-frozen-flight-law-20261109.
inline float crowFlightControllerV1BaseAction(
    const uint action,
    const MRTaskStateGPU state,
    const float controlStepSeconds,
    const bool navigation
) {
    const float seconds =
        float(state.episode.x) * controlStepSeconds;
    const bool groundOnly = state.episode.z <= 1u;
    const bool takeoffOnly = state.episode.z == 2u;
    const bool isolatedCruise = state.episode.z == 3u;
    const bool takeoffCruise = state.episode.z == 4u;
    const bool straightCruise =
        takeoffOnly || isolatedCruise || takeoffCruise || navigation;
    const float launchStart = groundOnly ? 32.0f
        : navigation ? 2.0f
        : takeoffOnly || isolatedCruise ? 0.0f
        : takeoffCruise ? 1.0f : 5.0f;
    const float approachStart = straightCruise ? 32.0f : 21.0f;
    const float landingStart = straightCruise ? 32.0f : 27.0f;
    const bool airborne = seconds >= launchStart && seconds < landingStart;
    float targetHeight = 0.1873f;
    if (isolatedCruise || navigation) {
        targetHeight = 0.85f;
    } else if (seconds >= launchStart && seconds < launchStart + 4.0f) {
        targetHeight = mix(
            0.1873f,
            0.85f,
            clamp((seconds - launchStart) / 4.0f, 0.0f, 1.0f)
        );
    } else if (seconds >= launchStart + 4.0f &&
               seconds < approachStart) {
        targetHeight = 0.85f;
    } else if (seconds >= approachStart && seconds < landingStart) {
        targetHeight = mix(
            0.85f,
            0.45f,
            clamp(
                (seconds - approachStart) /
                    max(landingStart - approachStart, 1.0f),
                0.0f,
                1.0f
            )
        );
    }
    // Navigation targets carry authored altitude as well as planar route.
    // The former showcase teacher always regulated to 0.85 m and therefore
    // actively suppressed the climb from the second slalom to the 1.155 m
    // perch-approach target. Task completion publishes accepted target-height,
    // vertical-rate, pitch, and pitch-rate feedback in the private teacher
    // lane below; autonomous actors neither observe nor consume this lane.
    const float heightError = navigation
        ? state.threatGeometry.x
        : targetHeight - state.airReturnTracking.y;
    const float verticalRate = navigation
        ? state.threatGeometry.y
        : 0.0f;
    const float desiredVerticalRate = navigation
        ? clamp(1.50f * heightError, -0.50f, 0.50f)
        : 0.0f;
    const float forwardSpeedError = navigation
        ? state.commandExtension.z -
              max(state.commandAndPhase.x, 0.15f)
        : straightCruise
        ? state.commandExtension.z - 0.35f
        : state.threatGeometry.x;
    const float lateralSpeedError = navigation
        ? state.threatTeacher.x
        : state.threatGeometry.y;
    const float headingError = state.commandExtension.x;
    const float yawRate = straightCruise || navigation
        ? state.commandExtension.y
        : state.threatGeometry.z;
    const float targetRoll = clamp(0.25f * lateralSpeedError, -0.12f, 0.12f);
    const float navigationBankCorrection = navigation
        ? clamp(1.50f * (targetRoll - state.threatTeacher.y) - 0.25f * state.threatTeacher.z, -0.45f, 0.45f)
        : 0.0f;
    const float navigationPitchTarget = navigation
        ? clamp(
              0.10f + 0.35f * heightError -
                  0.12f * verticalRate - 1.20f * forwardSpeedError,
              -0.25f,
              0.25f
          )
        : 0.0f;
    if (action < 2u) {
        return airborne
            ? clamp(
                  -0.150f + 0.350f * heightError +
                      0.300f * (desiredVerticalRate - verticalRate) -
                      0.120f * forwardSpeedError +
                      (action == 0u
                          ? navigationBankCorrection
                          : -navigationBankCorrection),
                  -0.500f,
                  0.500f
              )
            : 0.0f;
    }
    if (action == 2u || action == 3u) {
        if (straightCruise) {
            const float correction = clamp(
                -0.25f * headingError - 0.08f * yawRate,
                -0.22f,
                0.22f
            );
            return action == 2u ? correction : -correction;
        }
        if (seconds < 9.0f || seconds >= 24.0f) {
            return 0.0f;
        }
        const float correction = clamp(
            -0.18f * lateralSpeedError - 0.10f * yawRate,
            -0.18f,
            0.18f
        );
        return action == 2u ? correction : -correction;
    }
    if (action == 4u || action == 5u) {
        const float pitchCorrection = 0.0f;
        return airborne
            ? clamp(
                  0.20f * sin(state.commandAndPhase.w + 2.62f) +
                      pitchCorrection +
                      (action == 4u
                          ? 0.50f * navigationBankCorrection
                          : -0.50f * navigationBankCorrection),
                  -0.65f,
                  0.65f
              )
            : 0.0f;
    }
    if (action == 6u) {
        const float pitchError = navigation
            ? state.threatGeometry.z - navigationPitchTarget
            : straightCruise ? 0.0f
            : state.threatGeometry.w;
        return airborne
            ? clamp(
                  -0.20f + 0.60f * heightError - 2.00f * pitchError,
                  -0.80f,
                  0.60f
              )
            : 0.0f;
    }
    if (action >= 7u && action <= 12u) {
        const uint legAction = action - 7u;
        const bool right = legAction >= 3u;
        const uint joint = legAction % 3u;
        if ((state.episode.z == 1u || !straightCruise) &&
            seconds >= 2.0f && seconds < 5.0f) {
            const float phase = kTwoPi * seconds / 0.50f;
            const float swing = right ? -sin(phase) : sin(phase);
            const float lift = max(swing, 0.0f);
            return joint == 0u
                ? -0.014f * swing
                : joint == 1u
                ? 0.018f * lift
                : -0.010f * lift;
        }
        if (seconds >= launchStart &&
            seconds < (straightCruise ? 32.0f : 24.0f)) {
            return joint == 0u ? -0.10f : joint == 1u ? 0.18f : -0.08f;
        }
        if (!straightCruise && seconds >= 24.0f && seconds < landingStart) {
            const float tuck = (landingStart - seconds) /
                max(landingStart - 24.0f, 1.0f);
            return tuck *
                (joint == 0u ? -0.10f : joint == 1u ? 0.18f : -0.08f);
        }
        return 0.0f;
    }
    if ((straightCruise || navigation) &&
        action == 13u && airborne) {
        return clamp(
            16.00f * headingError - 0.50f * yawRate -
                (navigation ? 0.35f * lateralSpeedError : 0.0f),
            -1.00f,
            1.00f
        );
    }
    if (navigation && action == 14u && airborne) {
        return clamp(
            -4.00f * (
                state.threatGeometry.z - navigationPitchTarget
            ) -
                1.00f * state.threatGeometry.w,
            -1.00f,
            1.00f
        );
    }
    if (!straightCruise && action == 13u &&
        seconds >= 9.0f && seconds < 24.0f) {
        return clamp(
            -0.30f * yawRate - 0.12f * lateralSpeedError,
            -0.35f,
            0.35f
        );
    }
    if (!straightCruise && action == 14u && airborne) {
        // The invocation-scoped teacher publishes the same live
        // proportional-and-rate target as the hierarchical supervisor, but
        // this action label passes through the ordinary actuator response
        // filter before reaching the body moment. Compensate for that lag in
        // the teacher gains; this remains a bounded training-time label and
        // neural deployment receives no supervisor authority.
        return clamp(
            -8.00f * state.threatGeometry.w -
                2.00f * state.threatTeacher.x,
            -1.0f,
            1.0f
        );
    }
    return 0.0f;
}

// Frozen v15 low-level action law, version 1. Its accepted-state inputs and
// ordinary actuator filtering are supplied by the caller. It does not read
// route targets, teacher labels, scene geometry, or learned-policy weights.
inline float crowFlightControllerV1Action(
    const uint action,
    const MRTaskStateGPU state,
    const float controlStepSeconds
) {
    float requested = crowFlightControllerV1BaseAction(
        action, state, controlStepSeconds, true
    );
    if (float(state.episode.x) * controlStepSeconds >= 6.0f &&
        (action == 4u || action == 5u)) {
        const float speedBias = clamp(-0.75f - 1.00f * (state.commandExtension.z - max(state.commandAndPhase.x, 0.15f)), -1.0f, -0.25f);
        const float yawDemand = clamp(16.0f * state.commandExtension.x - 0.50f * state.commandExtension.y - 0.35f * state.threatTeacher.x, -1.0f, 1.0f);
        const float phaseOffset = 0.60f * clamp(-yawDemand, 0.0f, 1.0f) * sin(state.commandAndPhase.w);
        requested = clamp(requested + (action == 4u ? speedBias : -speedBias) + phaseOffset, -1.0f, 1.0f);
    }
    if (float(state.episode.x) * controlStepSeconds >= 6.0f && action < 2u) requested = clamp(requested + 0.28f, -1.0f, 1.0f);
    return requested;
}

