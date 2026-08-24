import Foundation
import Darwin
import AppKit

private struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var value = state
        value =
            (value ^ (value >> 30)) &*
            0xbf58476d1ce4e5b9
        value =
            (value ^ (value >> 27)) &*
            0x94d049bb133111eb
        return value ^ (value >> 31)
    }

    mutating func signedUnit() -> Float {
        let value = UInt32(truncatingIfNeeded: next() >> 40)
        return 2 * Float(value) / Float(1 << 24) - 1
    }
}

private func fingerprintWord(
    _ value: UInt64,
    byteCount: Int,
    into hash: inout UInt64
) {
    for byte in 0..<byteCount {
        hash ^= (value >> UInt64(8 * byte)) & 0xff
        hash &*= 1_099_511_628_211
    }
}

private func fingerprint(
    _ batch: MetalRoboPolicyRolloutBatch,
    into hash: inout UInt64
) {
    for sample in 0..<batch.sampleCount {
        let slices: [(values: [Float], width: Int)] = [
            (
                batch.actorObservations,
                batch.actorObservationCount
            ),
            (
                batch.criticObservations,
                batch.criticObservationCount
            ),
            (batch.latents, batch.actionCount),
            (batch.logProbabilities, 1),
            (batch.values, 1),
        ]
        for slice in slices {
            let lower = sample * slice.width
            for value in slice.values[
                lower..<(lower + slice.width)
            ] {
                fingerprintWord(
                    UInt64(value.bitPattern),
                    byteCount: 4,
                    into: &hash
                )
            }
        }
        let transition = batch.transitions[sample]
        for value in [
            transition.reward,
            transition.trackingScore,
            transition.rootHeight,
            transition.tilt,
            transition.taskReward,
            transition.baseReward,
            transition.jointVelocityReward,
            transition.jointAccelerationReward,
            transition.controlReward,
            transition.postureReward,
            transition.energyReward,
            transition.contactReward,
        ] {
            fingerprintWord(
                UInt64(value.bitPattern),
                byteCount: 4,
                into: &hash
            )
        }
        fingerprintWord(
            transition.done ? 1 : 0,
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            transition.timeout ? 1 : 0,
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            transition.physicsError ? 1 : 0,
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            UInt64(transition.terminationReason),
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            transition.policyRevision,
            byteCount: 8,
            into: &hash
        )
        fingerprintWord(
            UInt64(transition.timeoutBootstrapValue.bitPattern),
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            UInt64(transition.episodeTrackingScore.bitPattern),
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            UInt64(transition.difficultyBand),
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            UInt64(transition.terrainLevel),
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            UInt64(transition.impactSequenceIndex),
            byteCount: 4,
            into: &hash
        )
        fingerprintWord(
            UInt64(transition.impactEventFlags),
            byteCount: 4,
            into: &hash
        )
    }
}

private struct Options {
    var environments = 32
    var steps = 48
    var repeats = 20
    var chunk = 8
    var physicsSubsteps = 4
    var velocityIterations = 4
    var finalVelocityIterations = 2
    var surface = MetalRoboLocomotionSurface.terrain
    var unitreeG1Task = MetalRoboUnitreeG1Task.velocity
    var seed: UInt64 = 20_260_731
    var metallib = "build/shaders/MetalRobo.metallib"
    var verbose = false
    var nativePolicy = false
    var zeroActions = false
    var actionStream: String?
    var birdFlowFlapScript = false
    var birdFlowStrokeAmplitude: Float?
    var birdFlowTailPitch: Float?
    var birdFlowPronation: Float?
    var birdFlowPronationWavePhase: Float?
    var birdFlowSweepWaveAmplitude: Float?
    var birdFlowSweepWavePhase: Float?
    var birdFlowWingPulseAmplitude: Float?
    var birdFlowWingPulseTarget: String?
    var birdFlowWingPulseSecondaryAmplitude: Float?
    var birdFlowWingPulseSecondaryTarget: String?
    var birdFlowWingPulseStartStep: Int?
    var birdFlowWingPulseDurationSteps: Int?
    var birdFlowGroundGaitProbe = false
    var scheduledResets = true
    var policyPack: String?
    var basePolicyPack: String?
    var rolloutPack: String?
    var interactionPack: String?
    var interactionClip: String?
    var worldPack: String?
    var taskPack: String?
    var robotActuatorPack: String?
    var sensorProgramPack: String?
    var realityProgramPack: String?
    var urdf: String?
    var srdf: String?
    var dynamicSpheres: [MetalRoboDynamicSphere] = []
    var disableTaskTerminations = false
    var materializeArticulatedContactResponses = false
    var minimumDifficultyBand: Int?
    var maximumDifficultyBand: Int?
    var interactionResetOnly = false
    var interactionStudentAuthority: Float?
    var interactionResetPhaseFraction: Float?
    var interactionResetPhaseProbability: Float?
    var interactionResetMaximumPhase: Float?
    var stateTrace: String?
    var stateTraceEnvironment = 0
    var g1VisualPackDirectory: String?
    var ballVisualPackDirectory: String?
    var visualEnvironmentPack: String?
    var visualObservationConfig: String?
    var inspectionScene: String?
    var inspectionWidth = 640
    var inspectionHeight = 360
    var captureDirectory: String?
    var captureWidth = 480
    var captureHeight = 270
    var captureStride = 1
    var capturePolicyCamera = false
    var birdFlowDove = false
    var birdFlowAmericanCrow = false

    init(arguments: [String]) throws {
        var index = 1
        while index < arguments.count {
            let option = arguments[index]
            func value() throws -> String {
                guard index + 1 < arguments.count else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "\(option) requires a value."
                    )
                }
                return arguments[index + 1]
            }
            switch option {
            case "--envs":
                environments = try Self.integer(value(), option)
                index += 1
            case "--steps":
                steps = try Self.integer(value(), option)
                index += 1
            case "--repeats":
                repeats = try Self.integer(value(), option)
                index += 1
            case "--chunk":
                chunk = try Self.integer(value(), option)
                index += 1
            case "--physics-substeps":
                physicsSubsteps = try Self.integer(value(), option)
                index += 1
            case "--velocity-iterations":
                velocityIterations = try Self.integer(value(), option)
                index += 1
            case "--final-velocity-iterations":
                finalVelocityIterations = try Self.integer(value(), option)
                index += 1
            case "--seed":
                guard let parsed = UInt64(try value()) else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--seed requires an unsigned integer."
                    )
                }
                seed = parsed
                index += 1
            case "--metallib":
                metallib = try value()
                index += 1
            case "--scene":
                switch try value() {
                case "ground":
                    surface = .ground
                case "terrain":
                    surface = .terrain
                default:
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--scene must be ground or terrain."
                    )
                }
                index += 1
            case "--task":
                switch try value() {
                case "velocity":
                    unitreeG1Task = .velocity
                case "disturbance-recovery":
                    unitreeG1Task = .disturbanceRecovery
                case "supine-get-up":
                    unitreeG1Task = .supineGetUpDiscovery
                case "developmental-recovery":
                    unitreeG1Task = .developmentalRecovery
                case "adult-locomotion":
                    unitreeG1Task = .adultLocomotion
                case "g1-legs-locomotion":
                    unitreeG1Task = .g1LegsLocomotion
                case "ball-recovery":
                    unitreeG1Task = .ballDisturbanceRecovery
                case "ball-dodge":
                    unitreeG1Task = .ballDodge
                default:
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--task must be velocity, disturbance-recovery, supine-get-up, developmental-recovery, adult-locomotion, g1-legs-locomotion, ball-recovery, or ball-dodge."
                    )
                }
                index += 1
            case "--verbose":
                verbose = true
            case "--native-policy":
                nativePolicy = true
            case "--zero-actions":
                zeroActions = true
            case "--action-stream":
                actionStream = try value()
                index += 1
            case "--no-scheduled-resets":
                scheduledResets = false
            case "--continue-after-termination":
                disableTaskTerminations = true
            case "--materialize-articulated-contact-responses":
                materializeArticulatedContactResponses = true
            case "--minimum-difficulty-band":
                minimumDifficultyBand = try Self.integer(value(), option)
                index += 1
            case "--maximum-difficulty-band":
                maximumDifficultyBand = try Self.integer(value(), option)
                index += 1
            case "--interaction-reset-only":
                interactionResetOnly = true
            case "--interaction-student-authority":
                guard let parsed = Float(try value()),
                      parsed.isFinite,
                      parsed >= 0,
                      parsed <= 1
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--interaction-student-authority must be in [0, 1]."
                    )
                }
                interactionStudentAuthority = parsed
                index += 1
            case "--interaction-reset-phase-fraction":
                guard let parsed = Float(try value()),
                      parsed.isFinite,
                      parsed >= 0,
                      parsed <= 1
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--interaction-reset-phase-fraction must be in [0, 1]."
                    )
                }
                interactionResetPhaseFraction = parsed
                index += 1
            case "--interaction-reset-phase-probability":
                guard let parsed = Float(try value()),
                      parsed.isFinite,
                      parsed >= 0,
                      parsed <= 1
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--interaction-reset-phase-probability must be in [0, 1]."
                    )
                }
                interactionResetPhaseProbability = parsed
                index += 1
            case "--interaction-reset-maximum-phase":
                guard let parsed = Float(try value()),
                      parsed.isFinite,
                      parsed >= 0,
                      parsed <= 1
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--interaction-reset-maximum-phase must be in [0, 1]."
                    )
                }
                interactionResetMaximumPhase = parsed
                index += 1
            case "--policy-pack":
                policyPack = try value()
                index += 1
            case "--base-policy-pack":
                basePolicyPack = try value()
                index += 1
            case "--rollout-pack":
                rolloutPack = try value()
                index += 1
            case "--interaction-pack":
                interactionPack = try value()
                index += 1
            case "--interaction-clip":
                interactionClip = try value()
                index += 1
            case "--world-pack":
                worldPack = try value()
                index += 1
            case "--task-pack":
                taskPack = try value()
                index += 1
            case "--robot-actuator-pack":
                robotActuatorPack = try value()
                index += 1
            case "--sensor-program-pack":
                sensorProgramPack = try value()
                index += 1
            case "--reality-program-pack":
                realityProgramPack = try value()
                index += 1
            case "--urdf":
                urdf = try value()
                index += 1
            case "--srdf":
                srdf = try value()
                index += 1
            case "--ball":
                let fields = try value().split(separator: ",")
                let values = fields.compactMap(Float.init)
                guard values.count == 8 || values.count == 9 else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--ball requires x,y,z,vx,vy,vz,radius,mass[,launch_step]."
                    )
                }
                guard values.allSatisfy(\.isFinite),
                      values[6] > 0,
                      values[7] > 0,
                      values.count == 8 ||
                        (values[8] >= 0 &&
                         values[8] <= Float(0x00ff_ffff) &&
                         values[8].rounded() == values[8])
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--ball values must be finite with positive radius and mass."
                    )
                }
                dynamicSpheres.append(
                    MetalRoboDynamicSphere(
                        position: SIMD3(values[0], values[1], values[2]),
                        linearVelocity: SIMD3(
                            values[3], values[4], values[5]
                        ),
                        radius: values[6],
                        mass: values[7],
                        launchStep:
                            values.count == 9
                            ? UInt32(values[8])
                            : 0
                    )
                )
                index += 1
            case "--state-trace":
                stateTrace = try value()
                index += 1
            case "--state-trace-environment":
                stateTraceEnvironment = try Self.integer(value(), option)
                index += 1
            case "--g1-visual-pack-dir":
                g1VisualPackDirectory = try value()
                index += 1
            case "--ball-visual-pack-dir":
                ballVisualPackDirectory = try value()
                index += 1
            case "--visual-environment-pack":
                visualEnvironmentPack = try value()
                index += 1
            case "--visual-observation-config":
                visualObservationConfig = try value()
                index += 1
            case "--inspect-scene":
                inspectionScene = try value()
                index += 1
            case "--inspect-width":
                inspectionWidth = try Self.integer(value(), option)
                index += 1
            case "--inspect-height":
                inspectionHeight = try Self.integer(value(), option)
                index += 1
            case "--capture-dir":
                captureDirectory = try value()
                index += 1
            case "--capture-width":
                captureWidth = try Self.integer(value(), option)
                index += 1
            case "--capture-height":
                captureHeight = try Self.integer(value(), option)
                index += 1
            case "--capture-stride":
                captureStride = try Self.integer(value(), option)
                index += 1
            case "--capture-policy-camera":
                capturePolicyCamera = true
            case "--birdflow-dove":
                birdFlowDove = true
            case "--birdflow-american-crow":
                birdFlowAmericanCrow = true
            case "--birdflow-flap-script":
                birdFlowFlapScript = true
            case "--birdflow-stroke-amplitude":
                guard let amplitude = Float(try value()), amplitude.isFinite,
                      amplitude >= -1, amplitude <= 1
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--birdflow-stroke-amplitude requires a finite value in [-1, 1]."
                    )
                }
                birdFlowStrokeAmplitude = amplitude
                index += 1
            case "--birdflow-tail-pitch":
                guard let pitch = Float(try value()), pitch.isFinite,
                      pitch >= -1, pitch <= 1
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--birdflow-tail-pitch requires a finite value in [-1, 1]."
                    )
                }
                birdFlowTailPitch = pitch
                index += 1
            case "--birdflow-pronation":
                guard let pronation = Float(try value()), pronation.isFinite,
                      pronation >= -1, pronation <= 1
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--birdflow-pronation requires a finite value in [-1, 1]."
                    )
                }
                birdFlowPronation = pronation
                index += 1
            case "--birdflow-pronation-wave-phase":
                guard let phase = Float(try value()), phase.isFinite,
                      phase >= -Float.pi, phase <= Float.pi
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--birdflow-pronation-wave-phase requires radians in [-pi, pi]."
                    )
                }
                birdFlowPronationWavePhase = phase
                index += 1
            case "--birdflow-sweep-wave-amplitude":
                guard let amplitude = Float(try value()), amplitude.isFinite,
                      amplitude >= -1, amplitude <= 1
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--birdflow-sweep-wave-amplitude requires a finite value in [-1, 1]."
                    )
                }
                birdFlowSweepWaveAmplitude = amplitude
                index += 1
            case "--birdflow-sweep-wave-phase":
                guard let phase = Float(try value()), phase.isFinite,
                      phase >= -Float.pi, phase <= Float.pi
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--birdflow-sweep-wave-phase requires radians in [-pi, pi]."
                    )
                }
                birdFlowSweepWavePhase = phase
                index += 1
            case "--birdflow-wing-pulse-amplitude":
                guard let amplitude = Float(try value()), amplitude.isFinite,
                      amplitude >= -1, amplitude <= 1
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--birdflow-wing-pulse-amplitude requires a finite value in [-1, 1]."
                    )
                }
                birdFlowWingPulseAmplitude = amplitude
                index += 1
            case "--birdflow-wing-pulse-target":
                let target = try value()
                guard target == "wings" || target == "left-wing" ||
                      target == "right-wing" || target == "sweep" ||
                      target == "pronation" || target == "tail"
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--birdflow-wing-pulse-target requires wings, left-wing, right-wing, sweep, pronation, or tail."
                    )
                }
                birdFlowWingPulseTarget = target
                index += 1
            case "--birdflow-wing-pulse-secondary-amplitude":
                guard let amplitude = Float(try value()), amplitude.isFinite,
                      amplitude >= -1, amplitude <= 1
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--birdflow-wing-pulse-secondary-amplitude requires a finite value in [-1, 1]."
                    )
                }
                birdFlowWingPulseSecondaryAmplitude = amplitude
                index += 1
            case "--birdflow-wing-pulse-secondary-target":
                let target = try value()
                guard target == "wings" || target == "left-wing" ||
                      target == "right-wing" || target == "sweep" ||
                      target == "pronation" || target == "tail"
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--birdflow-wing-pulse-secondary-target requires wings, left-wing, right-wing, sweep, pronation, or tail."
                    )
                }
                birdFlowWingPulseSecondaryTarget = target
                index += 1
            case "--birdflow-wing-pulse-start-step":
                birdFlowWingPulseStartStep = try Self.integer(value(), option)
                index += 1
            case "--birdflow-wing-pulse-duration-steps":
                birdFlowWingPulseDurationSteps = try Self.integer(
                    value(), option
                )
                index += 1
            case "--birdflow-ground-gait-probe":
                birdFlowGroundGaitProbe = true
            default:
                throw MetalRoboTaskRolloutError.invalidShape(
                    "Unknown option \(option)."
                )
            }
            index += 1
        }
        guard environments > 0,
              steps > 0,
              repeats > 0,
              chunk > 0,
              physicsSubsteps > 0,
              velocityIterations > 0,
              finalVelocityIterations >= 0
        else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "envs, steps, repeats, chunk, physics substeps, and velocity iterations must be positive; final velocity iterations must be non-negative."
            )
        }
        if (minimumDifficultyBand == nil) !=
            (maximumDifficultyBand == nil) ||
            (minimumDifficultyBand ?? 0) < 0 ||
            (minimumDifficultyBand ?? 0) >
                (maximumDifficultyBand ?? 0)
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "difficulty-band overrides require an ordered non-negative minimum and maximum."
            )
        }
        if nativePolicy && policyPack != nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--native-policy and --policy-pack are mutually exclusive."
            )
        }
        if zeroActions && (nativePolicy || policyPack != nil) {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--zero-actions cannot be combined with a compiled policy."
            )
        }
        if actionStream != nil &&
            (zeroActions || nativePolicy || policyPack != nil)
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--action-stream cannot be combined with another action source."
            )
        }
        if birdFlowDove && birdFlowAmericanCrow {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--birdflow-dove and --birdflow-american-crow are mutually exclusive."
            )
        }
        let hasBirdFlowWingPulse =
            birdFlowWingPulseAmplitude != nil ||
            birdFlowWingPulseTarget != nil ||
            birdFlowWingPulseSecondaryAmplitude != nil ||
            birdFlowWingPulseSecondaryTarget != nil ||
            birdFlowWingPulseStartStep != nil ||
            birdFlowWingPulseDurationSteps != nil
        if hasBirdFlowWingPulse &&
            (birdFlowWingPulseAmplitude == nil ||
             birdFlowWingPulseStartStep == nil ||
             birdFlowWingPulseDurationSteps == nil ||
             birdFlowWingPulseStartStep! < 0 ||
             birdFlowWingPulseDurationSteps! <= 0)
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "BirdFlow wing pulse requires amplitude, non-negative start step, and positive duration."
            )
        }
        if (birdFlowWingPulseSecondaryAmplitude == nil) !=
            (birdFlowWingPulseSecondaryTarget == nil) {
            throw MetalRoboTaskRolloutError.invalidShape(
                "BirdFlow secondary wing pulse requires both target and amplitude."
            )
        }
        if let secondaryTarget = birdFlowWingPulseSecondaryTarget {
            let primaryTarget = birdFlowWingPulseTarget ?? "wings"
            if birdFlowWingPulseTargetsOverlap(primaryTarget, secondaryTarget) {
                throw MetalRoboTaskRolloutError.invalidShape(
                    "BirdFlow primary and secondary pulse targets must not overlap."
                )
            }
        }
        if (birdFlowFlapScript || birdFlowStrokeAmplitude != nil ||
            birdFlowPronation != nil || birdFlowPronationWavePhase != nil ||
            birdFlowSweepWaveAmplitude != nil ||
            birdFlowSweepWavePhase != nil ||
            hasBirdFlowWingPulse ||
            birdFlowGroundGaitProbe) &&
            (!(birdFlowDove || birdFlowAmericanCrow) ||
                zeroActions || actionStream != nil ||
                nativePolicy || policyPack != nil)
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "BirdFlow qualification actions require one BirdFlow bird source and cannot be combined with another action source."
            )
        }
        if (birdFlowFlapScript && birdFlowStrokeAmplitude != nil) ||
            (hasBirdFlowWingPulse &&
             (birdFlowFlapScript || birdFlowStrokeAmplitude != nil ||
              birdFlowGroundGaitProbe)) ||
            (birdFlowGroundGaitProbe &&
             (birdFlowFlapScript || birdFlowStrokeAmplitude != nil))
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "BirdFlow scripted action sources are mutually exclusive."
            )
        }
        if birdFlowTailPitch != nil && birdFlowStrokeAmplitude == nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--birdflow-tail-pitch requires --birdflow-stroke-amplitude."
            )
        }
        if birdFlowPronation != nil && birdFlowStrokeAmplitude == nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--birdflow-pronation requires --birdflow-stroke-amplitude."
            )
        }
        if birdFlowPronationWavePhase != nil &&
            birdFlowStrokeAmplitude == nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--birdflow-pronation-wave-phase requires --birdflow-stroke-amplitude."
            )
        }
        if birdFlowSweepWaveAmplitude != nil &&
            birdFlowStrokeAmplitude == nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--birdflow-sweep-wave-amplitude requires --birdflow-stroke-amplitude."
            )
        }
        if birdFlowSweepWavePhase != nil &&
            birdFlowStrokeAmplitude == nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--birdflow-sweep-wave-phase requires --birdflow-stroke-amplitude."
            )
        }
        if birdFlowPronation != nil && !birdFlowAmericanCrow {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--birdflow-pronation is available only for --birdflow-american-crow."
            )
        }
        if birdFlowPronationWavePhase != nil && !birdFlowAmericanCrow {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--birdflow-pronation-wave-phase is available only for --birdflow-american-crow."
            )
        }
        if (birdFlowSweepWaveAmplitude != nil ||
            birdFlowSweepWavePhase != nil) && !birdFlowAmericanCrow {
            throw MetalRoboTaskRolloutError.invalidShape(
                "BirdFlow sweep wave is available only for --birdflow-american-crow."
            )
        }
        if hasBirdFlowWingPulse && !birdFlowAmericanCrow {
            throw MetalRoboTaskRolloutError.invalidShape(
                "BirdFlow wing pulse is available only for --birdflow-american-crow."
            )
        }
        if birdFlowPronation != nil && birdFlowPronationWavePhase != nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--birdflow-pronation and --birdflow-pronation-wave-phase are mutually exclusive."
            )
        }
        if (birdFlowSweepWaveAmplitude == nil) !=
            (birdFlowSweepWavePhase == nil) {
            throw MetalRoboTaskRolloutError.invalidShape(
                "BirdFlow sweep wave requires both amplitude and phase."
            )
        }
        if rolloutPack != nil &&
            !nativePolicy && policyPack == nil
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--rollout-pack requires --native-policy or --policy-pack."
            )
        }
        if worldPack != nil && urdf != nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--world-pack and --urdf are mutually exclusive."
            )
        }
        if (interactionPack == nil) != (interactionClip == nil) {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Interaction tracking requires both --interaction-pack and --interaction-clip."
            )
        }
        if let interactionPack,
           interactionPack.isEmpty || interactionClip?.isEmpty != false
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "InteractionPack path and clip identity cannot be empty."
            )
        }
        if interactionResetOnly && interactionPack == nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--interaction-reset-only requires an InteractionPack."
            )
        }
        if interactionStudentAuthority != nil && interactionPack == nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--interaction-student-authority requires an InteractionPack."
            )
        }
        if interactionResetPhaseFraction != nil && interactionPack == nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--interaction-reset-phase-fraction requires an InteractionPack."
            )
        }
        if (interactionResetPhaseProbability != nil ||
            interactionResetMaximumPhase != nil) && interactionPack == nil
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Interaction reset curriculum options require an InteractionPack."
            )
        }
        if interactionPack != nil &&
            (worldPack != nil ||
             (unitreeG1Task != .velocity &&
              unitreeG1Task != .ballDodge &&
              unitreeG1Task != .supineGetUpDiscovery))
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "InteractionPack evaluation supports imported URDF owner packs or bundled G1 velocity, ball-dodge, and supine-get-up mechanics; it cannot be combined with a WorldPack."
            )
        }
        let importing = worldPack != nil || urdf != nil
        let ownerArtifacts = [
            taskPack, robotActuatorPack, sensorProgramPack, realityProgramPack,
        ]
        if importing != ownerArtifacts.allSatisfy({ $0 != nil }) ||
            (!importing && ownerArtifacts.contains(where: { $0 != nil }))
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Imported mechanics require --task-pack, --robot-actuator-pack, --sensor-program-pack, and --reality-program-pack."
            )
        }
        if srdf != nil && urdf == nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--srdf requires --urdf."
            )
        }
        if stateTrace != nil && (repeats != 1 || chunk != 1) {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--state-trace requires --repeats 1 --chunk 1."
            )
        }
        if stateTrace == nil && stateTraceEnvironment != 0 {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--state-trace-environment requires --state-trace."
            )
        }
        if stateTraceEnvironment < 0 ||
            stateTraceEnvironment >= environments {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--state-trace-environment must select an active environment."
            )
        }
        if g1VisualPackDirectory == nil && ballVisualPackDirectory != nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--ball-visual-pack-dir requires --g1-visual-pack-dir."
            )
        }
        if visualObservationConfig != nil &&
            (g1VisualPackDirectory != nil ||
             ballVisualPackDirectory != nil ||
             visualEnvironmentPack != nil)
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--visual-observation-config cannot be combined with legacy visual preset options."
            )
        }
        if g1VisualPackDirectory != nil &&
            ((unitreeG1Task != .ballDisturbanceRecovery &&
              unitreeG1Task != .ballDodge &&
              unitreeG1Task != .supineGetUpDiscovery &&
              unitreeG1Task != .developmentalRecovery) ||
             worldPack != nil || urdf != nil)
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "The bundled visual preset requires --task supine-get-up, developmental-recovery, ball-recovery, or ball-dodge."
            )
        }
        if (unitreeG1Task == .ballDisturbanceRecovery ||
            unitreeG1Task == .ballDodge) &&
            g1VisualPackDirectory != nil &&
            ballVisualPackDirectory == nil
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Ball-task visualization requires --ball-visual-pack-dir."
            )
        }
        if captureDirectory != nil &&
            (environments != 1 || chunk != 1 ||
             (g1VisualPackDirectory == nil &&
              visualObservationConfig == nil) ||
             captureWidth <= 0 || captureHeight <= 0 ||
             captureStride <= 0)
        {
            throw MetalRoboTaskRolloutError.invalidShape(
                "Native capture requires one environment, chunk 1, a visual observation, positive dimensions, and a positive stride."
            )
        }
        if capturePolicyCamera && captureDirectory == nil {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--capture-policy-camera requires --capture-dir."
            )
        }
        if inspectionScene != nil &&
            (inspectionWidth <= 0 || inspectionHeight <= 0) {
            throw MetalRoboTaskRolloutError.invalidShape(
                "--inspect-width and --inspect-height must be positive."
            )
        }
    }

    private static func integer(
        _ value: String,
        _ option: String
    ) throws -> Int {
        guard let parsed = Int(value) else {
            throw MetalRoboTaskRolloutError.invalidShape(
                "\(option) requires an integer."
            )
        }
        return parsed
    }
}

private func makeVisualObservation(
    options: Options
) throws -> MetalRoboTaskVisualObservationConfiguration? {
    if let path = options.visualObservationConfig {
        var observation = try
            MetalRoboTaskVisualObservationConfiguration.loadArtifact(
                at: URL(fileURLWithPath: path)
            )
        if options.captureDirectory != nil {
            observation.captureWidth = UInt32(options.captureWidth)
            observation.captureHeight = UInt32(options.captureHeight)
            observation.capturePolicyCamera =
                options.capturePolicyCamera
        }
        return observation
    }
    guard let directory = options.g1VisualPackDirectory else {
        return nil
    }
    let robotDirectory = URL(fileURLWithPath: directory)
    let environment = options.visualEnvironmentPack.map {
        URL(fileURLWithPath: $0)
    }
    var observation: MetalRoboTaskVisualObservationConfiguration
    if let ballDirectory = options.ballVisualPackDirectory {
        observation = try .unitreeG1BallRecovery(
            robotPackDirectory: robotDirectory,
            ballPackDirectory: URL(fileURLWithPath: ballDirectory),
            environmentPackURL: environment
        )
    } else {
        observation = try .unitreeG1(
            robotPackDirectory: robotDirectory,
            environmentPackURL: environment
        )
    }
    if options.captureDirectory != nil {
        observation.captureWidth = UInt32(options.captureWidth)
        observation.captureHeight = UInt32(options.captureHeight)
        observation.capturePolicyCamera = options.capturePolicyCamera
    }
    return observation
}

private func makeInspectionVisual(
    options: Options
) throws -> MetalRoboTaskVisualObservationConfiguration? {
    guard let path = options.inspectionScene else {
        return nil
    }
    var inspection = try MetalRoboTaskVisualObservationConfiguration.loadArtifact(
        at: URL(fileURLWithPath: path)
    )
    inspection.width = UInt32(options.inspectionWidth)
    inspection.height = UInt32(options.inspectionHeight)
    // Inspection is a presentation path, never a media-export configuration.
    inspection.captureWidth = 0
    inspection.captureHeight = 0
    inspection.capturePolicyCamera = false
    return inspection
}

private func writeCaptureFrame(
    _ frame: (width: Int, height: Int, values: [Float]),
    index: Int,
    directory: URL
) throws {
    var data = Data("P6\n\(frame.width) \(frame.height)\n255\n".utf8)
    var pixels = [UInt8](
        repeating: 0,
        count: frame.width * frame.height * 3
    )
    for pixel in 0..<(frame.width * frame.height) {
        for channel in 0..<3 {
            let linear = max(
                0,
                0.62 * frame.values[pixel * 4 + channel]
            )
            let mapped = min(
                1,
                max(
                    0,
                    linear * (2.51 * linear + 0.03) /
                        (linear * (2.43 * linear + 0.59) + 0.14)
                )
            )
            let display = Foundation.pow(Double(mapped), 1.0 / 2.2)
            pixels[pixel * 3 + channel] = UInt8(
                min(255, max(0, Int(display * 255 + 0.5)))
            )
        }
    }
    data.append(contentsOf: pixels)
    let name = String(format: "frame-%06d.ppm", index)
    try data.write(to: directory.appendingPathComponent(name))
}

private func readActionStream(
    path: String,
    expectedCount: Int
) throws -> [Float] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let byteCount = expectedCount.multipliedReportingOverflow(
        by: MemoryLayout<Float>.stride
    )
    guard expectedCount > 0,
          !byteCount.overflow,
          data.count == byteCount.partialValue
    else {
        throw MetalRoboTaskRolloutError.invalidShape(
            "--action-stream must contain exactly \(expectedCount) little-endian Float32 values."
        )
    }
    var values = [Float](repeating: 0, count: expectedCount)
    let copied = values.withUnsafeMutableBufferPointer { buffer in
        data.copyBytes(to: UnsafeMutableRawBufferPointer(buffer))
    }
    guard copied == data.count,
          values.allSatisfy(\.isFinite)
    else {
        throw MetalRoboTaskRolloutError.invalidShape(
            "--action-stream contains an invalid Float32 value."
        )
    }
    return values
}

private func makeContext(
    options: Options
) throws -> (MetalRoboTaskRolloutContext, String) {
    let visualSensor = try makeVisualObservation(options: options)
    let inspectionVisual = try makeInspectionVisual(options: options)
    let dynamicSpheres = options.dynamicSpheres.isEmpty &&
        options.unitreeG1Task == .ballDodge
        ? MetalRoboDynamicSphere.g1BallDodgeDefaults
        : options.unitreeG1Task == .ballDisturbanceRecovery &&
            options.dynamicSpheres.isEmpty
            ? MetalRoboDynamicSphere.g1BallRecoveryDefaults
            : options.dynamicSpheres
    let configuration = MetalRoboTaskRolloutConfiguration(
        environmentCount: UInt32(options.environments),
        surface: options.surface,
        physicsSubsteps: UInt32(options.physicsSubsteps),
        velocityIterations: UInt32(options.velocityIterations),
        finalVelocityIterations:
            UInt32(options.finalVelocityIterations),
        seed: options.seed,
        dynamicSpheres: dynamicSpheres,
        disableTaskTerminations: options.disableTaskTerminations,
        materializeArticulatedContactResponses:
            options.materializeArticulatedContactResponses,
        difficultyBandRange:
            options.minimumDifficultyBand.map {
                UInt32($0)...UInt32(options.maximumDifficultyBand!)
            },
        interactionReferenceMode: options.interactionResetOnly
            ? .resetOnly
            : .taskDefault,
        interactionStudentAuthority:
            options.interactionStudentAuthority,
        interactionResetPhaseFraction:
            options.interactionResetPhaseFraction,
        interactionResetPhaseProbability:
            options.interactionResetPhaseProbability,
        interactionResetMaximumPhase:
            options.interactionResetMaximumPhase,
        unitreeG1Task: options.unitreeG1Task
    )
    if options.birdFlowDove {
        return (
            try MetalRoboTaskRolloutContext(
                manifest: MetalRoboRunManifest(
                    source: .birdFlowDove,
                    sensorsAndPhysics: configuration,
                    visualSensor: visualSensor,
                    inspectionVisual: inspectionVisual
                ),
                metallibPath: options.metallib
            ),
            "birdflow_deetjen_dove_hybrid"
        )
    }
    if options.birdFlowAmericanCrow {
        return (
            try MetalRoboTaskRolloutContext(
                manifest: MetalRoboRunManifest(
                    source: .birdFlowAmericanCrow,
                    sensorsAndPhysics: configuration,
                    visualSensor: visualSensor,
                    inspectionVisual: inspectionVisual
                ),
                metallibPath: options.metallib
            ),
            "birdflow_american_crow_estimated_hybrid"
        )
    }
    if let interactionPack = options.interactionPack,
       let interactionClip = options.interactionClip,
       options.urdf == nil
    {
        return (
            try MetalRoboTaskRolloutContext(
                manifest: MetalRoboRunManifest(
                    source: .unitreeG1,
                    sensorsAndPhysics: configuration,
                    visualSensor: visualSensor,
                    inspectionVisual: inspectionVisual,
                    teacher: MetalRoboTeacherSource(
                        pack: URL(fileURLWithPath: interactionPack),
                        clipID: interactionClip
                    )
                ),
                metallibPath: options.metallib
            ),
            "bundled_g1_interaction"
        )
    }
    if let worldPack = options.worldPack,
       let taskPack = options.taskPack,
       let actuatorPack = options.robotActuatorPack,
       let sensorPack = options.sensorProgramPack,
       let realityPack = options.realityProgramPack
    {
        return (
            try MetalRoboTaskRolloutContext(
                manifest: MetalRoboRunManifest(
                    source: .worldPack(
                        world: URL(fileURLWithPath: worldPack),
                        taskPack: URL(fileURLWithPath: taskPack),
                        actuatorPack: URL(fileURLWithPath: actuatorPack),
                        sensorPack: URL(fileURLWithPath: sensorPack),
                        realityPack: URL(fileURLWithPath: realityPack)
                    ),
                    sensorsAndPhysics: configuration,
                    visualSensor: visualSensor,
                    inspectionVisual: inspectionVisual
                ),
                metallibPath: options.metallib
            ),
            "world_pack"
        )
    }
    if let urdf = options.urdf,
       let taskPack = options.taskPack,
       let actuatorPack = options.robotActuatorPack,
       let sensorPack = options.sensorProgramPack,
       let realityPack = options.realityProgramPack
    {
        return (
            try MetalRoboTaskRolloutContext(
                manifest: MetalRoboRunManifest(
                    source: .importedURDF(
                        urdf: URL(fileURLWithPath: urdf),
                        srdf: options.srdf.map {
                            URL(fileURLWithPath: $0)
                        },
                        taskPack: URL(fileURLWithPath: taskPack),
                        actuatorPack: URL(fileURLWithPath: actuatorPack),
                        sensorPack: URL(fileURLWithPath: sensorPack),
                        realityPack: URL(fileURLWithPath: realityPack)
                    ),
                    sensorsAndPhysics: configuration,
                    visualSensor: visualSensor,
                    inspectionVisual: inspectionVisual,
                    teacher: options.interactionPack.flatMap { pack in
                        options.interactionClip.map {
                            MetalRoboTeacherSource(
                                pack: URL(fileURLWithPath: pack),
                                clipID: $0
                            )
                        }
                    }
                ),
                metallibPath: options.metallib
            ),
            "urdf"
        )
    }
    return (
        try MetalRoboTaskRolloutContext(
            manifest: MetalRoboRunManifest(
                source: .unitreeG1,
                sensorsAndPhysics: configuration,
                visualSensor: visualSensor,
                inspectionVisual: inspectionVisual
            ),
            metallibPath: options.metallib
        ),
        "bundled_g1"
    )
}

private func actions(
    startStep: Int,
    stepCount: Int,
    environmentCount: Int,
    actionCount: Int,
    generator: inout SplitMix64
) -> [Float] {
    var result = [Float](
        repeating: 0,
        count: stepCount * environmentCount * actionCount
    )
    for step in 0..<stepCount {
        let globalStep = startStep + step
        for environment in 0..<environmentCount {
            for joint in 0..<actionCount {
                let phase =
                    0.071 * Double(globalStep) +
                    0.113 * Double(environment) +
                    0.173 * Double(joint)
                let value =
                    0.28 * Float(sin(phase)) +
                    0.04 * generator.signedUnit()
                let index =
                    (step * environmentCount + environment) *
                    actionCount + joint
                result[index] = min(max(value, -1), 1)
            }
        }
    }
    return result
}

// A deterministic full-stroke qualification input, not a policy or a flight
// result. It proves that the robot-owned wingbeat reaches the device-side
// aerodynamic coupling through the same amplitude-modulation action contract.
private func birdFlowFlapActions(
    startStep: Int,
    stepCount: Int,
    environmentCount: Int,
    actionCount: Int
) -> [Float] {
    precondition(actionCount >= 2)
    var result = [Float](
        repeating: 0,
        count: stepCount * environmentCount * actionCount
    )
    for step in 0..<stepCount {
        for environment in 0..<environmentCount {
            let base = (step * environmentCount + environment) * actionCount
            result[base] = 1.0
            result[base + 1] = 1.0
        }
    }
    return result
}

private func birdFlowStrokeActions(
    amplitude: Float,
    tailPitch: Float,
    pronation: Float?,
    pronationWavePhase: Float?,
    sweepWaveAmplitude: Float?,
    sweepWavePhase: Float?,
    startStep: Int,
    stepCount: Int,
    environmentCount: Int,
    actionCount: Int
) -> [Float] {
    precondition(actionCount == 10 || actionCount == 12 || actionCount == 14)
    precondition(pronation == nil || actionCount == 12 || actionCount == 14)
    precondition(pronationWavePhase == nil || actionCount == 12 || actionCount == 14)
    precondition(sweepWaveAmplitude == nil || actionCount == 14)
    precondition(sweepWavePhase == nil || actionCount == 14)
    precondition((sweepWaveAmplitude == nil) == (sweepWavePhase == nil))
    var result = [Float](
        repeating: 0,
        count: stepCount * environmentCount * actionCount
    )
    for step in 0..<stepCount {
        for environment in 0..<environmentCount {
            let base = (step * environmentCount + environment) * actionCount
            // The current crow inserts bilateral sweep and distal-wing
            // pronation after the first two flap lanes. Equal command signs
            // are mirror-symmetric because the physical axes are mirrored in
            // its pack. Leave sweep neutral in this pronation-specific probe.
            // Leave all leg actions neutral during this calibration.
            result[base] = amplitude
            result[base + 1] = amplitude
            if actionCount == 12 || actionCount == 14 {
                let sweepAction: Float
                if let sweepWaveAmplitude, let sweepWavePhase {
                    sweepAction = sweepWaveAmplitude * sin(
                        2.0 * Float.pi *
                            Float(startStep + step) * 0.020 * 4.6 +
                            sweepWavePhase
                    )
                } else {
                    sweepAction = 0
                }
                if actionCount == 14 {
                    // Qualification-only phase sweep through the two real
                    // ABA position drives. It is not a deployed carrier or
                    // a bird-kinematics reconstruction.
                    result[base + 2] = sweepAction
                    result[base + 3] = sweepAction
                }
                // This is a deterministic response probe, not the deployed
                // controller. It exercises the same normalized position
                // bindings as a learned policy while sweeping feathering
                // phase against the 4.6 Hz authored wingbeat.
                let wave = pronationWavePhase.map { phase in
                    sin(2.0 * Float.pi *
                        Float(startStep + step) * 0.020 * 4.6 + phase)
                }
                let pronationAction = pronation ?? wave ?? 0
                let pronationOffset = actionCount == 14 ? 4 : 2
                result[base + pronationOffset] = pronationAction
                result[base + pronationOffset + 1] = pronationAction
                result[base + pronationOffset + 2] = tailPitch
            } else {
                result[base + 2] = tailPitch
            }
        }
    }
    return result
}

// A late-horizon local action-response probe. It applies a selected bilateral
// flight-control residual through ordinary policy-action lanes only over the
// selected interval; the live Metal carrier, aerodynamic solve, and
// articulated ABA joints remain authoritative. This identifies whether the
// policy's bounded residual can regulate an already airborne speed state
// before another learner run is authorized.
private func birdFlowWingPulseTargetsOverlap(
    _ first: String,
    _ second: String
) -> Bool {
    if first == second {
        return true
    }
    if first == "wings" {
        return second == "left-wing" || second == "right-wing"
    }
    if second == "wings" {
        return first == "left-wing" || first == "right-wing"
    }
    return false
}

private func birdFlowWingPulseActionIndices(
    target: String,
    actionCount: Int
) -> [Int] {
    precondition(actionCount == 12 || actionCount == 14)
    switch target {
    case "wings":
        return [0, 1]
    case "left-wing":
        return [0]
    case "right-wing":
        return [1]
    case "sweep":
        precondition(actionCount == 14)
        return [2, 3]
    case "pronation":
        return actionCount == 14 ? [4, 5] : [2, 3]
    case "tail":
        return actionCount == 14 ? [6] : [4]
    default:
        preconditionFailure("Unsupported BirdFlow wing-pulse target.")
    }
}

private func birdFlowWingPulseActions(
    amplitude: Float,
    target: String,
    secondaryAmplitude: Float?,
    secondaryTarget: String?,
    pulseStartStep: Int,
    pulseDurationSteps: Int,
    startStep: Int,
    stepCount: Int,
    environmentCount: Int,
    actionCount: Int
) -> [Float] {
    precondition(actionCount == 12 || actionCount == 14)
    let targetActions = birdFlowWingPulseActionIndices(
        target: target,
        actionCount: actionCount
    )
    let secondaryActions = secondaryTarget.map {
        birdFlowWingPulseActionIndices(target: $0, actionCount: actionCount)
    } ?? []
    precondition(
        Set(targetActions).isDisjoint(with: Set(secondaryActions))
    )
    let pulseEndStep = pulseStartStep + pulseDurationSteps
    var result = [Float](
        repeating: 0,
        count: stepCount * environmentCount * actionCount
    )
    for step in 0..<stepCount {
        let absoluteStep = startStep + step
        guard absoluteStep >= pulseStartStep,
              absoluteStep < pulseEndStep
        else {
            continue
        }
        for environment in 0..<environmentCount {
            let base = (step * environmentCount + environment) * actionCount
            for action in targetActions {
                result[base + action] = amplitude
            }
            if let secondaryAmplitude {
                for action in secondaryActions {
                    result[base + action] = secondaryAmplitude
                }
            }
        }
    }
    return result
}

// A deterministic action-authority calibration, not an imitation target or a
// learned gait.  The low-amplitude alternating leg residuals expose whether
// the imported hybrid can remain supported under a physically plausible
// left/right stepping cadence before the learner is asked to discover it. The
// residual cap is one tenth of the failed first probe and stays inside the
// roughly 0.02-rad leg envelope observed in the held-out learned candidate.
private func birdFlowGroundGaitProbeActions(
    startStep: Int,
    stepCount: Int,
    environmentCount: Int,
    actionCount: Int
) -> [Float] {
    precondition(actionCount == 10 || actionCount == 12 || actionCount == 14)
    var result = [Float](
        repeating: 0,
        count: stepCount * environmentCount * actionCount
    )
    for step in 0..<stepCount {
        let phase = 2.0 * Double.pi *
            Double(startStep + step) * 0.02 / 0.50
        let leftSwing = Float(sin(phase))
        let rightSwing = -leftSwing
        for environment in 0..<environmentCount {
            let base = (step * environmentCount + environment) * actionCount
            let leftLift = max(leftSwing, 0.0)
            let rightLift = max(rightSwing, 0.0)
            // The held-out long-horizon calibration fixed this sign: the
            // opposite hip sweep remained stable but translated backward.
            // Keep lift timing unchanged while reversing only fore/aft swing.
            let legOffset = actionCount == 14 ? 7 : actionCount == 12 ? 5 : 3
            result[base + legOffset] = -0.014 * leftSwing
            result[base + legOffset + 1] = 0.018 * leftLift
            result[base + legOffset + 2] = -0.010 * leftLift
            result[base + legOffset + 3] = -0.014 * rightSwing
            result[base + legOffset + 4] = 0.018 * rightLift
            result[base + legOffset + 5] = -0.010 * rightLift
        }
    }
    return result
}

private func masks(
    startStep: Int,
    stepCount: Int,
    environmentCount: Int,
    pending: inout [Bool]
) -> [UInt32] {
    var result = [UInt32](
        repeating: 0,
        count: stepCount * environmentCount
    )
    for step in 0..<stepCount {
        let globalStep = startStep + step
        for environment in 0..<environmentCount {
            let requested =
                (step == 0 && pending[environment]) ||
                (
                    globalStep > 0 &&
                    (globalStep + 37 * environment) % 257 == 0
                )
            if requested {
                result[step * environmentCount + environment] = 1
                pending[environment] = false
            }
        }
    }
    return result
}

private func publishInspectionFrame(
    from context: MetalRoboTaskRolloutContext,
    to inspector: MetalRoboRunInspectorBridge?,
    loadPolicy: ((URL) throws -> UInt64)? = nil
) throws {
    guard let inspector else {
        return
    }
    if let enabled = inspector.takePendingNativeInspectionEnabled() {
        try context.setInspectionEnabled(enabled)
    }
    func install(_ url: URL) {
        guard let loadPolicy else {
            inspector.reportPolicyStatus("policy reload unavailable")
            return
        }
        do {
            let revision = try loadPolicy(url)
            inspector.reportPolicyInstalled(url, revision: revision)
        } catch {
            // Native policy installation is transactional: a failed load
            // leaves the currently rendered policy intact.
            inspector.reportPolicyRejected()
        }
    }
    let latestRequested = inspector.takeLatestPolicyReloadRequest()
    if let selected = inspector.takePolicySelection() {
        install(selected)
    } else if latestRequested {
        guard let selected = inspector.selectedPolicy else {
            inspector.reportPolicyStatus("select a policy first")
            return
        }
        install(selected)
    }
    guard inspector.acceptsFrames else {
        if inspector.isClosed {
            throw MetalRoboRunInspectorError.closed
        }
        // Pause and window occlusion are applied to the sidecar at this
        // scheduler-owned boundary. Physics and accepted-state submission keep
        // their original cadence.
        return
    }
    guard let frame = try context.acquireInspectionFrame() else {
        return
    }
    inspector.publish(frame: frame, context: context)
}

@main
private enum TaskRolloutMain {
    private static func run(
        options: Options,
        inspector: MetalRoboRunInspectorBridge?
    ) throws {
            let (context, worldSource) =
                try makeContext(options: options)
            let visualObservationEnabled =
                context.visualSceneFingerprint != 0
            if options.stateTrace != nil || options.captureDirectory != nil {
                try context.setStateReadback(true)
            }
            let streamedActions: [Float]? = try options.actionStream.map {
                let layout = context.layout
                let stepEnvironments = options.environments
                    .multipliedReportingOverflow(by: options.steps)
                let samples = stepEnvironments.partialValue
                    .multipliedReportingOverflow(by: options.repeats)
                let elements = samples.partialValue
                    .multipliedReportingOverflow(by: layout.actionCount)
                guard !stepEnvironments.overflow,
                      !samples.overflow,
                      !elements.overflow
                else {
                    throw MetalRoboTaskRolloutError.invalidShape(
                        "--action-stream dimensions overflow Int."
                    )
                }
                return try readActionStream(
                    path: $0,
                    expectedCount: elements.partialValue
                )
            }
            let usesCompiledPolicy =
                options.nativePolicy || options.policyPack != nil
            if let policyPack = options.policyPack {
                try context.loadPolicy(
                    at: URL(fileURLWithPath: policyPack)
                )
                if let basePolicyPack = options.basePolicyPack {
                    try context.loadBasePolicy(
                        at: URL(fileURLWithPath: basePolicyPack)
                    )
                }
            } else if options.nativePolicy {
                let layout = context.layout
                let hiddenCount = 64
                try context.setPolicy(
                    MetalRoboPolicyPack(
                        id: "native_rollout_zero_policy",
                        revision: 7,
                        contractVersion: 1,
                        worldFingerprint: layout.worldFingerprint,
                        taskFingerprint: layout.taskFingerprint,
                        observationFingerprint:
                            layout.observationFingerprint,
                        actionFingerprint: layout.actionFingerprint,
                        layers: [
                            .init(
                                inputCount:
                                    layout.actorObservationCount,
                                outputCount: hiddenCount,
                                activation: .tanh,
                                weights: [Float](
                                    repeating: 0,
                                    count:
                                        layout.actorObservationCount *
                                        hiddenCount
                                ),
                                bias: [Float](
                                    repeating: 0,
                                    count: hiddenCount
                                )
                            ),
                            .init(
                                inputCount: hiddenCount,
                                outputCount: layout.actionCount,
                                activation: .identity,
                                weights: [Float](
                                    repeating: 0,
                                    count:
                                        hiddenCount *
                                        layout.actionCount
                                ),
                                bias: [Float](
                                    repeating: 0,
                                    count: layout.actionCount
                                )
                            ),
                        ],
                        criticLayers: [
                            .init(
                                inputCount:
                                    layout.criticObservationCount,
                                outputCount: hiddenCount,
                                activation: .elu,
                                weights: [Float](
                                    repeating: 0,
                                    count:
                                        layout.criticObservationCount *
                                        hiddenCount
                                ),
                                bias: [Float](
                                    repeating: 0,
                                    count: hiddenCount
                                )
                            ),
                            .init(
                                inputCount: hiddenCount,
                                outputCount: 1,
                                activation: .identity,
                                weights: [Float](
                                    repeating: 0,
                                    count: hiddenCount
                                ),
                                bias: [0]
                            ),
                        ],
                        actionLogStandardDeviation: [Float](
                            repeating: -2,
                            count: layout.actionCount
                        )
                    )
                )
            }
            var installedPolicyRevision: UInt64 = 0
            if usesCompiledPolicy {
                _ = try context.advanceWithPolicy(
                    resetMasks: [UInt32](
                        repeating: 0,
                        count: options.environments
                    ),
                    controlStepCount: 1
                )
                try publishInspectionFrame(
                    from: context,
                    to: inspector,
                    loadPolicy: { url in
                            try context.loadPolicy(
                                at: url
                            )
                            installedPolicyRevision =
                                context.installedPolicyRevision
                            return installedPolicyRevision
                    }
                )
                let revisions = try context.transitions(
                    controlStepCount: 1
                ).map(\.policyRevision)
                guard let revision = revisions.first,
                      revision != 0,
                      revisions.allSatisfy({ $0 == revision })
                else {
                    throw MetalRoboTaskRolloutError.native(
                        "Native policy revision was not attached to transitions."
                    )
                }
                installedPolicyRevision = revision
                let policyLogProbabilities =
                    try context.policyLogProbabilities(
                        controlStepCount: 1
                    )
                let policyValues =
                    try context.policyValues(
                        controlStepCount: 1
                    )
                let policyLatents =
                    try context.policyLatents(
                        controlStepCount: 1
                    )
                guard
                    policyLogProbabilities
                        .allSatisfy(\.isFinite),
                    policyValues.allSatisfy(\.isFinite),
                    policyLatents.allSatisfy(\.isFinite)
                else {
                    throw MetalRoboTaskRolloutError.native(
                        "Native policy produced non-finite PPO records."
                    )
                }
                if options.nativePolicy {
                    let currentLayout = context.layout
                    let logStandardDeviation = -2.0
                    let standardDeviation =
                        Foundation.exp(logStandardDeviation)
                    let halfLogTwoPi =
                        0.5 * Foundation.log(2 * Double.pi)
                    for environment in 0..<options.environments {
                        var expected = 0.0
                        let base =
                            environment *
                            currentLayout.actionCount
                        for action in
                            0..<currentLayout.actionCount
                        {
                            let latent = Double(
                                policyLatents[base + action]
                            )
                            let normal =
                                latent / standardDeviation
                            let gaussian =
                                -0.5 * normal * normal -
                                logStandardDeviation -
                                halfLogTwoPi
                            expected += gaussian
                        }
                        guard abs(
                            expected -
                            Double(
                                policyLogProbabilities[
                                    environment
                                ]
                            )
                        ) < 2.0e-3,
                              abs(policyValues[environment]) <
                                  1.0e-6
                        else {
                            throw MetalRoboTaskRolloutError.native(
                                "Native stochastic-policy record disagrees with the reference distribution."
                            )
                        }
                    }
                }
            } else {
                _ = try context.advance(
                    normalizedActions: [Float](
                        repeating: 0,
                        count:
                            options.environments *
                            context.layout.actionCount
                    ),
                    resetMasks: [UInt32](
                        repeating: 0,
                        count: options.environments
                    ),
                    controlStepCount: 1
                )
                try publishInspectionFrame(
                    from: context,
                    to: inspector,
                    loadPolicy: { url in
                            try context.loadPolicy(
                                at: url
                            )
                            installedPolicyRevision =
                                context.installedPolicyRevision
                            return installedPolicyRevision
                    }
                )
            }
            try context.reset(seed: options.seed)
            let baseline = context.layout
            var generator = SplitMix64(seed: options.seed)
            var pending = [Bool](
                repeating: false,
                count: options.environments
            )
            var globalStep = 0
            var totalResets = 0
            var maximumContacts = 0
            var maximumManifolds = 0
            var stageHighWater: [String: Int] = [:]
            var failedSteps = 0
            var policySampleCount = 0
            var transitionCount = 0
            var minimumDifficultyBand = UInt32.max
            var maximumDifficultyBand: UInt32 = 0
            var terminationCount = 0
            var rewardSum = 0.0
            var trackingSum = 0.0
            var rootHeightSum = 0.0
            var tiltSum = 0.0
            var maximumRootHeight = 0.0
            var maximumTilt = 0.0
            var standingStepCount = 0
            var restoredStepCount = 0
            var recoveryBraceStepCount = 0
            var trunkClearStepCount = 0
            var footSupportStepCount = 0
            var supportTransferStepCount = 0
            var recoveryRiseStepCount = 0
            var quietStandStepCount = 0
            var transitionCountByDifficultyBand: [UInt32: Int] = [:]
            var rootHeightSumByDifficultyBand: [UInt32: Double] = [:]
            var tiltSumByDifficultyBand: [UInt32: Double] = [:]
            var recoveryPhaseCountsByDifficultyBand:
                [UInt32: [String: Int]] = [:]
            var taskRewardSum = 0.0
            var baseRewardSum = 0.0
            var jointVelocityRewardSum = 0.0
            var jointAccelerationRewardSum = 0.0
            var controlRewardSum = 0.0
            var postureRewardSum = 0.0
            var energyRewardSum = 0.0
            var contactRewardSum = 0.0
            var dodgeLinkClearanceRewardSum = 0.0
            var dodgeEvasionRewardSum = 0.0
            var dodgeMissRewardSum = 0.0
            var dodgeSafeStillnessRewardSum = 0.0
            var dodgeSafeActionRateRewardSum = 0.0
            var dodgeCBFCorrectionRewardSum = 0.0
            var dodgeCBFBufferRewardSum = 0.0
            var dodgePredictedClearanceRewardSum = 0.0
            let reportsDodgeRewardBreakdown =
                options.worldPack == nil && options.urdf == nil &&
                options.unitreeG1Task == .ballDodge
            var terminationReasonCounts: [String: Int] = [:]
            var timeoutCount = 0
            var terminationCountByEnvironment = [Int](
                repeating: 0,
                count: options.environments
            )
            var transitionCountByEnvironment = [Int](
                repeating: 0,
                count: options.environments
            )
            var trackingSumByEnvironment = [Double](
                repeating: 0.0,
                count: options.environments
            )
            var rootHeightSumByEnvironment = [Double](
                repeating: 0.0,
                count: options.environments
            )
            var minimumRootHeightByEnvironment = [Double](
                repeating: .infinity,
                count: options.environments
            )
            var tiltSumByEnvironment = [Double](
                repeating: 0.0,
                count: options.environments
            )
            var maximumTiltByEnvironment = [Double](
                repeating: 0.0,
                count: options.environments
            )
            var physicsErrorCountByEnvironment = [Int](
                repeating: 0,
                count: options.environments
            )
            var initialForwardPositionByEnvironment = [Double](
                repeating: .nan,
                count: options.environments
            )
            var maximumForwardProgressByEnvironment = [Double](
                repeating: 0.0,
                count: options.environments
            )
            var finalForwardProgressByEnvironment = [Double](
                repeating: 0.0,
                count: options.environments
            )
            var footSupportStepCountByEnvironment = [Int](
                repeating: 0,
                count: options.environments
            )
            var initialPelvisHeightByEnvironment = [Double](
                repeating: .nan,
                count: options.environments
            )
            var minimumPelvisHeightByEnvironment = [Double](
                repeating: .infinity,
                count: options.environments
            )
            var finalPelvisHeightByEnvironment = [Double](
                repeating: .nan,
                count: options.environments
            )
            var initialKneeFlexionByEnvironment = [Double](
                repeating: .nan,
                count: options.environments
            )
            var maximumLeftKneeFlexionByEnvironment = [Double](
                repeating: -.infinity,
                count: options.environments
            )
            var maximumRightKneeFlexionByEnvironment = [Double](
                repeating: -.infinity,
                count: options.environments
            )
            var finalKneeFlexionByEnvironment = [Double](
                repeating: .nan,
                count: options.environments
            )
            var tracedFootSupportStepCount = 0
            var tracedInitialPelvisHeight: Double?
            var tracedMinimumPelvisHeight = Double.infinity
            var tracedFinalPelvisHeight: Double?
            var tracedInitialKneeFlexion: Double?
            var tracedMaximumLeftKneeFlexion = -Double.infinity
            var tracedMaximumRightKneeFlexion = -Double.infinity
            var tracedFinalKneeFlexion: Double?
            let impactSpheres =
                options.unitreeG1Task == .ballDodge &&
                options.dynamicSpheres.isEmpty
                ? MetalRoboDynamicSphere.g1BallDodgeDefaults
                : options.unitreeG1Task == .ballDisturbanceRecovery &&
                options.dynamicSpheres.isEmpty
                ? MetalRoboDynamicSphere.g1BallRecoveryDefaults
                : options.dynamicSpheres
            let supportsG1SquatEvidence =
                options.stateTrace != nil &&
                (options.unitreeG1Task == .supineGetUpDiscovery ||
                 options.unitreeG1Task == .developmentalRecovery) &&
                options.worldPack == nil && options.urdf == nil
            let supportsG1ForwardEvidence =
                options.stateTrace != nil &&
                (options.unitreeG1Task == .velocity ||
                 options.unitreeG1Task == .adultLocomotion ||
                 options.unitreeG1Task == .g1LegsLocomotion) &&
                options.worldPack == nil && options.urdf == nil
            var impactTouches = [Int](
                repeating: 0,
                count: impactSpheres.count
            )
            var impactContacts = impactTouches
            var impactRecoveries = impactTouches
            var impactMisses = impactTouches
            var impactActiveSteps = impactTouches
            var impactPeakTilt = [Double](
                repeating: 0.0,
                count: impactSpheres.count
            )
            var impactMinimumHeight = [Double](
                repeating: .infinity,
                count: impactSpheres.count
            )
            var impactSequenceEnabledSteps = 0
            var policyRolloutFingerprint:
                UInt64 = 1_469_598_103_934_665_603
            let outcomeSchema = context.outcomeSchema
            var outcomeSums = [Double](
                repeating: 0,
                count: outcomeSchema.count
            )
            var outcomeSampleCount = 0
            var collectedPolicyBatches:
                [MetalRoboPolicyRolloutBatch] = []
            var stateTraceLines: [String] = []
            if options.stateTrace != nil {
                let traceLayout = context.layout
                stateTraceLines.append(
                    "# step nq=\(traceLayout.configurationCount) " +
                    "scene_bodies=\(traceLayout.sceneBodyCount) " +
                    "scene_stride=13 timestep=0.02 " +
                    "environment=\(options.stateTraceEnvironment)"
                )
            }
            let captureDirectory = options.captureDirectory.map {
                URL(fileURLWithPath: $0, isDirectory: true)
            }
            if let captureDirectory {
                try FileManager.default.createDirectory(
                    at: captureDirectory,
                    withIntermediateDirectories: true
                )
            }
            let clock = ContinuousClock()
            let start = clock.now

            for repeatIndex in 0..<options.repeats {
                let repeatStart = clock.now
                var completed = 0
                while completed < options.steps {
                    let stepCount = min(
                        options.chunk,
                        options.steps - completed
                    )
                    let resetBatch = options.scheduledResets
                        ? masks(
                            startStep: globalStep,
                            stepCount: stepCount,
                            environmentCount: options.environments,
                            pending: &pending
                        )
                        : []
                    let advance: MetalRoboTaskRolloutAdvance
                    if usesCompiledPolicy {
                        advance = try context.advanceWithPolicy(
                            resetMasks: resetBatch,
                            controlStepCount: stepCount,
                            policyRevision:
                                installedPolicyRevision,
                            evaluateFinalPolicy:
                                options.rolloutPack != nil &&
                                repeatIndex + 1 ==
                                    options.repeats &&
                                completed + stepCount ==
                                    options.steps
                        )
                    } else {
                        let actionBatch: [Float]
                        if let streamedActions {
                            let actionCount = context.layout.actionCount
                            let start = globalStep * options.environments *
                                actionCount
                            let end = start + stepCount *
                                options.environments * actionCount
                            actionBatch = Array(streamedActions[start..<end])
                        } else if options.zeroActions {
                            actionBatch = [Float](
                                repeating: 0,
                                count:
                                    stepCount *
                                    options.environments *
                                    context.layout.actionCount
                            )
                        } else if options.birdFlowFlapScript {
                            actionBatch = birdFlowFlapActions(
                                startStep: globalStep,
                                stepCount: stepCount,
                                environmentCount: options.environments,
                                actionCount: context.layout.actionCount
                            )
                        } else if let amplitude = options.birdFlowStrokeAmplitude {
                            actionBatch = birdFlowStrokeActions(
                                amplitude: amplitude,
                                tailPitch: options.birdFlowTailPitch ?? 0,
                                pronation: options.birdFlowPronation,
                                pronationWavePhase:
                                    options.birdFlowPronationWavePhase,
                                sweepWaveAmplitude:
                                    options.birdFlowSweepWaveAmplitude,
                                sweepWavePhase: options.birdFlowSweepWavePhase,
                                startStep: globalStep,
                                stepCount: stepCount,
                                environmentCount: options.environments,
                                actionCount: context.layout.actionCount
                            )
                        } else if let amplitude =
                            options.birdFlowWingPulseAmplitude,
                            let pulseStartStep =
                                options.birdFlowWingPulseStartStep,
                            let pulseDurationSteps =
                                options.birdFlowWingPulseDurationSteps
                        {
                            actionBatch = birdFlowWingPulseActions(
                                amplitude: amplitude,
                                target: options.birdFlowWingPulseTarget ??
                                    "wings",
                                secondaryAmplitude:
                                    options.birdFlowWingPulseSecondaryAmplitude,
                                secondaryTarget:
                                    options.birdFlowWingPulseSecondaryTarget,
                                pulseStartStep: pulseStartStep,
                                pulseDurationSteps: pulseDurationSteps,
                                startStep: globalStep,
                                stepCount: stepCount,
                                environmentCount: options.environments,
                                actionCount: context.layout.actionCount
                            )
                        } else if options.birdFlowGroundGaitProbe {
                            actionBatch = birdFlowGroundGaitProbeActions(
                                startStep: globalStep,
                                stepCount: stepCount,
                                environmentCount: options.environments,
                                actionCount: context.layout.actionCount
                            )
                        } else {
                            actionBatch = actions(
                                startStep: globalStep,
                                stepCount: stepCount,
                                environmentCount:
                                    options.environments,
                                actionCount:
                                    context.layout.actionCount,
                                generator: &generator
                            )
                        }
                        advance = try context.advance(
                            normalizedActions: actionBatch,
                            resetMasks: resetBatch,
                            controlStepCount: stepCount,
                            policyRevision: 1
                        )
                    }
                    try publishInspectionFrame(
                        from: context,
                        to: inspector,
                        loadPolicy: { url in
                                try context.loadPolicy(
                                    at: url
                                )
                                installedPolicyRevision =
                                    context.installedPolicyRevision
                                return installedPolicyRevision
                        }
                    )
                    let statuses = try context.statusCodes(
                        controlStepCount: stepCount
                    )
                    guard statuses.allSatisfy({ $0 == 0 }) else {
                        throw MetalRoboTaskRolloutError.native(
                            "Task rollout returned a nonzero GPU status."
                        )
                    }
                    if let captureDirectory,
                       globalStep % options.captureStride == 0 {
                        try writeCaptureFrame(
                            context.visualRGBA(),
                            index: globalStep,
                            directory: captureDirectory
                        )
                    }
                    let observedTransitions:
                        [MetalRoboTaskTransition]
                    if usesCompiledPolicy {
                        let policyBatch =
                            try context.policyRolloutBatch(
                                controlStepCount: stepCount
                            )
                        guard policyBatch.transitions
                            .allSatisfy({
                                $0.policyRevision ==
                                    installedPolicyRevision
                            })
                        else {
                            throw MetalRoboTaskRolloutError.native(
                                "Policy revision changed inside a collected rollout batch."
                            )
                        }
                        fingerprint(
                            policyBatch,
                            into: &policyRolloutFingerprint
                        )
                        policySampleCount +=
                            policyBatch.sampleCount
                        if options.rolloutPack != nil {
                            collectedPolicyBatches.append(
                                policyBatch
                            )
                        }
                        observedTransitions =
                            policyBatch.transitions
                    } else {
                        observedTransitions =
                            try context.transitions(
                                controlStepCount: stepCount
                            )
                    }
                    let typedOutcomes = try context.outcomeValues(
                        controlStepCount: stepCount
                    )
                    guard typedOutcomes.count ==
                            observedTransitions.count * outcomeSchema.count
                    else {
                        throw MetalRoboTaskRolloutError.native(
                            "Task-authored outcome shape is inconsistent."
                        )
                    }
                    for transitionIndex in observedTransitions.indices {
                        let base = transitionIndex * outcomeSchema.count
                        for outcomeIndex in outcomeSchema.indices {
                            outcomeSums[outcomeIndex] += Double(
                                typedOutcomes[base + outcomeIndex]
                            )
                        }
                    }
                    outcomeSampleCount += observedTransitions.count
                    for (transitionIndex, transition) in
                        observedTransitions.enumerated()
                    {
                        let transitionEnvironment =
                            transitionIndex % options.environments
                        transitionCount += 1
                        transitionCountByEnvironment[
                            transitionEnvironment
                        ] += 1
                        trackingSumByEnvironment[
                            transitionEnvironment
                        ] += Double(transition.trackingScore)
                        rootHeightSumByEnvironment[
                            transitionEnvironment
                        ] += Double(transition.rootHeight)
                        minimumRootHeightByEnvironment[
                            transitionEnvironment
                        ] = min(
                            minimumRootHeightByEnvironment[
                                transitionEnvironment
                            ],
                            Double(transition.rootHeight)
                        )
                        tiltSumByEnvironment[
                            transitionEnvironment
                        ] += Double(transition.tilt)
                        maximumTiltByEnvironment[
                            transitionEnvironment
                        ] = max(
                            maximumTiltByEnvironment[
                                transitionEnvironment
                            ],
                            Double(transition.tilt)
                        )
                        if transition.physicsError {
                            physicsErrorCountByEnvironment[
                                transitionEnvironment
                            ] += 1
                        }
                        let difficultyBand = transition.difficultyBand
                        transitionCountByDifficultyBand[
                            difficultyBand,
                            default: 0
                        ] += 1
                        rootHeightSumByDifficultyBand[
                            difficultyBand,
                            default: 0
                        ] += Double(transition.rootHeight)
                        tiltSumByDifficultyBand[
                            difficultyBand,
                            default: 0
                        ] += Double(transition.tilt)
                        minimumDifficultyBand = min(
                            minimumDifficultyBand,
                            transition.difficultyBand
                        )
                        maximumDifficultyBand = max(
                            maximumDifficultyBand,
                            transition.difficultyBand
                        )
                        rewardSum += Double(transition.reward)
                        trackingSum +=
                            Double(transition.trackingScore)
                        rootHeightSum +=
                            Double(transition.rootHeight)
                        tiltSum += Double(transition.tilt)
                        maximumRootHeight = max(
                            maximumRootHeight,
                            Double(transition.rootHeight)
                        )
                        maximumTilt = max(
                            maximumTilt,
                            Double(transition.tilt)
                        )
                        if transition.impactEventFlags &
                            (UInt32(1) << 31) != 0 {
                            standingStepCount += 1
                            recoveryPhaseCountsByDifficultyBand[
                                difficultyBand,
                                default: [:]
                            ]["standing", default: 0] += 1
                        }
                        if transition.impactEventFlags &
                            (UInt32(1) << 30) != 0 {
                            restoredStepCount += 1
                            recoveryPhaseCountsByDifficultyBand[
                                difficultyBand,
                                default: [:]
                            ]["restored", default: 0] += 1
                        }
                        if transition.impactEventFlags &
                            (UInt32(1) << 29) != 0 {
                            recoveryBraceStepCount += 1
                            recoveryPhaseCountsByDifficultyBand[
                                difficultyBand,
                                default: [:]
                            ]["brace", default: 0] += 1
                        }
                        if transition.impactEventFlags &
                            (UInt32(1) << 28) != 0 {
                            trunkClearStepCount += 1
                            recoveryPhaseCountsByDifficultyBand[
                                difficultyBand,
                                default: [:]
                            ]["trunk_clear", default: 0] += 1
                        }
                        if transition.impactEventFlags &
                            (UInt32(1) << 27) != 0 {
                            footSupportStepCount += 1
                            footSupportStepCountByEnvironment[
                                transitionEnvironment
                            ] += 1
                            if transitionEnvironment ==
                                options.stateTraceEnvironment {
                                tracedFootSupportStepCount += 1
                            }
                            recoveryPhaseCountsByDifficultyBand[
                                difficultyBand,
                                default: [:]
                            ]["foot_support", default: 0] += 1
                        }
                        if transition.impactEventFlags &
                            (UInt32(1) << 26) != 0 {
                            supportTransferStepCount += 1
                            recoveryPhaseCountsByDifficultyBand[
                                difficultyBand,
                                default: [:]
                            ]["support_transfer", default: 0] += 1
                        }
                        if transition.impactEventFlags &
                            (UInt32(1) << 25) != 0 {
                            recoveryRiseStepCount += 1
                            recoveryPhaseCountsByDifficultyBand[
                                difficultyBand,
                                default: [:]
                            ]["rise", default: 0] += 1
                        }
                        if transition.impactEventFlags &
                            (UInt32(1) << 24) != 0 {
                            quietStandStepCount += 1
                            recoveryPhaseCountsByDifficultyBand[
                                difficultyBand,
                                default: [:]
                            ]["quiet_stand", default: 0] += 1
                        }
                        taskRewardSum +=
                            Double(transition.taskReward)
                        baseRewardSum +=
                            Double(transition.baseReward)
                        jointVelocityRewardSum +=
                            Double(transition.jointVelocityReward)
                        jointAccelerationRewardSum +=
                            Double(transition.jointAccelerationReward)
                        controlRewardSum +=
                            Double(transition.controlReward)
                        postureRewardSum +=
                            Double(transition.postureReward)
                        energyRewardSum +=
                            Double(transition.energyReward)
                        contactRewardSum +=
                            Double(transition.contactReward)
                        if reportsDodgeRewardBreakdown {
                            dodgeLinkClearanceRewardSum += Double(
                                transition.taskOutcomeChannel0
                            )
                            dodgeEvasionRewardSum += Double(
                                transition.taskOutcomeChannel1
                            )
                            dodgeMissRewardSum += Double(
                                transition.taskOutcomeChannel2
                            )
                            dodgeSafeStillnessRewardSum += Double(
                                transition.taskOutcomeChannel3
                            )
                            dodgeSafeActionRateRewardSum += Double(
                                transition.taskOutcomeChannel4
                            )
                            dodgeCBFCorrectionRewardSum += Double(
                                transition.taskOutcomeChannel5
                            )
                            dodgeCBFBufferRewardSum += Double(
                                transition.taskOutcomeChannel6
                            )
                            dodgePredictedClearanceRewardSum += Double(
                                transition.taskOutcomeChannel7
                            )
                        }
                        if transition.impactSequenceIndex > 0 {
                            let impact = Int(
                                transition.impactSequenceIndex - 1
                            )
                            if impact < impactSpheres.count {
                                impactActiveSteps[impact] += 1
                                impactPeakTilt[impact] = max(
                                    impactPeakTilt[impact],
                                    Double(transition.tilt)
                                )
                                impactMinimumHeight[impact] = min(
                                    impactMinimumHeight[impact],
                                    Double(transition.rootHeight)
                                )
                                if transition.impactEventFlags & 1 != 0 {
                                    impactTouches[impact] += 1
                                }
                                if transition.impactEventFlags & 16 != 0 {
                                    impactContacts[impact] += 1
                                }
                                if transition.impactEventFlags & 2 != 0 {
                                    impactRecoveries[impact] += 1
                                }
                                if transition.impactEventFlags & 4 != 0 {
                                    impactMisses[impact] += 1
                                }
                            }
                        }
                        if transition.impactEventFlags & 8 != 0 {
                            impactSequenceEnabledSteps += 1
                        }
                        if transition.done {
                            terminationCount += 1
                            terminationCountByEnvironment[
                                transitionEnvironment
                            ] += 1
                            let reason = String(
                                transition.terminationReason
                            )
                            terminationReasonCounts[
                                reason,
                                default: 0
                            ] += 1
                            if transition.terminationReason ==
                                MetalRoboTaskTerminationReason.timeout.rawValue
                            {
                                timeoutCount += 1
                            }
                        }
                    }
                    totalResets += advance.hostRequestedResets
                    maximumContacts = max(
                        maximumContacts,
                        advance.maximumActiveContacts
                    )
                    maximumManifolds = max(
                        maximumManifolds,
                        advance.maximumManifolds
                    )
                    for (stage, count) in
                        advance.stageHighWater
                    {
                        stageHighWater[stage] = max(
                            stageHighWater[stage, default: 0],
                            count
                        )
                    }
                    failedSteps += advance.failedEnvironmentSteps
                    if options.stateTrace != nil {
                        let traceLayout = context.layout
                        let environment = options.stateTraceEnvironment
                        let allConfigurations =
                            try context.finalConfiguration()
                        if supportsG1ForwardEvidence &&
                            traceLayout.configurationCount >= 7
                        {
                            for metricEnvironment in
                                0..<options.environments
                            {
                                let metricBase = metricEnvironment *
                                    traceLayout.configurationCount
                                let position = Double(
                                    allConfigurations[metricBase]
                                )
                                if !initialForwardPositionByEnvironment[
                                    metricEnvironment
                                ].isFinite {
                                    initialForwardPositionByEnvironment[
                                        metricEnvironment
                                    ] = position
                                }
                                let origin = initialForwardPositionByEnvironment[
                                    metricEnvironment
                                ]
                                let progress = position - origin
                                maximumForwardProgressByEnvironment[
                                    metricEnvironment
                                ] = max(
                                    maximumForwardProgressByEnvironment[
                                        metricEnvironment
                                    ],
                                    progress
                                )
                                finalForwardProgressByEnvironment[
                                    metricEnvironment
                                ] = progress
                            }
                        }
                        if supportsG1SquatEvidence &&
                            traceLayout.configurationCount > 16 {
                            for metricEnvironment in
                                0..<options.environments
                            {
                                let metricBase = metricEnvironment *
                                    traceLayout.configurationCount
                                let pelvisHeight = Double(
                                    allConfigurations[metricBase + 2]
                                )
                                let leftKnee = Double(
                                    allConfigurations[metricBase + 10]
                                )
                                let rightKnee = Double(
                                    allConfigurations[metricBase + 16]
                                )
                                let kneeFlexion =
                                    0.5 * (leftKnee + rightKnee)
                                if initialPelvisHeightByEnvironment[
                                    metricEnvironment
                                ].isNaN {
                                    initialPelvisHeightByEnvironment[
                                        metricEnvironment
                                    ] = pelvisHeight
                                    initialKneeFlexionByEnvironment[
                                        metricEnvironment
                                    ] = kneeFlexion
                                }
                                minimumPelvisHeightByEnvironment[
                                    metricEnvironment
                                ] = min(
                                    minimumPelvisHeightByEnvironment[
                                        metricEnvironment
                                    ],
                                    pelvisHeight
                                )
                                maximumLeftKneeFlexionByEnvironment[
                                    metricEnvironment
                                ] = max(
                                    maximumLeftKneeFlexionByEnvironment[
                                        metricEnvironment
                                    ],
                                    leftKnee
                                )
                                maximumRightKneeFlexionByEnvironment[
                                    metricEnvironment
                                ] = max(
                                    maximumRightKneeFlexionByEnvironment[
                                        metricEnvironment
                                    ],
                                    rightKnee
                                )
                                finalPelvisHeightByEnvironment[
                                    metricEnvironment
                                ] = pelvisHeight
                                finalKneeFlexionByEnvironment[
                                    metricEnvironment
                                ] = kneeFlexion
                            }
                        }
                        let qStart =
                            environment * traceLayout.configurationCount
                        let qEnd =
                            qStart + traceLayout.configurationCount
                        let configuration = Array(
                            allConfigurations[qStart..<qEnd]
                        )
                        let allSceneStates =
                            try context.finalSceneStates()
                        let sceneStride =
                            traceLayout.sceneBodyCount * 13
                        let sceneStart = environment * sceneStride
                        let sceneEnd = sceneStart + sceneStride
                        let scene = Array(
                            allSceneStates[sceneStart..<sceneEnd]
                        )
                        let values = configuration + scene
                        if configuration.count > 16 {
                            let pelvisHeight =
                                Double(configuration[2])
                            let leftKnee =
                                Double(configuration[10])
                            let rightKnee =
                                Double(configuration[16])
                            let kneeFlexion =
                                0.5 * (leftKnee + rightKnee)
                            if tracedInitialPelvisHeight == nil {
                                tracedInitialPelvisHeight = pelvisHeight
                                tracedInitialKneeFlexion = kneeFlexion
                            }
                            tracedMinimumPelvisHeight = min(
                                tracedMinimumPelvisHeight,
                                pelvisHeight
                            )
                            tracedMaximumLeftKneeFlexion = max(
                                tracedMaximumLeftKneeFlexion,
                                leftKnee
                            )
                            tracedMaximumRightKneeFlexion = max(
                                tracedMaximumRightKneeFlexion,
                                rightKnee
                            )
                            tracedFinalPelvisHeight = pelvisHeight
                            tracedFinalKneeFlexion = kneeFlexion
                        }
                        let payload = values.map {
                            String(format: "%.9g", $0)
                        }.joined(separator: "\t")
                        stateTraceLines.append(
                            "\(globalStep + stepCount)\t\(payload)"
                        )
                    }
                    completed += stepCount
                    globalStep += stepCount
                }
                if options.verbose {
                    let elapsed =
                        repeatStart.duration(to: clock.now)
                    let measuredSubmissions =
                        context.layout.submissionCount -
                        baseline.submissionCount
                    print(
                        "repeat=\(repeatIndex + 1) " +
                        "seconds=\(elapsed.components.seconds) " +
                        "submissions=\(measuredSubmissions)"
                    )
                }
            }

            let collectionEnd = clock.now
            let collectionLayout = context.layout
            if let stateTrace = options.stateTrace {
                try (stateTraceLines.joined(separator: "\n") + "\n")
                    .write(
                        to: URL(fileURLWithPath: stateTrace),
                        atomically: true,
                        encoding: .utf8
                    )
            }
            var rolloutPackBytes = 0
            if let rolloutPack = options.rolloutPack {
                let batch =
                    try MetalRoboPolicyRolloutBatch
                        .concatenating(
                            collectedPolicyBatches
                        )
                let bootstrapValues =
                    try context.bootstrapPolicyValues()
                let outputURL = URL(
                    fileURLWithPath: rolloutPack
                )
                try context.writePolicyRolloutPack(
                    batch,
                    bootstrapValues: bootstrapValues,
                    id: "swift_native_task_rollout",
                    to: outputURL
                )
                let attributes =
                    try FileManager.default.attributesOfItem(
                        atPath: outputURL.path
                    )
                rolloutPackBytes =
                    (attributes[.size] as? NSNumber)?
                        .intValue ?? 0
            }

            let elapsed = start.duration(to: collectionEnd)
            let seconds =
                Double(elapsed.components.seconds) +
                Double(elapsed.components.attoseconds) / 1e18
            let layout = collectionLayout
            let environmentSteps =
                options.environments *
                options.steps *
                options.repeats
            let impactMetrics: [[String: Any]] =
                impactSpheres.indices.map { impact in
                    [
                        "sequence_index": impact + 1,
                        "mass_kg": impactSpheres[impact].mass,
                        "touch_count": impactTouches[impact],
                        "contact_count": impactContacts[impact],
                        "recovery_count": impactRecoveries[impact],
                        "miss_count": impactMisses[impact],
                        "active_steps": impactActiveSteps[impact],
                        "peak_tilt": impactPeakTilt[impact],
                        "minimum_root_height":
                            impactMinimumHeight[impact].isFinite
                            ? impactMinimumHeight[impact]
                            : 0.0,
                    ]
                }
            let anyLinkProjectileHitCount =
                impactContacts.reduce(0, +)
            let cleanProjectileMissCount = impactMisses.reduce(0, +)
            let completedProjectileTrialCount =
                anyLinkProjectileHitCount + cleanProjectileMissCount
            let balanceFailureCount =
                (terminationReasonCounts[
                    String(MetalRoboTaskTerminationReason.height.rawValue)
                ] ?? 0) +
                (terminationReasonCounts[
                    String(MetalRoboTaskTerminationReason.tilt.rawValue)
                ] ?? 0)
            let evidence = try context.evidenceTelemetry()
            let recoveryPhaseRatesByDifficultyBand: [String: Any] =
                Dictionary(
                    uniqueKeysWithValues:
                        transitionCountByDifficultyBand.keys.sorted().map {
                            band in
                            let count = max(
                                transitionCountByDifficultyBand[band] ?? 0,
                                1
                            )
                            let phases =
                                recoveryPhaseCountsByDifficultyBand[band] ?? [:]
                            return (
                                String(band),
                                [
                                    "transition_count": count,
                                    "mean_root_height":
                                        (rootHeightSumByDifficultyBand[band] ?? 0) /
                                        Double(count),
                                    "mean_tilt":
                                        (tiltSumByDifficultyBand[band] ?? 0) /
                                        Double(count),
                                    "brace": Double(phases["brace"] ?? 0) /
                                        Double(count),
                                    "trunk_clear":
                                        Double(phases["trunk_clear"] ?? 0) /
                                        Double(count),
                                    "foot_support":
                                        Double(phases["foot_support"] ?? 0) /
                                        Double(count),
                                    "support_transfer":
                                        Double(phases["support_transfer"] ?? 0) /
                                        Double(count),
                                    "rise": Double(phases["rise"] ?? 0) /
                                        Double(count),
                                    "standing":
                                        Double(phases["standing"] ?? 0) /
                                        Double(count),
                                    "quiet_stand":
                                        Double(phases["quiet_stand"] ?? 0) /
                                        Double(count),
                                    "restored":
                                        Double(phases["restored"] ?? 0) /
                                        Double(count),
                                ] as [String: Any]
                            )
                        }
                )
            let cleanHorizonEnvironmentCount =
                terminationCountByEnvironment.filter { $0 == 0 }.count
            let forwardEvidenceAvailable =
                supportsG1ForwardEvidence &&
                initialForwardPositionByEnvironment.allSatisfy {
                    $0.isFinite
                }
            let meanPeakForwardProgress = forwardEvidenceAvailable
                ? maximumForwardProgressByEnvironment.reduce(0.0, +) /
                    Double(max(options.environments, 1))
                : 0.0
            let meanFinalForwardProgress = forwardEvidenceAvailable
                ? finalForwardProgressByEnvironment.reduce(0.0, +) /
                    Double(max(options.environments, 1))
                : 0.0
            let squatCycleEvidenceByEnvironment: [[String: Any]] =
                (0..<options.environments).map { environment in
                    let available = supportsG1SquatEvidence &&
                        initialPelvisHeightByEnvironment[
                            environment
                        ].isFinite &&
                        minimumPelvisHeightByEnvironment[
                            environment
                        ].isFinite &&
                        finalPelvisHeightByEnvironment[
                            environment
                        ].isFinite &&
                        initialKneeFlexionByEnvironment[
                            environment
                        ].isFinite &&
                        maximumLeftKneeFlexionByEnvironment[
                            environment
                        ].isFinite &&
                        maximumRightKneeFlexionByEnvironment[
                            environment
                        ].isFinite &&
                        finalKneeFlexionByEnvironment[
                            environment
                        ].isFinite
                    let initialPelvis = available
                        ? initialPelvisHeightByEnvironment[environment]
                        : 0.0
                    let minimumPelvis = available
                        ? minimumPelvisHeightByEnvironment[environment]
                        : 0.0
                    let finalPelvis = available
                        ? finalPelvisHeightByEnvironment[environment]
                        : 0.0
                    let initialKnee = available
                        ? initialKneeFlexionByEnvironment[environment]
                        : 0.0
                    let finalKnee = available
                        ? finalKneeFlexionByEnvironment[environment]
                        : 0.0
                    let pelvisExcursion = initialPelvis - minimumPelvis
                    let kneeExcursion = available
                        ? min(
                            maximumLeftKneeFlexionByEnvironment[
                                environment
                            ],
                            maximumRightKneeFlexionByEnvironment[
                                environment
                            ]
                        ) - initialKnee
                        : 0.0
                    let returned = available &&
                        abs(finalPelvis - initialPelvis) <= 0.05 &&
                        abs(finalKnee - initialKnee) <= 0.08
                    return [
                        "environment": environment,
                        "available": available,
                        "termination_count":
                            terminationCountByEnvironment[environment],
                        "pelvis_descent": pelvisExcursion,
                        "bilateral_knee_excursion": kneeExcursion,
                        "bilateral_support_rate": Double(
                            footSupportStepCountByEnvironment[environment]
                        ) / Double(max(options.steps * options.repeats, 1)),
                        "returned_to_initial_pose": returned,
                        "completed": available &&
                            terminationCountByEnvironment[environment] == 0 &&
                            pelvisExcursion >= 0.02 &&
                            kneeExcursion >= 0.10 &&
                            returned,
                    ]
                }
            let completedSquatCycleEnvironments =
                squatCycleEvidenceByEnvironment.compactMap { evidence in
                    evidence["completed"] as? Bool == true
                        ? evidence["environment"] as? Int
                        : nil
                }
            let tracedStateAvailable =
                supportsG1SquatEvidence &&
                tracedInitialPelvisHeight != nil &&
                tracedInitialKneeFlexion != nil &&
                tracedFinalPelvisHeight != nil &&
                tracedFinalKneeFlexion != nil &&
                tracedMinimumPelvisHeight.isFinite &&
                tracedMaximumLeftKneeFlexion.isFinite &&
                tracedMaximumRightKneeFlexion.isFinite
            let initialPelvisHeight = tracedInitialPelvisHeight ?? 0.0
            let finalPelvisHeight = tracedFinalPelvisHeight ?? 0.0
            let initialKneeFlexion = tracedInitialKneeFlexion ?? 0.0
            let finalKneeFlexion = tracedFinalKneeFlexion ?? 0.0
            let pelvisDescent = tracedStateAvailable
                ? initialPelvisHeight - tracedMinimumPelvisHeight
                : 0.0
            let bilateralKneeExcursion = tracedStateAvailable
                ? min(
                    tracedMaximumLeftKneeFlexion,
                    tracedMaximumRightKneeFlexion
                ) - initialKneeFlexion
                : 0.0
            let returnedToInitialPose = tracedStateAvailable &&
                abs(finalPelvisHeight - initialPelvisHeight) <= 0.05 &&
                abs(finalKneeFlexion - initialKneeFlexion) <= 0.08
            let tracedTerminationCount =
                terminationCountByEnvironment[
                    options.stateTraceEnvironment
                ]
            let squatCycleEvidence: [String: Any] = [
                "available": tracedStateAvailable,
                "environment": options.stateTraceEnvironment,
                "termination_count": tracedTerminationCount,
                "initial_pelvis_height": initialPelvisHeight,
                "minimum_pelvis_height": tracedStateAvailable
                    ? tracedMinimumPelvisHeight : 0.0,
                "final_pelvis_height": finalPelvisHeight,
                "pelvis_descent": pelvisDescent,
                "initial_mean_knee_flexion": initialKneeFlexion,
                "maximum_left_knee_flexion": tracedStateAvailable
                    ? tracedMaximumLeftKneeFlexion : 0.0,
                "maximum_right_knee_flexion": tracedStateAvailable
                    ? tracedMaximumRightKneeFlexion : 0.0,
                "bilateral_knee_excursion": bilateralKneeExcursion,
                "final_mean_knee_flexion": finalKneeFlexion,
                "bilateral_support_rate":
                    Double(tracedFootSupportStepCount) /
                    Double(max(options.steps * options.repeats, 1)),
                "returned_to_initial_pose": returnedToInitialPose,
                "completed": tracedStateAvailable &&
                    tracedTerminationCount == 0 &&
                    pelvisDescent >= 0.02 &&
                    bilateralKneeExcursion >= 0.10 &&
                    returnedToInitialPose,
            ]
            var output: [String: Any] = [
                "benchmark": "swift_native_task_rollout",
                "benchmark_seed": NSNumber(value: options.seed),
                "task": context.taskID,
                "run_fingerprint": String(context.layout.runFingerprint),
                "robot_fingerprint": String(
                    context.layout.robotFingerprint
                ),
                "sensor_fingerprint": String(
                    context.layout.sensorFingerprint
                ),
                "reality_fingerprint": String(
                    context.layout.realityFingerprint
                ),
                "teacher_fingerprint": String(
                    context.layout.teacherFingerprint
                ),
                "world_fingerprint": String(layout.worldFingerprint),
                "task_fingerprint": String(layout.taskFingerprint),
                "observation_fingerprint":
                    String(layout.observationFingerprint),
                "action_fingerprint": String(layout.actionFingerprint),
                "actor_observation_count": layout.actorObservationCount,
                "action_count": layout.actionCount,
                "outcomes": Dictionary(uniqueKeysWithValues:
                    outcomeSchema.enumerated().map { index, descriptor in
                        (
                            descriptor.id,
                            [
                                "mean": outcomeSampleCount == 0
                                    ? 0.0
                                    : outcomeSums[index] /
                                        Double(outcomeSampleCount),
                                "unit": descriptor.unit,
                                "direction": descriptor.direction,
                            ] as [String: Any]
                        )
                    }
                ),
                "world_source": worldSource,
                "action_source":
                    options.policyPack != nil
                    ? "policy_pack"
                    : options.nativePolicy
                    ? "compiled_policy"
                    : options.zeroActions
                    ? "zero"
                    : options.actionStream != nil
                    ? "foundation_action_stream"
                    : options.birdFlowFlapScript
                    ? "birdflow_4hz_flap_qualification"
                    : options.birdFlowWingPulseAmplitude != nil
                    ? "birdflow_late_flight_control_pulse_qualification"
                    : options.birdFlowStrokeAmplitude != nil
                    ? options.birdFlowSweepWavePhase != nil
                        ? "birdflow_stroke_amplitude_sweep_wave_qualification"
                        : options.birdFlowPronationWavePhase != nil
                        ? "birdflow_stroke_amplitude_pronation_wave_qualification"
                        : options.birdFlowPronation != nil
                        ? "birdflow_stroke_amplitude_pronation_qualification"
                        : options.birdFlowTailPitch != nil
                        ? "birdflow_stroke_amplitude_tail_qualification"
                        : "birdflow_stroke_amplitude_qualification"
                    : options.birdFlowGroundGaitProbe
                    ? "birdflow_ground_gait_action_probe"
                    : "host_stream",
                "action_stream": options.actionStream ?? "",
                "birdflow_stroke_amplitude": options.birdFlowStrokeAmplitude ?? 0,
                "birdflow_tail_pitch": options.birdFlowTailPitch ?? 0,
                "birdflow_pronation": options.birdFlowPronation ?? 0,
                "birdflow_pronation_wave_phase":
                    options.birdFlowPronationWavePhase ?? 0,
                "birdflow_sweep_wave_amplitude":
                    options.birdFlowSweepWaveAmplitude ?? 0,
                "birdflow_sweep_wave_phase":
                    options.birdFlowSweepWavePhase ?? 0,
                "birdflow_wing_pulse_amplitude":
                    options.birdFlowWingPulseAmplitude ?? 0,
                "birdflow_wing_pulse_target":
                    options.birdFlowWingPulseTarget ?? "wings",
                "birdflow_wing_pulse_secondary_amplitude":
                    options.birdFlowWingPulseSecondaryAmplitude ?? 0,
                "birdflow_wing_pulse_secondary_target":
                    options.birdFlowWingPulseSecondaryTarget ?? "",
                "birdflow_wing_pulse_start_step":
                    options.birdFlowWingPulseStartStep ?? 0,
                "birdflow_wing_pulse_duration_steps":
                    options.birdFlowWingPulseDurationSteps ?? 0,
                "action_carrier": options.birdFlowAmericanCrow
                    ? "stage1_crow_gait_plus_bounded_policy_residual_0.25_band_1;stage2_live_altitude_vertical_rate_and_airspeed_trim_plus_phase_calibrated_pronation_target_amplitude_0.20_phase_2.62_plus_bounded_residual_0.25_wing_sweep_pronation_and_leg_residual_0.25_tail_residual_0.10_band_2"
                    : "none",
                "device": context.deviceName,
                "solver_mode": "temporal_cone",
                "articulated_contact_responses":
                    options.materializeArticulatedContactResponses
                    ? "materialized_inverse_aba"
                    : "streamed_inverse_aba",
                "scene":
                    options.surface == .terrain
                    ? "terrain"
                    : "ground",
                "dynamic_sphere_count":
                    options.unitreeG1Task == .ballDodge &&
                    options.dynamicSpheres.isEmpty
                    ? MetalRoboDynamicSphere.g1BallDodgeDefaults.count
                    : options.unitreeG1Task == .ballDisturbanceRecovery &&
                    options.dynamicSpheres.isEmpty
                    ? MetalRoboDynamicSphere.g1BallRecoveryDefaults.count
                    : options.dynamicSpheres.count,
                "impact_sequence_metrics": impactMetrics,
                "impact_sequence_enabled_steps":
                    impactSequenceEnabledSteps,
                "compiled_impact_event_count":
                    context.impactEventCount,
                "visual_observation":
                    visualObservationEnabled,
                "visual_scene_fingerprint":
                    context.visualSceneFingerprint,
                "visual_observation_config":
                    options.visualObservationConfig ?? "",
                "state_trace": options.stateTrace ?? "",
                "environments": options.environments,
                "steps_per_repeat": options.steps,
                "maximum_episode_steps":
                    layout.maximumEpisodeSteps,
                "repeats": options.repeats,
                "control_steps_per_submission": options.chunk,
                "physics_substeps": options.physicsSubsteps,
                "velocity_iterations": options.velocityIterations,
                "final_velocity_iterations":
                    options.finalVelocityIterations,
                "minimum_sampled_difficulty_band":
                    minimumDifficultyBand,
                "maximum_sampled_difficulty_band":
                    maximumDifficultyBand,
                "physical_evidence": [
                    "control_steps": evidence.controlSteps,
                    "evidence_windows": evidence.evidenceWindows,
                    "pending_completed_episode_count":
                        evidence.pendingCompletedEpisodeCount,
                    "pending_timeout_episode_count":
                        evidence.pendingTimeoutEpisodeCount,
                    "last_completed_episode_count":
                        evidence.lastCompletedEpisodeCount,
                    "last_contact_rate": evidence.lastContactRate,
                    "last_clean_miss_rate":
                        evidence.lastCleanMissRate,
                    "last_balance_failure_rate":
                        evidence.lastBalanceFailureRate,
                    "last_mean_tracking_per_million":
                        evidence.lastMeanTrackingPerMillion,
                ],
                "submission_count":
                    NSNumber(
                        value:
                            layout.submissionCount -
                            baseline.submissionCount
                    ),
                "environment_control_steps_per_second":
                    Double(environmentSteps) / seconds,
                "elapsed_seconds": seconds,
                "gpu_milliseconds":
                    layout.totalGPUMilliseconds -
                    baseline.totalGPUMilliseconds,
                "submission_milliseconds":
                    layout.totalSubmissionMilliseconds -
                    baseline.totalSubmissionMilliseconds,
                "retained_buffer_bytes":
                    layout.retainedBufferBytes,
                "immutable_private_bytes":
                    layout.immutablePrivateBytes,
                "persistent_state_private_bytes":
                    layout.persistentStatePrivateBytes,
                "transient_private_bytes":
                    layout.transientPrivateBytes,
                "shared_boundary_bytes":
                    layout.sharedBoundaryBytes,
                "peak_aliased_bytes":
                    layout.peakAliasedBytes,
                "host_requested_resets": totalResets,
                "scheduled_resets": options.scheduledResets,
                "maximum_active_contacts": maximumContacts,
                "maximum_manifolds": maximumManifolds,
                "stage_high_water": stageHighWater,
                "failed_environment_steps": failedSteps,
                "termination_count": terminationCount,
                "timeout_count": timeoutCount,
                "termination_reason_counts":
                    terminationReasonCounts,
                "termination_count_by_environment":
                    terminationCountByEnvironment,
                "physics_error_count_by_environment":
                    physicsErrorCountByEnvironment,
                "mean_tracking_score_by_environment":
                    trackingSumByEnvironment.enumerated().map {
                        environment, sum in
                        sum / Double(
                            max(
                                transitionCountByEnvironment[
                                    environment
                                ],
                                1
                            )
                        )
                    },
                "mean_root_height_by_environment":
                    rootHeightSumByEnvironment.enumerated().map {
                        environment, sum in
                        sum / Double(
                            max(
                                transitionCountByEnvironment[
                                    environment
                                ],
                                1
                            )
                        )
                    },
                "minimum_root_height_by_environment":
                    minimumRootHeightByEnvironment.map {
                        $0.isFinite ? $0 : 0.0
                    },
                "mean_tilt_by_environment":
                    tiltSumByEnvironment.enumerated().map {
                        environment, sum in
                        sum / Double(
                            max(
                                transitionCountByEnvironment[
                                    environment
                                ],
                                1
                            )
                        )
                    },
                "maximum_tilt_by_environment":
                    maximumTiltByEnvironment,
                "clean_horizon_environment_count":
                    cleanHorizonEnvironmentCount,
                "clean_horizon_environment_rate":
                    Double(cleanHorizonEnvironmentCount) /
                    Double(max(options.environments, 1)),
                "forward_progress_available": forwardEvidenceAvailable,
                "mean_peak_forward_progress_m":
                    meanPeakForwardProgress,
                "mean_final_forward_progress_m":
                    meanFinalForwardProgress,
                "maximum_forward_progress_m":
                    forwardEvidenceAvailable
                    ? maximumForwardProgressByEnvironment.max() ?? 0.0
                    : 0.0,
                "peak_forward_progress_by_environment_m":
                    forwardEvidenceAvailable
                    ? maximumForwardProgressByEnvironment
                    : [],
                "final_forward_progress_by_environment_m":
                    forwardEvidenceAvailable
                    ? finalForwardProgressByEnvironment
                    : [],
                "squat_cycle_evidence": squatCycleEvidence,
                "squat_cycle_completed_environment_count":
                    completedSquatCycleEnvironments.count,
                "squat_cycle_completed_environment_rate":
                    Double(completedSquatCycleEnvironments.count) /
                    Double(max(options.environments, 1)),
                "squat_cycle_completed_environments":
                    completedSquatCycleEnvironments,
                "squat_cycle_evidence_by_environment":
                    squatCycleEvidenceByEnvironment,
                "any_link_projectile_hit_count":
                    anyLinkProjectileHitCount,
                "clean_projectile_miss_count":
                    cleanProjectileMissCount,
                "completed_projectile_trial_count":
                    completedProjectileTrialCount,
                "any_link_dodge_rate":
                    Double(cleanProjectileMissCount) /
                    Double(max(completedProjectileTrialCount, 1)),
                "any_link_projectile_hit_rate":
                    Double(anyLinkProjectileHitCount) /
                    Double(max(completedProjectileTrialCount, 1)),
                "height_or_tilt_termination_count":
                    balanceFailureCount,
                "mean_reward":
                    rewardSum / Double(max(transitionCount, 1)),
                "mean_tracking_score":
                    trackingSum /
                    Double(max(transitionCount, 1)),
                "mean_root_height":
                    rootHeightSum /
                    Double(max(transitionCount, 1)),
                "mean_tilt":
                    tiltSum / Double(max(transitionCount, 1)),
                "maximum_root_height": maximumRootHeight,
                "maximum_tilt": maximumTilt,
                "standing_step_count": standingStepCount,
                "restored_step_count": restoredStepCount,
                "recovery_brace_step_count":
                    recoveryBraceStepCount,
                "trunk_clear_step_count": trunkClearStepCount,
                "foot_support_step_count": footSupportStepCount,
                "support_transfer_step_count":
                    supportTransferStepCount,
                "recovery_rise_step_count": recoveryRiseStepCount,
                "quiet_stand_step_count": quietStandStepCount,
                "recovery_phase_rates": [
                    "brace": Double(recoveryBraceStepCount) /
                        Double(max(transitionCount, 1)),
                    "trunk_clear": Double(trunkClearStepCount) /
                        Double(max(transitionCount, 1)),
                    "foot_support": Double(footSupportStepCount) /
                        Double(max(transitionCount, 1)),
                    "support_transfer":
                        Double(supportTransferStepCount) /
                        Double(max(transitionCount, 1)),
                    "rise": Double(recoveryRiseStepCount) /
                        Double(max(transitionCount, 1)),
                    "quiet_stand": Double(quietStandStepCount) /
                        Double(max(transitionCount, 1)),
                ],
                "recovery_phase_rates_by_difficulty_band":
                    recoveryPhaseRatesByDifficultyBand,
                "mean_task_reward":
                    taskRewardSum /
                    Double(max(transitionCount, 1)),
                "mean_base_reward":
                    baseRewardSum /
                    Double(max(transitionCount, 1)),
                "mean_joint_velocity_reward":
                    jointVelocityRewardSum /
                    Double(max(transitionCount, 1)),
                "mean_joint_acceleration_reward":
                    jointAccelerationRewardSum /
                    Double(max(transitionCount, 1)),
                "mean_control_reward":
                    controlRewardSum /
                    Double(max(transitionCount, 1)),
                "mean_posture_reward":
                    postureRewardSum /
                    Double(max(transitionCount, 1)),
                "mean_energy_reward":
                    energyRewardSum /
                    Double(max(transitionCount, 1)),
                "mean_contact_reward":
                    contactRewardSum /
                    Double(max(transitionCount, 1)),
                "policy_sample_count": policySampleCount,
                "policy_rollout_fingerprint":
                    usesCompiledPolicy
                    ? String(policyRolloutFingerprint)
                    : "",
                "rollout_pack": options.rolloutPack ?? "",
                "rollout_pack_bytes": rolloutPackBytes,
            ]
            if reportsDodgeRewardBreakdown {
                let sampleCount = Double(max(transitionCount, 1))
                output["mean_dodge_link_clearance_reward"] =
                    dodgeLinkClearanceRewardSum / sampleCount
                output["mean_dodge_evasion_reward"] =
                    dodgeEvasionRewardSum / sampleCount
                output["mean_dodge_miss_reward"] =
                    dodgeMissRewardSum / sampleCount
                output["mean_dodge_safe_stillness_reward"] =
                    dodgeSafeStillnessRewardSum / sampleCount
                output["mean_dodge_safe_action_rate_reward"] =
                    dodgeSafeActionRateRewardSum / sampleCount
                output["mean_dodge_cbf_correction_reward"] =
                    dodgeCBFCorrectionRewardSum / sampleCount
                output["mean_dodge_cbf_buffer_reward"] =
                    dodgeCBFBufferRewardSum / sampleCount
                output["mean_dodge_predicted_clearance_reward"] =
                    dodgePredictedClearanceRewardSum / sampleCount
            }
            let data = try JSONSerialization.data(
                withJSONObject: output,
                options: [.prettyPrinted, .sortedKeys]
            )
            print(String(decoding: data, as: UTF8.self))
    }

    static func main() {
        do {
            let options = try Options(arguments: CommandLine.arguments)
            guard options.inspectionScene != nil else {
                try run(options: options, inspector: nil)
                return
            }
            let policyChoices = metalRoboInspectorPolicyCatalog()
            let initialPolicyURL = options.policyPack.map {
                URL(fileURLWithPath: $0)
            }
            let inspector = try MetalRoboRunInspectorBridge.launch(
                canReloadLatestPolicy:
                    initialPolicyURL != nil || !policyChoices.isEmpty,
                policyChoices: policyChoices,
                initialPolicyURL: initialPolicyURL
            )
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try run(options: options, inspector: inspector)
                } catch {
                    if case MetalRoboRunInspectorError.closed = error {
                        // Closing the window ends its preview run cleanly.
                    } else {
                        FileHandle.standardError.write(
                            Data("task_rollout failed: \(error)\n".utf8)
                        )
                    }
                }
                DispatchQueue.main.async {
                    NSApplication.shared.terminate(nil)
                }
            }
            NSApplication.shared.run()
        } catch {
            FileHandle.standardError.write(
                Data("task_rollout failed: \(error)\n".utf8)
            )
            Darwin.exit(1)
        }
    }
}
