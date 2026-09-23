#pragma once

#include "NumiHumanBrainReceptors.hpp"
#include "metalrobo/NumiBrainHumanStandingABI.h"

#include <dlfcn.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <string_view>
#include <type_traits>

namespace numi_human_brain {

// A dynamic Brain participant in the existing native standing transaction.
// It neither constructs a physical world nor submits a Metal command buffer.
class Controller {
public:
    Controller(const std::string& libraryPath, id<MTLDevice> device,
               id<MTLLibrary> nativeLibrary, const std::string& sourceJSON,
               const std::string& programJSON, const std::string& outputDirectory,
               std::uint32_t bodyCount, std::uint32_t headBodyIndex,
               std::uint64_t modelFingerprint, std::span<const ContactBinding> contacts,
               std::span<const MRMujocoMuscleResultGPU> preparedMuscleResults,
               std::uint32_t timestepMicroseconds, std::uint64_t epochMicroseconds,
               std::uint32_t seed)
        : library_(libraryPath), brain_{&library_, nullptr}, device_(device),
          timestepMicroseconds_(timestepMicroseconds), seed_(seed), epochMicroseconds_(epochMicroseconds) {
        require(device != nil && bodyCount == 157u && headBodyIndex < bodyCount &&
                    modelFingerprint != 0u && !sourceJSON.empty() && !outputDirectory.empty() &&
                    timestepMicroseconds != 0u && epochMicroseconds >= timestepMicroseconds,
                "Human Brain controller requires the exact 157-body source and physical clock");
        Fingerprint source;
        source.text("numi.human.brain.source-json.v1"); source.text(sourceJSON);
        sourceJSONFingerprint_ = source.value();
        std::array<char, 2048u> error{};
        brain_.handle = library_.create((__bridge void*)device, sourceJSON.c_str(),
            programJSON.empty() ? nullptr : programJSON.c_str(), outputDirectory.c_str(),
            timestepMicroseconds, epochMicroseconds, seed, error.data(), error.size());
        require(brain_.handle != nullptr, error[0] == '\0'
            ? "NumiBrain standing plugin creation failed" : error.data());
        require(library_.info(brain_.handle, &info_) == 1u &&
                    info_.model_source_fingerprint == modelFingerprint &&
                    info_.locomotor_program_fingerprint != 0u &&
                    info_.baseline_program_fingerprint != 0u && info_.compiled_species_fingerprint != 0u &&
                    info_.sensory_profile_fingerprint != 0u && info_.parameter_version_fingerprint != 0u &&
                    info_.committed_generation == 0u && info_.last_joint_commit_fingerprint == 0u,
                "NumiBrain standing plugin source identity or initial generation disagrees");
        receptors_ = std::make_unique<Receptors>(device, nativeLibrary, bodyCount, headBodyIndex,
            modelFingerprint, contacts, preparedMuscleResults,
            timestepMicroseconds, epochMicroseconds);
        Fingerprint program;
        program.text("numi.human.brain.native-participant.v1");
        program.scalar(sourceJSONFingerprint_);
        program.scalar(info_.model_source_fingerprint);
        program.scalar(info_.locomotor_program_fingerprint);
        program.scalar(info_.baseline_program_fingerprint);
        program.scalar(info_.compiled_species_fingerprint);
        program.scalar(info_.sensory_profile_fingerprint);
        program.scalar(info_.parameter_version_fingerprint);
        program.scalar(receptors_->fingerprint());
        program.scalar(timestepMicroseconds_); program.scalar(epochMicroseconds_); program.scalar(seed_);
        programFingerprint_ = program.value();
    }

    ~Controller() { abandonNeuralCandidate(); }
    Controller(const Controller&) = delete;
    Controller& operator=(const Controller&) = delete;
    Controller(Controller&&) = delete;
    Controller& operator=(Controller&&) = delete;

    [[nodiscard]] metalrobo::MetalNumanXTransactionProgram program() noexcept {
        metalrobo::MetalNumanXTransactionProgram result;
        result.context = this; result.encode = &encode; result.abort = &abort;
        result.fingerprint = programFingerprint_;
        return result;
    }
    [[nodiscard]] const NBHumanStandingInfo& info() const noexcept { return info_; }
    [[nodiscard]] std::uint64_t fingerprint() const noexcept { return programFingerprint_; }
    [[nodiscard]] std::uint64_t sourceJSONFingerprint() const noexcept { return sourceJSONFingerprint_; }
    [[nodiscard]] std::uint64_t receptorFingerprint() const noexcept { return receptors_->fingerprint(); }
    [[nodiscard]] std::uint64_t lastPhysicalStateFingerprint() const noexcept { return physicalFingerprint_; }
    [[nodiscard]] bool failed() const noexcept { return failed_; }
    [[nodiscard]] const char* error() const noexcept { return error_.data(); }

    // Diagnostic only. The native and neural publications have both completed
    // before the caller may read this delivered, shared-memory receptor frame.
    // Validity is printed alongside every scalar so an absent observation is
    // never mistaken for a measured zero force or orientation.
    [[nodiscard]] std::string acceptedSensorAuditLine() const {
        require(!failed_ && phase_ == Phase::ready && !rootOpen_ &&
                    info_.committed_generation == std::uint64_t(step_) + 1u &&
                    physicalFingerprint_ != 0u &&
                    info_.last_joint_commit_fingerprint != 0u,
                "Human Brain sensor audit requires a jointly accepted step");
        const Frame& frame = receptors_->deliveredFrame(
            stepTimestamp(info_.committed_generation));
        require(frame.acceptedGeneration == info_.committed_generation &&
                    frame.acceptedPhysicsFingerprint == physicalFingerprint_ &&
                    frame.receptorTimestampMicroseconds + timestepMicroseconds_ ==
                        frame.deliveryTimestampMicroseconds,
                "Human Brain sensor audit frame is not the accepted delivery");
        const Channel& touch = frame.channels[2u];
        const Channel& spindles = frame.channels[3u];
        const Channel& vestibular = frame.channels[4u];
        require(touch.modality == 3u && touch.receptorCount == 10u &&
                    touch.featureCount == 7u && touch.values != nil &&
                    touch.validity != nil &&
                    touch.values.length >= 70u * sizeof(float) &&
                    touch.validity.length >= 10u * sizeof(std::uint32_t) &&
                    spindles.modality == 4u && spindles.receptorCount == 416u &&
                    spindles.featureCount == MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT &&
                    spindles.values != nil && spindles.validity != nil &&
                    spindles.values.length >= 416u *
                        MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT * sizeof(float) &&
                    spindles.validity.length >= 416u * sizeof(std::uint32_t) &&
                    vestibular.modality == 5u && vestibular.receptorCount == 1u &&
                    vestibular.featureCount == 22u && vestibular.values != nil &&
                    vestibular.validity != nil &&
                    vestibular.values.length >= 22u * sizeof(float) &&
                    vestibular.validity.length >= sizeof(std::uint32_t),
                "Human Brain sensor audit packet dimensions are invalid");
        const auto* touchValues = static_cast<const float*>(touch.values.contents);
        const auto* touchValidity = static_cast<const std::uint32_t*>(touch.validity.contents);
        const auto* spindleValues = static_cast<const float*>(spindles.values.contents);
        const auto* spindleValidity = static_cast<const std::uint32_t*>(spindles.validity.contents);
        const auto* vestibularValues = static_cast<const float*>(vestibular.values.contents);
        const auto* vestibularValidity = static_cast<const std::uint32_t*>(vestibular.validity.contents);
        require(touchValues != nullptr && touchValidity != nullptr &&
                    spindleValues != nullptr && spindleValidity != nullptr &&
                    vestibularValues != nullptr && vestibularValidity != nullptr,
                "Human Brain sensor audit shared buffers are unavailable");
        constexpr std::uint32_t pathValidity =
            (1u << MR_NUMANX_HUMAN_FEATURE_PATH_LENGTH_METRES) |
            (1u << MR_NUMANX_HUMAN_FEATURE_PATH_VELOCITY_METRES_PER_SECOND);
        std::size_t spindleValidFiniteCount = 0u;
        std::size_t spindleUsableCount = 0u;
        for (std::size_t muscle = 0u; muscle < 416u; ++muscle) {
            if ((spindleValidity[muscle] & pathValidity) != pathValidity) continue;
            const std::size_t offset = muscle * MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT;
            const float length = spindleValues[offset + MR_NUMANX_HUMAN_FEATURE_PATH_LENGTH_METRES];
            const float velocity = spindleValues[
                offset + MR_NUMANX_HUMAN_FEATURE_PATH_VELOCITY_METRES_PER_SECOND];
            if (!std::isfinite(length) || !std::isfinite(velocity)) continue;
            ++spindleValidFiniteCount;
            if (length > 0.0f) ++spindleUsableCount;
        }
        std::ostringstream line;
        line.imbue(std::locale::classic());
        line << std::setprecision(std::numeric_limits<float>::max_digits10)
             << "human_brain_sensor_audit=accepted"
             << " step=" << frame.acceptedGeneration
             << " brain_generation=" << info_.committed_generation
             << " receptor_timestamp_us=" << frame.receptorTimestampMicroseconds
             << " delivery_timestamp_us=" << frame.deliveryTimestampMicroseconds
             << " physical_fingerprint=" << frame.acceptedPhysicsFingerprint
             << " joint_commit_fingerprint=" << info_.last_joint_commit_fingerprint
             << " touch_validity=[";
        for (std::size_t index = 0u; index < 10u; ++index)
            line << (index == 0u ? "" : ",") << touchValidity[index];
        line << "] touch_normal_force_n=[";
        for (std::size_t index = 0u; index < 10u; ++index) {
            const float force = touchValues[index * 7u + 4u];
            require((touchValidity[index] & (1u << 4u)) == 0u ||
                        (std::isfinite(force) && force >= 0.0f),
                "Human Brain accepted touch force is nonfinite or negative");
            line << (index == 0u ? "" : ",") << force;
        }
        line << "] head_validity=" << (vestibularValidity[0u] & 0x000f0000u)
             << " head_quaternion_xyzw=[";
        for (std::size_t component = 0u; component < 4u; ++component) {
            const float value = vestibularValues[16u + component];
            require((vestibularValidity[0u] & (1u << (16u + component))) == 0u ||
                        std::isfinite(value),
                "Human Brain accepted head orientation is nonfinite");
            line << (component == 0u ? "" : ",") << value;
        }
        line << "] root_position_validity="
             << (vestibularValidity[0u] & 0x00000007u)
             << " root_position_xyz_m=[";
        for (std::size_t component = 0u; component < 3u; ++component) {
            const float value = vestibularValues[component];
            require((vestibularValidity[0u] & (1u << component)) == 0u ||
                        std::isfinite(value),
                    "Human Brain accepted root position is nonfinite");
            line << (component == 0u ? "" : ",") << value;
        }
        line << "] root_linear_velocity_validity="
             << (vestibularValidity[0u] & 0x00000380u)
             << " root_linear_velocity_xyz_m_s=[";
        for (std::size_t component = 0u; component < 3u; ++component) {
            const float value = vestibularValues[7u + component];
            require((vestibularValidity[0u] & (1u << (7u + component))) == 0u ||
                        std::isfinite(value),
                    "Human Brain accepted root velocity is nonfinite");
            line << (component == 0u ? "" : ",") << value;
        }
        line << "] spindle_valid_finite_count=" << spindleValidFiniteCount
             << " spindle_usable_count=" << spindleUsableCount;
        return line.str();
    }

    // Called immediately after a ONE-step context.run(). A successful native
    // step remains physical evidence even if the later neural completion fails.
    // Such failure halts this participant; it never claims joint acceptance.
    void complete(metalrobo::MetalArticulatedOperatorContext& context,
                  const metalrobo::MetalArticulatedOperatorDiagnostics& diagnostics,
                  const metalrobo::MetalArticulatedOperatorResult& result) {
        if (failed_ || phase_ != Phase::candidateEncoded || !rootOpen_ ||
            !receptors_->canPublish(diagnostics, result, 1u)) {
            fail("Human Brain completion lacks the exact accepted native candidate");
            abandonNeuralCandidate();
            throw std::runtime_error(error_.data());
        }
        try {
            physicalFingerprint_ = physicalFingerprint(result, diagnostics.completedStandSteps);
            // Both pointer flips below are prevalidated before encoding the
            // completion pass. A host-result sentinel cannot mint this receipt.
            require(receptors_->canPublish(diagnostics, result, physicalFingerprint_),
                    "Human Brain accepted receptor preflight failed");
            const Frame& candidate = receptors_->pendingFrame(diagnostics.completedStandSteps);
            require(candidate.deliveryTimestampMicroseconds == stepTimestamp(std::uint64_t(step_) + 1u),
                    "Human Brain accepted receptor delivery clock disagrees");
            double gpuStart = 0.0, gpuEnd = 0.0;
            std::string completionError;
            if (!context.finishStandController(diagnostics, this, &encodeAccepted,
                    gpuStart, gpuEnd, completionError)) {
                if (error_[0] == '\0') fail(completionError.c_str());
                throw std::runtime_error(error_.data());
            }
            require(phase_ == Phase::acceptedEncoded,
                    "Human Brain owner completion did not encode the accepted consequence");
            const Channel& vestibular = candidate.channels[4u];
            require(vestibular.values != nil && vestibular.validity != nil &&
                        vestibular.values.length >= 10u * sizeof(float) &&
                        vestibular.validity.length >= sizeof(std::uint32_t),
                    "Human Brain accepted root coordinate receptor packet is absent");
            const auto* values = static_cast<const float*>(vestibular.values.contents);
            const auto* validity = static_cast<const std::uint32_t*>(
                vestibular.validity.contents);
            require(values != nullptr && validity != nullptr &&
                        (validity[0u] & 0x00000007u) == 0x00000007u &&
                        (validity[0u] & 0x00000380u) == 0x00000380u,
                    "Human Brain accepted root coordinates are not physically valid");
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                require(std::bit_cast<std::uint32_t>(values[axis]) ==
                            std::bit_cast<std::uint32_t>(result.standQ[axis]),
                        "Human Brain root position differs from accepted native state");
                require(std::bit_cast<std::uint32_t>(values[7u + axis]) ==
                            std::bit_cast<std::uint32_t>(result.standV[axis]),
                        "Human Brain root velocity differs from accepted native state");
            }
            std::array<char, 2048u> pluginError{};
            if (library_.publish(brain_.handle, gpuStart, gpuEnd, pluginError.data(), pluginError.size()) != 1u) {
                fail(pluginError[0] == '\0' ? "NumiBrain standing publication failed" : pluginError.data());
                throw std::runtime_error(error_.data());
            }
            rootOpen_ = false;
            NBHumanStandingInfo published{};
            if (library_.info(brain_.handle, &published) != 1u ||
                !sameProgram(info_, published) || published.committed_generation != std::uint64_t(step_) + 1u ||
                published.last_joint_commit_fingerprint == 0u) {
                fail("NumiBrain standing publication returned an incoherent committed generation");
                throw std::runtime_error(error_.data());
            }
            // canPublish was checked above and no receptor mutation occurred
            // during the neural pass. This is the same already-validated flip.
            if (!receptors_->publishAccepted(diagnostics, result, physicalFingerprint_)) {
                fail("Human Brain accepted receptor publication lost its prevalidated candidate");
                throw std::runtime_error(error_.data());
            }
            info_ = published;
            phase_ = Phase::ready; physicalCommand_ = 0u;
        } catch (const std::exception& exception) {
            fail(exception.what()); abandonNeuralCandidate();
            throw std::runtime_error(error_.data());
        } catch (...) {
            fail("Human Brain accepted-consequence completion raised an unknown exception");
            abandonNeuralCandidate();
            throw std::runtime_error(error_.data());
        }
    }

private:
    struct Library {
        void* image = nullptr;
        decltype(&nb_human_standing_create_v1) create = nullptr;
        decltype(&nb_human_standing_encode_motor_v1) motor = nullptr;
        decltype(&nb_human_standing_encode_accepted_v1) accepted = nullptr;
        decltype(&nb_human_standing_publish_v1) publish = nullptr;
        decltype(&nb_human_standing_abort_v1) abort = nullptr;
        decltype(&nb_human_standing_info_v1) info = nullptr;
        decltype(&nb_human_standing_destroy_v1) destroy = nullptr;
        explicit Library(const std::string& path) {
            if (path.empty()) throw std::runtime_error("NumiBrain standing library path is empty");
            image = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (image == nullptr) {
                const char* reason = dlerror();
                throw std::runtime_error(reason == nullptr ? "NumiBrain standing library could not be loaded" : reason);
            }
            try {
                create = symbol<decltype(create)>("nb_human_standing_create_v1");
                motor = symbol<decltype(motor)>("nb_human_standing_encode_motor_v1");
                accepted = symbol<decltype(accepted)>("nb_human_standing_encode_accepted_v1");
                publish = symbol<decltype(publish)>("nb_human_standing_publish_v1");
                abort = symbol<decltype(abort)>("nb_human_standing_abort_v1");
                info = symbol<decltype(info)>("nb_human_standing_info_v1");
                destroy = symbol<decltype(destroy)>("nb_human_standing_destroy_v1");
            } catch (...) { dlclose(image); image = nullptr; throw; }
        }
        ~Library() { if (image != nullptr) dlclose(image); }
        template <typename Function> Function symbol(const char* name) {
            dlerror();
            void* address = dlsym(image, name);
            const char* reason = dlerror();
            if (address == nullptr || reason != nullptr)
                throw std::runtime_error(std::string("NumiBrain standing symbol unavailable: ") + name);
            return reinterpret_cast<Function>(address);
        }
    };
    struct BrainHandle {
        Library* library = nullptr;
        void* handle = nullptr;
        ~BrainHandle() { if (handle != nullptr) library->destroy(handle); }
    };
    struct Fingerprint {
        std::uint64_t state = 14695981039346656037ull;
        void bytes(const void* source, std::size_t count) noexcept {
            const auto* data = static_cast<const unsigned char*>(source);
            for (std::size_t index = 0u; index < count; ++index) {
                state ^= data[index]; state *= 1099511628211ull;
            }
        }
        template <typename T> void scalar(const T& value) noexcept {
            static_assert(std::is_integral_v<T>);
            bytes(&value, sizeof(value));
        }
        void text(std::string_view value) noexcept {
            scalar(static_cast<std::uint64_t>(value.size())); bytes(value.data(), value.size());
        }
        void floats(std::span<const float> values) noexcept {
            scalar(static_cast<std::uint64_t>(values.size()));
            for (float value : values) scalar(std::bit_cast<std::uint32_t>(value));
        }
        void vector(const mr_float4& value) noexcept {
            const std::array<float, 4u> components{value.x, value.y, value.z, value.w};
            floats(components);
        }
        [[nodiscard]] std::uint64_t value() const noexcept {
            return state == 0u ? 14695981039346656037ull : state;
        }
    };
    enum class Phase { ready, motorEncoded, preDynamics, candidateEncoded, acceptedEncoded };
    Library library_;
    BrainHandle brain_;
    id<MTLDevice> device_ = nil;
    std::unique_ptr<Receptors> receptors_;
    NBHumanStandingInfo info_{};
    std::uint32_t timestepMicroseconds_ = 0u, seed_ = 0u, step_ = 0u;
    std::uint64_t epochMicroseconds_ = 0u, sourceJSONFingerprint_ = 0u, programFingerprint_ = 0u;
    std::uint64_t physicalFingerprint_ = 0u;
    std::uintptr_t physicalCommand_ = 0u;
    Phase phase_ = Phase::ready;
    bool rootOpen_ = false, failed_ = false;
    std::array<char, 2048u> error_{};

    static void require(bool condition, const std::string& reason) {
        if (!condition) throw std::runtime_error(reason);
    }
    void fail(const char* reason) noexcept {
        failed_ = true;
        if (error_[0] == '\0') std::snprintf(error_.data(), error_.size(), "%s",
            reason == nullptr || reason[0] == '\0' ? "Human Brain controller failed" : reason);
    }
    void abandonNeuralCandidate() noexcept {
        if (rootOpen_ && brain_.handle != nullptr) {
            std::array<char, 1024u> abortError{};
            if (library_.abort(brain_.handle, abortError.data(), abortError.size()) != 1u)
                fail(abortError[0] == '\0' ? "NumiBrain candidate abort failed" : abortError.data());
        }
        rootOpen_ = false;
        if (receptors_) receptors_->abortCandidate();
        phase_ = Phase::ready; physicalCommand_ = 0u;
    }
    [[nodiscard]] std::uint64_t stepTimestamp(std::uint64_t step) const {
        require(step <= (std::numeric_limits<std::uint64_t>::max() - epochMicroseconds_) / timestepMicroseconds_,
                "Human Brain physical clock overflow");
        return epochMicroseconds_ + step * timestepMicroseconds_;
    }
    static std::array<NBHumanStandingReceptor, 7u> receptorRecords(const Frame& frame) noexcept {
        std::array<NBHumanStandingReceptor, 7u> result{};
        for (std::size_t index = 0u; index < result.size(); ++index) {
            const Channel& channel = frame.channels[index];
            result[index] = {channel.modality, channel.receptorCount, channel.featureCount, 0u,
                frame.receptorTimestampMicroseconds, (__bridge void*)channel.values,
                (__bridge void*)channel.validity};
        }
        return result;
    }
    static bool sameProgram(const NBHumanStandingInfo& a, const NBHumanStandingInfo& b) noexcept {
        return a.model_source_fingerprint == b.model_source_fingerprint &&
            a.locomotor_program_fingerprint == b.locomotor_program_fingerprint &&
            a.baseline_program_fingerprint == b.baseline_program_fingerprint &&
            a.compiled_species_fingerprint == b.compiled_species_fingerprint &&
            a.sensory_profile_fingerprint == b.sensory_profile_fingerprint &&
            a.parameter_version_fingerprint == b.parameter_version_fingerprint;
    }
    std::uint64_t physicalFingerprint(const metalrobo::MetalArticulatedOperatorResult& result,
                                      std::uint32_t completedSteps) const noexcept {
        Fingerprint hash;
        hash.text("numi.human.brain.accepted-native-state.v1");
        hash.scalar(sourceJSONFingerprint_); hash.scalar(programFingerprint_);
        hash.scalar(info_.model_source_fingerprint); hash.scalar(completedSteps);
        hash.scalar(timestepMicroseconds_); hash.scalar(epochMicroseconds_); hash.scalar(seed_);
        hash.floats(result.standQ); hash.floats(result.standV);
        hash.scalar(static_cast<std::uint64_t>(result.mujocoActivationStates.size()));
        for (const auto& muscle : result.mujocoActivationStates) hash.vector(muscle.excitationAndActivation);
        hash.scalar(static_cast<std::uint64_t>(result.standRootTranslations.size()));
        for (const auto& root : result.standRootTranslations) {
            hash.vector(root.reference); hash.vector(root.displacement); hash.vector(root.correction);
        }
        return hash.value();
    }

    static bool encode(void* context, const metalrobo::MetalNumanXTransactionPass& pass) noexcept {
        auto& owner = *static_cast<Controller*>(context);
        @autoreleasepool {
            try {
                using NativePhase = metalrobo::MetalNumanXTransactionPhase;
                if (owner.failed_ || pass.abiVersion != metalrobo::kMetalNumanXTransactionABIVersion ||
                    pass.structSize != sizeof(pass) || pass.programFingerprint != owner.programFingerprint_ ||
                    pass.commandBuffer == nullptr || pass.environmentCount != 1u || pass.bodyCount != 157u ||
                    pass.qCoordinateCount != 129u || pass.dofCount != 128u ||
                    pass.mujocoStateElementCount != 416u || pass.mujocoStateStride != 416u ||
                    pass.mujocoMuscleCount != 416u || pass.stepIndex >= pass.stepCount ||
                    pass.stepIndex != owner.info_.committed_generation ||
                    pass.timestepSeconds != static_cast<float>(owner.timestepMicroseconds_) * 1.0e-6f) {
                    owner.fail("Human Brain native transaction identity, generation or dimensions disagree");
                    return false;
                }
                __unsafe_unretained id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
                if (command.device.registryID != owner.device_.registryID) {
                    owner.fail("Human Brain native transaction changed Metal device"); return false;
                }
                if (pass.phase == NativePhase::beginStep) {
                    if (owner.phase_ != Phase::ready || owner.rootOpen_ ||
                        (pass.accessFlags & metalrobo::MetalNumanXTransactionWriteMujocoExcitation) == 0u ||
                        pass.mujocoStates == nullptr) {
                        owner.fail("Human Brain motor phase was repeated or lacks native excitation access"); return false;
                    }
                    const Frame& frame = owner.receptors_->deliveredFrame(owner.stepTimestamp(pass.stepIndex));
                    const auto records = receptorRecords(frame);
                    __unsafe_unretained id<MTLBuffer> states = (__bridge id<MTLBuffer>)pass.mujocoStates;
                    if (states.device.registryID != owner.device_.registryID ||
                        states.length < 416u * sizeof(MRMujocoMuscleStateGPU)) {
                        owner.fail("Human Brain native muscle-state lease is invalid"); return false;
                    }
                    owner.physicalCommand_ = reinterpret_cast<std::uintptr_t>(pass.commandBuffer);
                    owner.step_ = pass.stepIndex;
                    owner.rootOpen_ = true;
                    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                    if (encoder == nil) { owner.fail("Human Brain motor encoder allocation failed"); return false; }
                    std::array<char, 2048u> error{};
                    const auto success = owner.library_.motor(owner.brain_.handle, (__bridge void*)encoder,
                        pass.stepIndex, records.data(), static_cast<std::uint32_t>(records.size()),
                        pass.mujocoStates, 416u, error.data(), error.size());
                    [encoder endEncoding];
                    if (success != 1u) {
                        owner.fail(error[0] == '\0' ? "NumiBrain motor encoding failed" : error.data()); return false;
                    }
                    owner.phase_ = Phase::motorEncoded;
                    return true;
                }
                if (!owner.rootOpen_ || owner.physicalCommand_ != reinterpret_cast<std::uintptr_t>(pass.commandBuffer) ||
                    owner.step_ != pass.stepIndex) {
                    owner.fail("Human Brain native continuation replaced its open root"); return false;
                }
                if (pass.phase == NativePhase::preDynamics && owner.phase_ == Phase::motorEncoded) {
                    owner.phase_ = Phase::preDynamics; return true;
                }
                if (pass.phase == NativePhase::postDynamics && owner.phase_ == Phase::preDynamics &&
                    owner.receptors_->encodeCandidate(pass)) {
                    owner.phase_ = Phase::candidateEncoded; return true;
                }
                owner.fail("Human Brain receptor encoding or native phase order failed");
                return false;
            } catch (const std::exception& exception) { owner.fail(exception.what()); return false; }
              catch (...) { owner.fail("Human Brain native phase raised an unknown exception"); return false; }
        }
    }
    static void abort(void* context, void* commandBuffer) noexcept {
        auto& owner = *static_cast<Controller*>(context);
        if (owner.physicalCommand_ != 0u && owner.physicalCommand_ != reinterpret_cast<std::uintptr_t>(commandBuffer))
            owner.fail("Human Brain abort named a different native command buffer");
        else owner.fail("native owner abandoned the Human Brain physical command");
        owner.abandonNeuralCandidate();
    }
    static bool encodeAccepted(void* context, void* commandBuffer) noexcept {
        auto& owner = *static_cast<Controller*>(context);
        @autoreleasepool {
            try {
                if (owner.failed_ || !owner.rootOpen_ || owner.phase_ != Phase::candidateEncoded ||
                    owner.physicalFingerprint_ == 0u || commandBuffer == nullptr) {
                    owner.fail("Human Brain accepted encoding lacks a physical receipt"); return false;
                }
                __unsafe_unretained id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)commandBuffer;
                if (command.device.registryID != owner.device_.registryID) {
                    owner.fail("Human Brain accepted completion changed Metal device"); return false;
                }
                const auto records = receptorRecords(owner.receptors_->pendingFrame(owner.step_ + 1u));
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (encoder == nil) { owner.fail("Human Brain accepted encoder allocation failed"); return false; }
                std::array<char, 2048u> error{};
                const auto success = owner.library_.accepted(owner.brain_.handle, (__bridge void*)encoder,
                    owner.physicalFingerprint_, owner.step_ + 1u, records.data(),
                    static_cast<std::uint32_t>(records.size()), error.data(), error.size());
                [encoder endEncoding];
                if (success != 1u) {
                    owner.fail(error[0] == '\0' ? "NumiBrain accepted consequence encoding failed" : error.data()); return false;
                }
                owner.phase_ = Phase::acceptedEncoded;
                return true;
            } catch (const std::exception& exception) { owner.fail(exception.what()); return false; }
              catch (...) { owner.fail("Human Brain accepted encoding raised an unknown exception"); return false; }
        }
    }
};

} // namespace numi_human_brain
