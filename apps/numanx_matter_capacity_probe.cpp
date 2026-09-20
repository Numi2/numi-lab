#include "numi/matter/matter.hpp"
#include "numi/matter/detail.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#ifndef NUMI_MATTER_MATERIAL
#error "NUMI_MATTER_MATERIAL must name a reference Matter material"
#endif

namespace {

static_assert(sizeof(NMFEMHumanAttachmentGPU) == 32u);
static_assert(alignof(NMFEMHumanAttachmentGPU) == 16u);

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

class TemporaryPackage final {
public:
    TemporaryPackage()
        : path_(
              std::filesystem::temp_directory_path() /
              ("numanx-matter-attachment-" + std::to_string(
                  std::chrono::steady_clock::now()
                      .time_since_epoch().count()
              ) + ".nmatterpkg")
          ) {}

    ~TemporaryPackage() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

numi::matter::WorldSource makeWorld(
    const std::uint32_t dofCapacity,
    const std::uint32_t qCapacity,
    const bool includeArticulatedProxy = true
) {
    auto parsed = numi::matter::parseMatterFile(NUMI_MATTER_MATERIAL);
    require(parsed.succeeded(), "reference Matter material did not parse");

    numi::matter::WorldSource source;
    source.frameTimestep = 1.0 / 240.0;
    source.gravity = {0.0, 0.0, 0.0};
    source.articulatedDofCapacity = dofCapacity;
    source.articulatedQCapacity = qCapacity;
    source.materials.push_back(std::move(parsed.material));

    if (includeArticulatedProxy) {
        numi::matter::RigidProxySource articulated;
        articulated.shape = NM_RIGID_PLANE;
        articulated.bodyIndex = 1u;
        articulated.localCenter = {0.0, 0.0, 1.0};
        articulated.articulated = true;
        source.rigidProxies.push_back(articulated);
    }

    numi::matter::ObjectSource tissue;
    tissue.name = "numanx_capacity_tissue";
    tissue.representation = numi::matter::Representation::fem;
    tissue.characteristicLength = 0.01;
    tissue.femNodes = {
        {-0.01, -0.01, 0.01},
        { 0.01, -0.01, 0.01},
        {-0.01,  0.01, 0.01},
        {-0.01, -0.01, 0.03},
    };
    tissue.tetrahedra.push_back({{0u, 1u, 2u, 3u}});
    numi::matter::FEMHumanAttachmentSource attachment;
    attachment.node = 0u;
    attachment.bodyIndex = 12u;
    attachment.stableIdentifier = 0x4e580001u;
    attachment.localPoint = {0.001, -0.002, 0.003};
    tissue.femHumanAttachments.push_back(attachment);
    source.objects.push_back(std::move(tissue));
    return source;
}

void requireRejected(
    numi::matter::WorldSource source,
    const std::string& message
) {
    require(!numi::matter::compileWorld(source).succeeded(), message);
}

void requireRejectedWithDiagnostic(
    numi::matter::WorldSource source,
    const std::string_view expected,
    const std::string& message
) {
    const auto result = numi::matter::compileWorld(source);
    require(
        !result.succeeded() &&
            std::ranges::any_of(
                result.diagnostics,
                [expected](const numi::matter::Diagnostic& diagnostic) {
                    return diagnostic.message.find(expected) !=
                        std::string::npos;
                }
            ),
        message
    );
}

} // namespace

int main() {
    try {
        constexpr std::uint32_t kHumanDofs = 160u;
        constexpr std::uint32_t kLastFittingAttachmentCount =
            std::numeric_limits<std::uint32_t>::max() /
            (3u * kHumanDofs);
        constexpr std::uint32_t kFirstOverflowAttachmentCount =
            kLastFittingAttachmentCount + 1u;
        static_assert(
            numi::matter::detail::
                femHumanAttachmentPointJacobianStrideFits(
                    kLastFittingAttachmentCount, kHumanDofs));
        static_assert(
            !numi::matter::detail::
                femHumanAttachmentPointJacobianStrideFits(
                    kFirstOverflowAttachmentCount, kHumanDofs));
        require(
            numi::matter::detail::
                femHumanAttachmentPointJacobianScalarCount(
                    kLastFittingAttachmentCount, kHumanDofs) <=
                std::numeric_limits<std::uint32_t>::max() &&
            numi::matter::detail::
                femHumanAttachmentPointJacobianScalarCount(
                    kFirstOverflowAttachmentCount, kHumanDofs) >
                std::numeric_limits<std::uint32_t>::max(),
            "first overflowing Human attachment Jacobian count is not exact"
        );
        auto tooManyAttachments = makeWorld(160u, 161u);
        tooManyAttachments.objects.front().femHumanAttachments.resize(
            NM_MATTER_MAX_HUMAN_ATTACHMENT_POINTS + 1u
        );
        requireRejectedWithDiagnostic(
            std::move(tooManyAttachments),
            "coupled-candidate point capacity",
            "Matter compiled a Human attachment batch that the candidate service cannot admit"
        );
        const auto compatibility = numi::matter::compileWorld(
            makeWorld(40u, 41u)
        );
        require(
            compatibility.succeeded(),
            "legacy MetalWorld capacity world did not compile"
        );
        require(
            compatibility.world.dispatch.rigidGeneralizedCapacity == 40u &&
                compatibility.world.dispatch.rigidQCapacity == 41u,
            "compatibility world did not retain exact 40-DoF/41-q capacity"
        );

        auto human = numi::matter::compileWorld(makeWorld(160u, 161u));
        require(human.succeeded(), "Numi Human capacity world did not compile");
        require(
            human.world.dispatch.rigidGeneralizedCapacity == 160u &&
                human.world.dispatch.rigidQCapacity == 161u &&
                human.world.dispatch.
                    femHumanAttachmentPointJacobianStride == 480u,
            "Numi Human world did not retain exact 160-DoF/161-q capacity"
        );
        const auto attachmentOnly = numi::matter::compileWorld(
            makeWorld(160u, 161u, false)
        );
        require(
            attachmentOnly.succeeded() &&
                attachmentOnly.world.contact.rigidProxies.empty() &&
                attachmentOnly.world.dispatch.rigidGeneralizedCapacity ==
                    160u &&
                attachmentOnly.world.dispatch.rigidQCapacity == 161u &&
                attachmentOnly.world.dispatch.femHumanAttachmentCount == 1u,
            "attachment-only Human FEM did not retain exact candidate capacity without a fake rigid proxy"
        );
        require(
            human.world.dispatch.abiVersion == NM_MATTER_ABI_VERSION &&
                human.world.dispatch.femHumanAttachmentCount == 1u &&
                human.world.fem.humanAttachments.size() == 1u,
            "Numi Human attachment count or ABI was not cooked exactly"
        );
        const NMFEMHumanAttachmentGPU attachment =
            human.world.fem.humanAttachments.front();
        require(
            attachment.identity.x == 0u &&
                attachment.identity.y == 12u &&
                attachment.identity.z == 0u &&
                attachment.identity.w == 0x4e580001u &&
                attachment.localPoint.x == 0.001f &&
                attachment.localPoint.y == -0.002f &&
                attachment.localPoint.z == 0.003f &&
                attachment.localPoint.w == 0.0f &&
                human.world.fem.nodes.front().restAndFixed.w == 2.0f,
            "Numi Human attachment record or moving-node marker is invalid"
        );
        std::string validationError;
        require(
            numi::matter::validateCompiledWorldLayout(
                human.world,
                &validationError
            ),
            "Numi Human capacity package validation failed: " +
                validationError
        );

        TemporaryPackage package;
        std::string packageError;
        require(
            numi::matter::writePackage(
                human,
                package.path(),
                &packageError
            ),
            "Numi Human attachment package write failed: " + packageError
        );
        numi::matter::CompiledWorld roundTripped;
        require(
            numi::matter::readPackage(
                package.path(),
                roundTripped,
                nullptr,
                &packageError
            ),
            "Numi Human attachment package read failed: " + packageError
        );
        require(
            roundTripped.fingerprint == human.world.fingerprint &&
                roundTripped.physicsFingerprint ==
                    human.world.physicsFingerprint &&
                roundTripped.dispatch.femHumanAttachmentCount == 1u &&
                roundTripped.dispatch.
                    femHumanAttachmentPointJacobianStride == 480u &&
                roundTripped.fem.humanAttachments.size() == 1u &&
                roundTripped.fem.humanAttachments.front().identity.w ==
                    attachment.identity.w &&
                roundTripped.fem.nodes.front().restAndFixed.w == 2.0f,
            "Numi Human attachment package roundtrip changed the cooked ABI"
        );
        validationError.clear();
        require(
            numi::matter::validateCompiledWorldLayout(
                roundTripped,
                &validationError
            ),
            "roundtripped Numi Human attachment layout failed: " +
                validationError
        );
        auto malformedStride = roundTripped;
        malformedStride.dispatch.femHumanAttachmentPointJacobianStride += 1u;
        malformedStride.fingerprint =
            numi::matter::compiledWorldFingerprint(malformedStride);
        validationError.clear();
        require(
            !numi::matter::validateCompiledWorldLayout(
                malformedStride,
                &validationError),
            "Matter layout validation admitted a non-authoritative Human attachment Jacobian stride"
        );
        auto malformedLayout = roundTripped;
        malformedLayout.fem.nodes.front().restAndFixed.w = 1.0f;
        malformedLayout.fingerprint =
            numi::matter::compiledWorldFingerprint(malformedLayout);
        validationError.clear();
        require(
            !numi::matter::validateCompiledWorldLayout(
                malformedLayout,
                &validationError
            ),
            "Matter layout validation admitted an attachment with a static marker"
        );

        auto duplicateNode = makeWorld(160u, 161u);
        auto secondAttachment =
            duplicateNode.objects.front().femHumanAttachments.front();
        secondAttachment.stableIdentifier = 0x4e580002u;
        duplicateNode.objects.front().femHumanAttachments.push_back(
            secondAttachment
        );
        requireRejected(
            std::move(duplicateNode),
            "Matter admitted duplicate Human attachments on one FEM node"
        );

        auto duplicateIdentifier = makeWorld(160u, 161u);
        secondAttachment.node = 1u;
        secondAttachment.stableIdentifier = 0x4e580001u;
        duplicateIdentifier.objects.front().femHumanAttachments.push_back(
            secondAttachment
        );
        requireRejected(
            std::move(duplicateIdentifier),
            "Matter admitted duplicate Human attachment stable identifiers"
        );

        auto fixedOverlap = makeWorld(160u, 161u);
        fixedOverlap.objects.front().femFixedNodes.push_back(0u);
        requireRejected(
            std::move(fixedOverlap),
            "Matter admitted a statically fixed Human-attached FEM node"
        );

        auto badPoint = makeWorld(160u, 161u);
        badPoint.objects.front().femHumanAttachments.front().localPoint[1] =
            std::numeric_limits<double>::quiet_NaN();
        requireRejected(
            std::move(badPoint),
            "Matter admitted a nonfinite Human attachment local point"
        );

        auto badIdentifier = makeWorld(160u, 161u);
        badIdentifier.objects.front().femHumanAttachments.front()
            .stableIdentifier = 0u;
        requireRejected(
            std::move(badIdentifier),
            "Matter admitted a zero Human attachment stable identifier"
        );

        auto badBody = makeWorld(160u, 161u);
        badBody.objects.front().femHumanAttachments.front().bodyIndex =
            NM_INVALID_INDEX;
        requireRejected(
            std::move(badBody),
            "Matter admitted an invalid Human attachment body index"
        );

        auto driftSource = makeWorld(160u, 161u);
        driftSource.objects.front().femHumanAttachments.front().localPoint[2] =
            0.004;
        const auto drifted = numi::matter::compileWorld(driftSource);
        require(
            drifted.succeeded() &&
                drifted.world.fingerprint != human.world.fingerprint &&
                drifted.world.physicsFingerprint !=
                    human.world.physicsFingerprint,
            "Human attachment changes did not drift Matter fingerprints"
        );

        const auto tooManyDofs = numi::matter::compileWorld(
            makeWorld(161u, 161u)
        );
        const auto tooManyCoordinates = numi::matter::compileWorld(
            makeWorld(160u, 162u)
        );
        require(
            !tooManyDofs.succeeded() && !tooManyCoordinates.succeeded(),
            "Matter admitted an articulated capacity above its Human ceiling"
        );

        std::cout << "NumanX Matter capacity probe passed: compatibility=40/41"
                  << " human=160/161 attachments=1 abi="
                  << NM_MATTER_ABI_VERSION << " fingerprint="
                  << human.world.fingerprint << " roundtrip="
                  << roundTripped.fingerprint << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "NumanX Matter capacity probe failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
