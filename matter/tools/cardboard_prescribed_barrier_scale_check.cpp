#include "numi/matter/matter.hpp"

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace numi::matter;
namespace {
unsigned checks = 0u;
void require(bool ok, const std::string& message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
std::string diagnostics(const CompileResult& result) {
    std::string message;
    for (const auto& diagnostic : result.diagnostics)
        message += diagnostic.message + "; ";
    return message;
}
WorldSource fixture(const MaterialProgram& material, const double scale,
                    const bool prescribed = true,
                    const bool assignScale = true) {
    WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 25.0e-6;
    source.gravity = {0.0, 0.0, 0.0};
    source.materials = {material};
    ObjectSource object;
    object.name = "scale_fingerprint_pad";
    object.materialIndex = 0u;
    object.representation = Representation::fem;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.femNodes = {{0.0,0.0,0.0}, {0.01,0.0,0.0},
                       {0.0,0.01,0.0}, {0.0,0.0,0.01}};
    object.tetrahedra = {{{0u,1u,2u,3u}}};
    object.femContactNodes = {0u,1u,2u,3u};
    source.objects.push_back(object);
    RigidProxySource punch;
    punch.shape = NM_RIGID_CAPSULE;
    punch.bodyIndex = 0u;
    punch.materialIndex = 0u;
    punch.localCenter = {0.0,0.0,0.0};
    punch.localExtent = {0.0,0.0,0.01};
    punch.radiusOrOffset = 0.001;
    punch.prescribedEndPoseTranslation = prescribed;
    if (assignScale) punch.prescribedBarrierStiffnessScale = scale;
    source.rigidProxies.push_back(punch);
    return source;
}
CompileResult cook(const WorldSource& source) {
    CompileOptions options;
    options.maximumRateExponent = 0u;
    return compileWorld(source, options);
}
void deniedSource(const WorldSource& source, const char* label) {
    const auto result = cook(source);
    require(!result.succeeded(), std::string("invalid source accepted: ") +
        label + " " + diagnostics(result));
}
void deniedCooked(CompiledWorld world, const char* label) {
    world.fingerprint = compiledWorldFingerprint(world);
    std::string error;
    require(!validateCompiledWorldLayout(world, &error),
        std::string("invalid cooked proxy accepted: ") + label);
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected material asset path");
        const auto parsed = parseMatterFile(argv[1]);
        require(parsed.succeeded(), "material parse failed");
        const auto defaultCook = cook(fixture(parsed.material, 1.0, true, false));
        const auto explicitOne = cook(fixture(parsed.material, 1.0, true, true));
        const double nearOne = std::nextafter(1.0, 2.0);
        const double nearTwoBelow = std::nextafter(2.0, 1.0);
        const double nearTwoAbove = std::nextafter(2.0, 3.0);
        const double adjacentFloatAboveTwo = static_cast<double>(
            std::nextafter(2.0f, 3.0f));
        const auto roundedOne = cook(fixture(parsed.material, nearOne));
        const auto twice = cook(fixture(parsed.material, 2.0));
        const auto roundedTwoBelow = cook(fixture(parsed.material, nearTwoBelow));
        const auto roundedTwoAbove = cook(fixture(parsed.material, nearTwoAbove));
        const auto adjacentTwo = cook(fixture(parsed.material, adjacentFloatAboveTwo));
        const auto fourfold = cook(fixture(parsed.material, 4.0));
        for (const auto* result : {&defaultCook,&explicitOne,&roundedOne,&twice,
                                   &roundedTwoBelow,&roundedTwoAbove,&adjacentTwo,
                                   &fourfold})
            require(result->succeeded(), "valid scale cook failed: " + diagnostics(*result));
        require(defaultCook.world.contact.rigidProxies[0].reserved2 == 0u &&
                explicitOne.world.contact.rigidProxies[0].reserved2 == 0u &&
                roundedOne.world.contact.rigidProxies[0].reserved2 == 0u,
                "FP32-effective scale 1 must retain canonical zero/legacy representation");
        require(defaultCook.world.fingerprint == explicitOne.world.fingerprint &&
                defaultCook.world.fingerprint == roundedOne.world.fingerprint,
                "default, explicit 1, and near-1 rounding to FP32 1 must share fingerprint");
        require(defaultCook.world.fingerprint != twice.world.fingerprint &&
                twice.world.fingerprint != fourfold.world.fingerprint,
                "scale 1, 2, and 4 must bind distinct cooked-world fingerprints");
        const std::uint32_t exactTwoBits = std::bit_cast<std::uint32_t>(2.0f);
        require(twice.world.contact.rigidProxies[0].reserved2 == exactTwoBits &&
                roundedTwoBelow.world.contact.rigidProxies[0].reserved2 == exactTwoBits &&
                roundedTwoAbove.world.contact.rigidProxies[0].reserved2 == exactTwoBits &&
                twice.world.fingerprint == roundedTwoBelow.world.fingerprint &&
                twice.world.fingerprint == roundedTwoAbove.world.fingerprint,
                "nearby source doubles rounding to FP32 2 must share the cooked factor and fingerprint");
        const float adjacentTwoValue = static_cast<float>(adjacentFloatAboveTwo);
        require(std::bit_cast<float>(adjacentTwo.world.contact.rigidProxies[0].reserved2) ==
                    adjacentTwoValue && adjacentTwo.world.fingerprint != twice.world.fingerprint,
                "a distinct adjacent FP32 factor must remain represented distinctly");
        require(std::bit_cast<float>(fourfold.world.contact.rigidProxies[0].reserved2) == 4.0f,
                "cooked scale bits must round-trip exact FP32 study factor 4");
        for (const double bad : {0.0, 0.999, 100.001,
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::quiet_NaN()})
            deniedSource(fixture(parsed.material, bad), "out-of-range/nonfinite scale");
        deniedSource(fixture(parsed.material, 2.0, false),
                     "nondefault scale without prescribed capsule mode");
        auto badCooked = twice.world;
        badCooked.contact.rigidProxies[0].reserved2 = std::bit_cast<std::uint32_t>(0.5f);
        deniedCooked(badCooked, "finite value below lower bound");
        badCooked = twice.world;
        badCooked.contact.rigidProxies[0].reserved2 = std::bit_cast<std::uint32_t>(
            std::numeric_limits<float>::quiet_NaN());
        deniedCooked(badCooked, "NaN bits");
        badCooked = twice.world;
        badCooked.contact.rigidProxies[0].flags &= ~NM_RIGID_PRESCRIBED_TRANSLATION;
        deniedCooked(badCooked, "scale bits on an unprescribed proxy");

        const auto package = std::filesystem::temp_directory_path() /
            ("matter-prescribed-barrier-scale-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) + ".nmpkg");
        std::string error;
        require(writePackage(fourfold, package, &error),
                "package write failed: " + error);
        CompiledWorld decoded;
        const bool read = readPackage(package, decoded, nullptr, &error);
        std::filesystem::remove(package);
        require(read, "package read failed: " + error);
        require(decoded.fingerprint == fourfold.world.fingerprint &&
                decoded.contact.rigidProxies[0].reserved2 ==
                    fourfold.world.contact.rigidProxies[0].reserved2,
                "package round-trip lost barrier scale or fingerprint binding");
        std::cout << "PASS prescribed barrier scale compiler checks=" << checks
                  << " factors=1,2,4 physical_steps=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL prescribed barrier scale after " << checks
                  << " checks: " << error.what() << '\n';
        return 1;
    }
}
