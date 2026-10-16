#include "numi/matter/matter.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <type_traits>
#include <limits>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
using namespace numi::matter;
struct Digest {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> bytes{};
};

std::string hexDigest(const Digest& digest);

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Digest sha256(const void* bytes, const std::size_t count) {
    Digest result;
    require(count <= std::numeric_limits<CC_LONG>::max(),
            "SHA-256 input exceeds CommonCrypto's bounded length");
    CC_SHA256(bytes, static_cast<CC_LONG>(count), result.bytes.data());
    return result;
}
std::string hexDigest(const Digest& digest) {
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (const unsigned char byte : digest.bytes)
        result << std::setw(2) << static_cast<unsigned>(byte);
    return result.str();
}
template<class T>
std::string sha256Arena(const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (values.empty()) return hexDigest(sha256("", 0u));
    require(values.size() <= std::numeric_limits<std::size_t>::max() / sizeof(T),
            "shipping state arena byte count overflowed");
    return hexDigest(sha256(values.data(), values.size() * sizeof(T)));
}
int main(int argc,char** argv) {
 require(argc==3,"output directory and empty/full case required");
 const std::filesystem::path directory=argv[1];
 std::filesystem::create_directories(directory);
 RuntimeStateSnapshot snapshot;
 if(std::string(argv[2])=="full") {
   NMFEMNodeStateGPU n{};
   n.positionAndMass={1.0f,2.0f,3.0f,0.125f};
   n.velocityAndInverseMass={-0.0f,1.0f/3.0f,-2.5f,8.0f};
   n.restAndFixed={4.0f,5.0f,6.0f,2.0f};
   n.referenceDisplacementAndMode={1.0e-8f,-2.0e-8f,3.0e-8f,1.0f};
   snapshot.femRegionalGalerkinLinearizationNodes={n};
   snapshot.femRegionalGalerkinLinearizationMaterialState={-0.0f,1.0f/3.0f,1.0e-30f,-1.0e20f};
   snapshot.femRegionalGalerkinLinearizationSchedulerExponent={0u,1u,3u};
   snapshot.femRegionalGalerkinLinearizationLocalFHigh={{1.0f,0.0f,0.0f,0.0f},{0.0f,1.0f,0.0f,0.0f},{0.0f,0.0f,1.0f,0.0f}};
   snapshot.femRegionalGalerkinLinearizationLocalFLow={{1.0e-8f,-2.0e-8f,3.0e-8f,0.0f},{-4.0e-8f,5.0e-8f,-6.0e-8f,0.0f},{7.0e-8f,-8.0e-8f,9.0e-8f,-0.0f}};
 }
 std::ofstream metadata(directory/"manifest.json");
 require(metadata.good(),"cannot create fixture metadata");
 metadata<<"{\"files\":[";
 bool first=true;
        const auto writeArena = [&](const char* name, const auto& values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            const auto filename = std::string(name) + ".bin";
            std::ofstream arena(directory / filename, std::ios::binary);
            require(arena.good(), "cannot write rejected conditioning arena");
            const std::size_t bytes = values.size() * sizeof(Value);
            if (bytes != 0u) arena.write(reinterpret_cast<const char*>(values.data()), bytes);
            require(arena.good(), "rejected conditioning arena write failed");
            if (!first) metadata << ",\n";
            first = false;
            metadata << "    {\"name\": \"" << filename << "\", \"count\": "
                << values.size() << ", \"stride_bytes\": " << sizeof(Value)
                << ", \"bytes\": " << bytes << ", \"sha256\": \""
                << sha256Arena(values) << "\"}";
        };
        writeArena("candidate_nodes", snapshot.femRegionalGalerkinLinearizationNodes);
        // These arrays were already read back under Runtime's existing
        // regional 64 MiB bound. Preserve the exact AP linearization inputs
        // instead of leaving them only in the in-memory snapshot. This is
        // accepted constitutive history, never a current projected state.
        writeArena("linearization_accepted_material_history",
                   snapshot.femRegionalGalerkinLinearizationMaterialState);
        writeArena("linearization_scheduler_exponents",
                   snapshot.femRegionalGalerkinLinearizationSchedulerExponent);
        writeArena("linearization_F_high",
                   snapshot.femRegionalGalerkinLinearizationLocalFHigh);
        writeArena("linearization_F_low",
                   snapshot.femRegionalGalerkinLinearizationLocalFLow);

 metadata<<"]}\n";
 require(metadata.good(),"cannot close fixture metadata");
 return 0;
}
