// Synthetic presentation-only fixture. The cube is not Unitree source geometry.
#include "metalrobo/G1.hpp"
#include "metalrobo/VisualPresentation.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    namespace fs = std::filesystem;
    fs::path cube = fs::canonical(argv[1]);
    fs::path out = fs::absolute(argv[2]);
    fs::create_directories(out);
    fs::path urdf = out/"diagnostic.urdf";
    std::ofstream f(urdf);
    f << "<robot name=\"diagnostic\"><link name=\"pelvis\">"
      << "<visual><geometry><mesh filename=\"" << cube.string()
      << "\" scale=\"0.12 0.12 0.12\"/></geometry></visual></link></robot>";
    f.close();
    auto model = metalrobo::makeUnitreeG1EngineModel();
    metalrobo::VisualAssetCookOptions options;
    options.id = "synthetic-window-probe";
    options.license = "NOASSERTION";
    options.preprocessingProvenance = "synthetic presentation-only diagnostic tetrahedron";
    for (std::uint32_t i = 0; i < model.bodyNames.size(); ++i) {
        options.linkBodyIndices.emplace(model.bodyNames[i], i);
        options.linkCenterOfMassOffsets.emplace(model.bodyNames[i], model.bodies[i].centerOfMass);
    }
    std::vector<metalrobo::VisualAssetPackV2> packs;
    auto diagnostics = metalrobo::cookUrdfVisualDescription(urdf, packs, options);
    if (!diagnostics.succeeded() || packs.size() != 1) {
        std::cerr << "cook failed: " << diagnostics.message << "\n"; return 1;
    }
    std::string reason;
    if (!metalrobo::writeVisualAssetPack(packs[0], out/"diagnostic.mrvpack", &reason)) {
        std::cerr << reason << "\n"; return 1;
    }
    std::ofstream json(out/"diagnostic.json");
    json << R"({"format":"numi.visual-observation.v1","id":"synthetic_tetrahedron_diagnostic",
      "packs":[{"path":"diagnostic.mrvpack","asset_id":"robot","semantic_id":1,"instance_id":1}],
      "camera":{"parent_body":"pelvis","position":[1.05,-1.3,0.32],
      "orientation":[-0.73858354,-0.26102070,0.20711737,0.58605883],
      "width":16,"height":9,"minimum_visible_pixels":1,
      "vertical_field_of_view_degrees":52.0,"nominal_rate_hz":50.0,"maximum_retained_bytes":0}})";
    if (!json) return 1;
    std::cout << "created synthetic presentation-only G1 pelvis tetrahedron fixture\n";
}
