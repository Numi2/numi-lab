#include "numi/matter/matter.hpp"
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 2) return 64;
  using namespace numi::matter;
  auto parsed = parseMatterFile(argv[1]);
  if (!parsed.succeeded()) {
    for (const auto& d : parsed.diagnostics) std::cerr << d.message << "\n";
    return 2;
  }
  if (parsed.material.name != "human_optic_nerve_longitudinal_low_strain_isotropic_effective" || parsed.material.hint != ConstitutiveHint::generic) return 3;
  WorldSource source;
  source.environmentCount = 1;
  source.frameTimestep = 0.00025;
  source.materials.push_back(std::move(parsed.material));
  ObjectSource object;
  object.name = "optic_nerve_material_cpu_compile_fixture";
  object.materialIndex = 0;
  object.representation = Representation::fem;
  object.automaticRepresentation = false;
  object.twoWayCoupling = false;
  object.adaptive = false;
  object.identifiable = false;
  object.deformableContact = false;
  object.deformableSelfContact = false;
  object.mixedFEM = false;
  object.characteristicLength = 0.01;
  object.femNodes = {{0,0,0},{0.01,0,0},{0,0.01,0},{0,0,0.01}};
  object.tetrahedra.push_back({{0,1,2,3}});
  source.objects.push_back(std::move(object));
  auto compiled = compileWorld(source, {.maximumRateExponent=NM_MAX_RATE_EXPONENT});
  if (!compiled.succeeded()) {
    for (const auto& d : compiled.diagnostics) if (d.severity == Diagnostic::Severity::error) std::cerr << d.message << "\n";
    return 4;
  }
  if (compiled.world.objects.size()!=1 || compiled.world.constitutive.size()!=1 || compiled.world.fem.tetrahedra.size()!=1) return 5;
  std::cout << "parse=PASS compileWorld=PASS model=generic materials=" << compiled.world.constitutive.size() << " fem_tets=" << compiled.world.fem.tetrahedra.size() << " generated_metal_bytes=" << compiled.generatedMetal.size() << " gpu=NOT_USED\n";
}
