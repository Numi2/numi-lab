from pathlib import Path
import shutil,json,hashlib,datetime,sys
q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
base=Path(sys.argv[1]);s=q/'snapshots/source-cohesive-quadrature-area-205'
out=q/'analysis/source205-cohesive-quadrature-area-001';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((base/'frozen-source.json').read_text())
for n,h in f['source_sha256'].items():assert sha(base/n)==h,n
shutil.copytree(base,s)
files=[]
def change(rel,old,new):
 p=s/rel;t=p.read_text();assert t.count(old)==1,(rel,old[:80],t.count(old));p.write_text(t.replace(old,new));files.append(rel)
rel='matter/tools/cardboard_probe.mm'
change(rel,'#include "cardboard_reaction_observables.hpp"','#include "cardboard_reaction_observables.hpp"\n#include "cardboard_cohesive_capacity_area.hpp"')
change(rel,'''        if (isStarch) {
            starchArea += area;''','''        const auto capacityArea = numi::cardboard::cohesiveQuadratureCapacityArea(
            area, {static_cast<double>(face.mixedPointHistory[0].w),
                   static_cast<double>(face.mixedPointHistory[1].w),
                   static_cast<double>(face.mixedPointHistory[2].w)});
        if (!capacityArea.valid) {
            validFaces = false;
            break;
        }
        if (isStarch) {
            starchArea += area;''')
change(rel,'''            if (damage >= 1.0) {
                result.starchFullyFailedAreaM2 += area;
                ++result.starchFullyFailedFaces;
            }''','''            result.starchFullyFailedAreaM2 += capacityArea.failedReferenceAreaM2;
            if (capacityArea.wholeFaceFailed) ++result.starchFullyFailedFaces;''')
change(rel,'''            if (damage >= 1.0) {
                result.hmaFullyFailedAreaM2 += area;
                ++result.hmaFullyFailedFaces;
            }''','''            result.hmaFullyFailedAreaM2 += capacityArea.failedReferenceAreaM2;
            if (capacityArea.wholeFaceFailed) ++result.hmaFullyFailedFaces;''')
p=s/rel;t=p.read_text();old='available_zero_tensile_shear_capacity_fraction_not_geometric_open_area';new='available_quadrature_zero_tensile_shear_capacity_fraction_not_geometric_open_area';assert t.count(old)==2;t=t.replace(old,new)
needle='           << "  \\"compression_work_method\\":'
assert t.count(needle)==1
line='           << "  \\"cohesive_failed_capacity_area_definition\\": \\"Reference-area integral of indicator(accepted mixedPointHistory[q].w == 1) with the native three equal quadrature weights; fully_failed_faces counts only faces with all three points failed. Face maximum state.x is not used for this integral. This is lost cohesive capacity, not geometric open area.\\",\\n"\n'
t=t.replace(needle,line+needle);p.write_text(t)
change('matter/CMakeLists.txt','foreach(cardboard_helper shipping-driver shipping-memory shipping-closure-mesh','foreach(cardboard_helper cohesive-capacity-area shipping-driver shipping-memory shipping-closure-mesh')
for name,text in {"cardboard_cohesive_capacity_area.hpp":"#pragma once\n#include <array>\n#include <cmath>\n#include <cstdint>\nnamespace numi::cardboard {\n// The native mixed-mode face uses three equal reference-area quadrature\n// weights. face.state.x is their MAXIMUM damage, not a capacity-area integral.\nstruct CohesiveQuadratureCapacityArea {\n    bool valid = false;\n    double failedReferenceAreaM2 = 0.0;\n    std::uint32_t failedQuadraturePoints = 0u;\n    bool wholeFaceFailed = false;\n};\ninline CohesiveQuadratureCapacityArea cohesiveQuadratureCapacityArea(\n    const double referenceAreaM2, const std::array<double, 3>& pointDamage) {\n    CohesiveQuadratureCapacityArea result;\n    if (!(std::isfinite(referenceAreaM2) && referenceAreaM2 > 0.0))\n        return result;\n    for (const double value : pointDamage) {\n        if (!(std::isfinite(value) && value >= 0.0 && value <= 1.0))\n            return {};\n        if (value == 1.0) ++result.failedQuadraturePoints;\n    }\n    result.failedReferenceAreaM2 = referenceAreaM2 *\n        (static_cast<double>(result.failedQuadraturePoints) / 3.0);\n    result.wholeFaceFailed = result.failedQuadraturePoints == 3u;\n    result.valid = true;\n    return result;\n}\n}\n","cardboard_cohesive_capacity_area_check.cpp":"#include \"cardboard_cohesive_capacity_area.hpp\"\n#include <cmath>\n#include <iostream>\n#include <limits>\n#include <stdexcept>\nusing numi::cardboard::cohesiveQuadratureCapacityArea;\nnamespace {\nvoid require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }\nvoid near(double a, double b) { require(std::abs(a-b) < 1e-14, \"reference area mismatch\"); }\n}\nint main() {\n    try {\n        const auto intact = cohesiveQuadratureCapacityArea(6.0, {0.0,0.0,0.0});\n        require(intact.valid && intact.failedQuadraturePoints==0u &&\n                !intact.wholeFaceFailed, \"intact face failed\");\n        near(intact.failedReferenceAreaM2, 0.0);\n        // A triangular face with three equal native quadrature cells:\n        // failure of only one cell removes 2 of its 6 m^2 reference area,\n        // although the native face maximum is already 1.\n        const auto partial = cohesiveQuadratureCapacityArea(6.0, {0.2,1.0,0.7});\n        require(partial.valid && partial.failedQuadraturePoints==1u &&\n                !partial.wholeFaceFailed, \"nonuniform face counted wholly failed\");\n        near(partial.failedReferenceAreaM2, 2.0);\n        const auto two = cohesiveQuadratureCapacityArea(6.0, {1.0,1.0,0.999999});\n        require(two.valid && two.failedQuadraturePoints==2u &&\n                !two.wholeFaceFailed, \"near-complete damage was rounded to failure\");\n        near(two.failedReferenceAreaM2, 4.0);\n        const auto full = cohesiveQuadratureCapacityArea(6.0, {1.0,1.0,1.0});\n        require(full.valid && full.wholeFaceFailed, \"complete failure omitted\");\n        near(full.failedReferenceAreaM2, 6.0);\n        // Refining a uniformly failed cell into two half-area cells leaves\n        // the integrated failed reference area unchanged.\n        const auto left = cohesiveQuadratureCapacityArea(3.0, {1.0,1.0,1.0});\n        const auto right = cohesiveQuadratureCapacityArea(3.0, {1.0,1.0,1.0});\n        near(left.failedReferenceAreaM2+right.failedReferenceAreaM2, full.failedReferenceAreaM2);\n        for (const double d : {-0.1,1.1,std::numeric_limits<double>::quiet_NaN(),\n                              std::numeric_limits<double>::infinity()})\n            require(!cohesiveQuadratureCapacityArea(1.0,{0.0,d,1.0}).valid,\n                    \"invalid history admitted\");\n        for (const double a : {0.0,-1.0,std::numeric_limits<double>::infinity()})\n            require(!cohesiveQuadratureCapacityArea(a,{0.0,0.0,0.0}).valid,\n                    \"invalid area admitted\");\n        std::cout << \"cohesive capacity area: nonuniform quadrature, refinement and invalid-history checks passed; CPU observer only\\n\";\n        return 0;\n    } catch (const std::exception& e) { std::cerr<<e.what()<<'\\n'; return 1; }\n}\n"}.items():
 rel='matter/tools/'+name;(s/rel).write_text(text);files.append(rel)
rel='matter/tools/cardboard_stage_resolution_study.py';p=s/rel;t=p.read_text()
a='''        if status != "available_zero_tensile_shear_capacity_fraction_not_geometric_open_area":
            capacities[role] = {"status": status or "inconclusive_missing_status"}
            continue'''
b='''        if status != "available_quadrature_zero_tensile_shear_capacity_fraction_not_geometric_open_area":
            capacities[role] = {"status": (
                "inconclusive_legacy_face_maximum_not_quadrature_area"
                if status == "available_zero_tensile_shear_capacity_fraction_not_geometric_open_area"
                else status or "inconclusive_missing_status")}
            continue'''
assert t.count(a)==1;t=t.replace(a,b)
t=t.replace('sum of active cohesive reference face areas with accepted damage state >= 1, divided by all activated reference face area for this material group','sum of active reference face area times the fraction of the three equal quadrature weights with accepted point damage exactly 1; divided by activated reference area')
p.write_text(t);files.append(rel)
rel='matter/tools/test_cardboard_stage_resolution_study.py';p=s/rel;t=p.read_text();assert t.count(old)==1;t=t.replace(old,new)
anchor='    def test_fully_failed_area_rejects_inconsistent_active_area_denominator(self) -> None:'
test='''    def test_legacy_face_maximum_area_is_not_admitted_as_quadrature_capacity(self) -> None:
        run_dir = fake_box_run(self.root, "legacy-face-max")
        history_path = run_dir / "shipping-history-observables.csv"
        fields, rows = study._rows(history_path)
        rows[0]["starch_fully_failed_status"] = "available_zero_tensile_shear_capacity_fraction_not_geometric_open_area"
        write_csv(history_path, fields, rows)
        history = study._history_metrics(study.load_run(run_dir))
        self.assertEqual(history["fully_failed_cohesive_capacity_area"]["starch"]["status"],
                         "inconclusive_legacy_face_maximum_not_quadrature_area")
        self.assertEqual(history["fully_failed_cohesive_capacity_area_status"],
                         "inconclusive_missing_or_invalid_capacity_area")

'''
assert t.count(anchor)==1;t=t.replace(anchor,test+anchor);p.write_text(t);files.append(rel)
rel='matter/tools/cardboard_stage_resolution_study.md';p=s/rel;t=p.read_text()
a='''reference area of accepted active faces whose damage state is 1.0, divided by
the total activated reference area for that interface group. The numerator
means zero remaining tensile and shear capacity in the constitutive law.'''
b='''reference-area integral of fully failed accepted quadrature points, divided by
the total activated reference area for that interface group. Each mixed-mode
face has three equal reference-area weights. The maximum damage stored in
face.state.x is not an area fraction: one failed point contributes one third
of that face area, and a fully failed face count requires all three points.
Legacy exports using the face maximum are retained but cannot satisfy this
capacity-area comparison. The numerator means zero remaining tensile and shear
capacity at those quadrature points in the constitutive law.'''
assert t.count(a)==1;p.write_text(t.replace(a,b));files.append(rel)
files=sorted(set(files))
for rel in files:f['source_sha256'][rel]=sha(s/rel)
f.update(snapshot_label=s.name,created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_snapshot=str(base),scope='Accepted shipping cohesive capacity-area observer uses native point history and equal quadrature weights; no physical operator or tolerance change.')
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n')
(out/'source-overlay.json').write_text(json.dumps({'base':str(base),'base_frozen_sha256':sha(base/'frozen-source.json'),'source':str(s),'frozen_sha256':sha(s/'frozen-source.json'),'changed_sources':{n:{'before':sha(base/n) if (base/n).exists() else None,'after':sha(s/n)} for n in files},'physical_operator_changed':False,'native_check_pending':True,'historical_max_face_reports_are_not_retroactively_relabelled':True},indent=2,sort_keys=True)+'\n')
print(str(s));print(sha(s/'frozen-source.json'));print(len(f['source_sha256']))
