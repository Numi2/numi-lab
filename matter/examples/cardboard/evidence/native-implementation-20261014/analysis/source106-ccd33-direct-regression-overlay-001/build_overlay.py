from pathlib import Path
import json
import hashlib
import struct
import shutil

BASE = Path('/Users/home/cardboard-implementation-20261014/snapshots/source-capture-callback-lifetime-106')
PKG = Path('/Users/home/cardboard-implementation-20261014/analysis/source106-ccd33-direct-regression-overlay-001')
BASELINE_RUN = Path('/Users/home/cardboard-implementation-20261014/analysis/source103-pair-ccd-native-367/native-result.json')
CONTINUED_RUN = Path('/Users/home/cardboard-implementation-20261014/analysis/source103-pair-ccd-native-continuation-372/native-result.json')

contact_path = PKG/'overlay/matter/src/metal/contact.metalinc'
text = contact_path.read_text()
# Scope the mutation to the production definition, the second definition in this file.
needle = 'inline bool edgeEdgeCCD('
positions=[]
pos=0
while True:
    i=text.find(needle,pos)
    if i<0: break
    positions.append(i); pos=i+len(needle)
assert len(positions)==2, positions
start=positions[1]
loop=text.find('for (uint iteration = 0u; iteration < 17u; ++iteration)',start)
assert loop>=0 and text.find('inline bool vertexTriangleCCD(',start)<0
text=text[:loop]+text[loop:].replace('for (uint iteration = 0u; iteration < 17u; ++iteration)', 'for (uint iteration = 0u; iteration < 33u; ++iteration)',1)
contact_path.write_text(text)

check_path = PKG/'overlay/matter/tools/cardboard_contact_ccd_native_check.mm'
s = check_path.read_text()
s=s.replace('constexpr char kCapturedRaySha256[] =\n    "9b50eef2b182f38e061eff23eaa46d55b158539fa579340280b9f801a1f68bc7";', '''constexpr char kCapturedRaySha256[] =
    "e116911335b4fbf88683535cde08a97b26ca99bc1975989d67532196c2ff996f";
constexpr char kBaseline367ResultSha256[] =
    "8b9b1ef8b0a32adb3938a39254ac57698fceb7b19265bae3fa69329562b3b6cc";
constexpr char kContinuation372ResultSha256[] =
    "c8fd27646a56d117931552dbe99a3c119976ea7061d53365e24080af81bda45b";''')
s=s.replace('    bool expectShortenedCertificate = false;\n    bool verifySampledDistance = true;', '    bool expectShortenedCertificate = false;\n    bool expectRoundoffBandHit = false;\n    bool verifySampledDistance = true;')
needle='''Case degenerateCollinearCross(const float selector = 0.0f) {'''
insert='''float floatFromBits(const std::uint32_t bits) {
    float value = 0.0f;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

Point pointFromBits(const std::uint32_t x, const std::uint32_t y,
                    const std::uint32_t z) {
    return {floatFromBits(x), floatFromBits(y), floatFromBits(z)};
}

Case capturedNearBandEdgeEdgeExhaustion(const float selector) {
    // Exact FP32 input for source103 capture pair (primitive 92,127), case
    // trial358_ee_edge0_edge0. Source103 bound17 exhausted at iteration16;
    // bound33 reached a certified hit on evaluation17. The returned point is
    // inside the implementation's roundoff-expanded band, not a robust
    // nominal-thickness crossing. This is a CCD continuation regression only.
    Case c;
    c.name = selector >= 2.0f
        ? "captured_ee_roundoff_band_17_to_33_helper"
        : "captured_ee_roundoff_band_17_to_33_direct";
    c.kind = kEdgeEdge;
    c.selector = selector;
    c.thickness = floatFromBits(0x3727c5acu);
    c.expectedHit = true;
    c.expectRoundoffBandHit = true;
    c.verifySampledDistance = false;
    c.start = {{
        pointFromBits(0x3b031297u, 0x3236273cu, 0x3a831ea8u),
        pointFromBits(0x3b03126au, 0x3120b246u, 0x34dbee94u),
        pointFromBits(0xb3d85123u, 0xbb03137eu, 0x3a846bdfu),
        pointFromBits(0x3b0310a7u, 0xb3794115u, 0x3a84703bu),
    }};
    c.end = {{
        pointFromBits(0x3b0312a4u, 0x325a5b76u, 0x3a8321aeu),
        pointFromBits(0x3b031269u, 0x314b9a59u, 0x350b6ba8u),
        pointFromBits(0xb3d8283bu, 0xbb03137fu, 0x3a846d2fu),
        pointFromBits(0x3b0310a7u, 0xb379acd4u, 0x3a84712bu),
    }};
    return c;
}

float roundoffExpandedThickness(const Case& c) {
    float coordinateScale = 0.0f;
    for (const auto* points : {&c.start, &c.end}) {
        for (const Point& p : *points) {
            coordinateScale = std::max(coordinateScale,
                std::fabs(static_cast<float>(p[0])));
            coordinateScale = std::max(coordinateScale,
                std::fabs(static_cast<float>(p[1])));
            coordinateScale = std::max(coordinateScale,
                std::fabs(static_cast<float>(p[2])));
        }
    }
    const float scale = std::max(std::max(c.thickness, coordinateScale), 1.0e-12f);
    const float roundoff = 8.0f * std::numeric_limits<float>::epsilon() * scale;
    return c.thickness + roundoff;
}

void checkRoundoffBandHit(const Case& c, const Float4& result) {
    if (!c.expectRoundoffBandHit) return;
    const float effectiveThickness = roundoffExpandedThickness(c);
    require(result[0] == 1.0f && result[1] == 1.0f,
        c.name + ": exact captured ray was not certified as a hit at bound33");
    require(result[3] >= c.thickness && result[3] <= effectiveThickness,
        c.name + ": returned separation left nominal/roundoff band bounds");
    require(result[2] >= 0.95f && result[2] <= 1.0f,
        c.name + ": returned hit is outside the captured terminal-time region");
}

'''+needle
assert needle in s
s=s.replace(needle,insert)
# Add roundoff-band gate to both checker paths.
s=s.replace('''        require(std::isfinite(r[3]), c.name + ": direct CCD separation is nonfinite");
        if (c.verifySampledDistance) {''','''        require(std::isfinite(r[3]), c.name + ": direct CCD separation is nonfinite");
        checkRoundoffBandHit(c, r);
        if (c.verifySampledDistance) {''')
s=s.replace('''        require(std::isfinite(r[2]) && r[2] >= 0.0f && r[2] <= 1.0f &&
                std::isfinite(r[3]), c.name + ": helper outcome contains invalid geometry");
        if (c.verifySampledDistance) {''','''        require(std::isfinite(r[2]) && r[2] >= 0.0f && r[2] <= 1.0f &&
                std::isfinite(r[3]), c.name + ": helper outcome contains invalid geometry");
        checkRoundoffBandHit(c, r);
        if (c.verifySampledDistance) {''')
# Add exact case to both direct and helper lists.
s=s.replace('''                edgeEdgeEndpointClearCross(), edgeEdgeNearParallelLongTangent(),
                degenerateCollinearCross(), capturedUncertifiedClearRay(0.0f),''','''                edgeEdgeEndpointClearCross(), edgeEdgeNearParallelLongTangent(),
                capturedNearBandEdgeEdgeExhaustion(1.0f),
                degenerateCollinearCross(), capturedUncertifiedClearRay(0.0f),''')
s=s.replace('''                edgeEdgeEndpointClearCross(2.0f), edgeEdgeNearParallelLongTangent(2.0f),
                degenerateCollinearCross(2.0f),''','''                edgeEdgeEndpointClearCross(2.0f), edgeEdgeNearParallelLongTangent(2.0f),
                capturedNearBandEdgeEdgeExhaustion(2.0f),
                degenerateCollinearCross(2.0f),''')
# Change obsolete identifier to a descriptive evidence source name.
s=s.replace('''                << "\\\"captured_diagnostic_ray_sha256\\\":\\\"" << kCapturedRaySha256 << "\\\","
                << "\\\"captured_diagnostic_ray_id\\\":\\\"contact-regional-all-pair-gate-092/slot129\\\","
                << "\\\"captured_diagnostic_ray_artifact\\\":\\\"full-contact-self-ccd-input-092-001.json\\\","
                << "\\\"captured_diagnostic_ray_fixture_initial_geometry_valid\\\":false,"
                << "\\\"captured_diagnostic_ray_thickness_source\\\":\\\"16*FLT_EPSILON*max_abs_endpoint_coordinate reconstructed from FP32 endpoints\\\","
                << "\\\"captured_diagnostic_ray_physical_qualification\\\":false,"''','''                << "\\\"captured_diagnostic_ray_sha256\\\":\\\"" << kCapturedRaySha256 << "\\\","
                << "\\\"captured_ee_iteration_comparison\\\":{"
                << "\\\"case\\\":\\\"trial358_ee_edge0_edge0\\\","
                << "\\\"baseline_source\\\":\\\"source103\\\","
                << "\\\"baseline_iteration_bound\\\":17,"
                << "\\\"baseline_terminal_iteration\\\":16,"
                << "\\\"baseline_certified\\\":false,"
                << "\\\"baseline_result_sha256\\\":\\\"" << kBaseline367ResultSha256 << "\\\","
                << "\\\"continuation_iteration_bound\\\":33,"
                << "\\\"continuation_terminal_iteration\\\":17,"
                << "\\\"continuation_certified\\\":true,"
                << "\\\"continuation_result_sha256\\\":\\\"" << kContinuation372ResultSha256 << "\\\","
                << "\\\"near_band_only_not_nominal_crossing\\\":true,"
                << "\\\"full_contact_same_build_required\\\":true},"
                << "\\\"captured_diagnostic_ray_id\\\":\\\"source103/trial358_ee_edge0_edge0/pair92-127\\\","
                << "\\\"captured_diagnostic_ray_fixture_initial_geometry_valid\\\":false,"
                << "\\\"captured_diagnostic_ray_physical_qualification\\\":false,"''')
# Emit explicit classification on each case.
s=s.replace('''                    << "\\\",\\\"hit\\\":" << (direct[i][0] > 0.5f ? "true" : "false")
                    << ",\\\"certified\\\":"''','''                    << "\\\",\\\"near_band_only\\\":"
                    << (directCases[i].expectRoundoffBandHit ? "true" : "false")
                    << ",\\\"hit\\\":" << (direct[i][0] > 0.5f ? "true" : "false")
                    << ",\\\"certified\\\":"''')
s=s.replace('''                    << "\\\",\\\"hit\\\":" << (first[0] > 0.5f ? "true" : "false")
                    << ",\\\"certified\\\":"''','''                    << "\\\",\\\"near_band_only\\\":"
                    << (helperCases[i].expectRoundoffBandHit ? "true" : "false")
                    << ",\\\"hit\\\":" << (first[0] > 0.5f ? "true" : "false")
                    << ",\\\"certified\\\":"''')
check_path.write_text(s)

# Preserve just the selected evidence row and bind its originating complete artifacts.
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=json.loads(BASELINE_RUN.read_text())
cont=json.loads(CONTINUED_RUN.read_text())
def find_case(data):
    matches=[(i,c) for i,c in enumerate(data['cases']) if c.get('name')=='trial358_ee_edge0_edge0']
    assert len(matches)==1
    return matches[0]
i0,c0=find_case(base); i1,c1=find_case(cont)
assert i0==i1==21
assert c0['points_start_f32_bits']==c1['points_start_f32_bits']
assert c0['points_finish_f32_bits']==c1['points_finish_f32_bits']
assert c0['record_f32_bits'][1]=='3727c5ac'
evidence={
  'schema':'numi.cardboard.source106_ee33_captured_ray_evidence.v1',
  'scope':'direct CCD regression input only; not a physical contact or full-contact qualification',
  'capture_case':{
    'name':c1['name'],'case_index':i1,'kind':c1['kind'],'primitive_pair':c1['primitive_pair'],
    'primitive_nodes':c1['primitive_nodes'],'candidate_identity':c1['candidate_identity'],
    'thickness_f32_bits':c1['record_f32_bits'][1],
    'points_start_f32_bits':c1['points_start_f32_bits'],
    'points_finish_f32_bits':c1['points_finish_f32_bits'],
    'record_f32_bits':c1['record_f32_bits'],
  },
  'native_source103_comparison':{
    'iteration_bound_baseline':17,'iteration_bound_continuation':33,
    'baseline':{'run':'367','result_sha256':sha(BASELINE_RUN),'certified':c0['production_result']['certified'],
                'hit':c0['production_result']['hit'],'toi':c0['production_result']['toi'],
                'separation_m':c0['production_result']['separation_m'],
                'trace_rows':len(c0['native_conservative_advance_trace']),
                'terminal_iteration':c0['native_conservative_advance_trace'][-1]['iteration']},
    'continuation':{'run':'372','result_sha256':sha(CONTINUED_RUN),'certified':c1['production_result']['certified'],
                    'hit':c1['production_result']['hit'],'toi':c1['production_result']['toi'],
                    'separation_m':c1['production_result']['separation_m'],
                    'trace_rows':len(c1['native_conservative_advance_trace']),
                    'terminal_iteration':c1['native_conservative_advance_trace'][-1]['iteration']},
    'source103_same_input_corpus_sha256':base['cases_bin_sha256'],
    'baseline_and_continuation_input_corpus_match':base['cases_bin_sha256']==cont['cases_bin_sha256'],
  },
  'geometry_interpretation':{
    'nominal_thickness_m':c1['thickness_m'],
    'continuation_effective_thickness_m':c1['trace_clone_result']['effective_thickness_m'],
    'roundoff_addition_m':c1['cpu_fp64_reference']['roundoff_addition_m'],
    'returned_separation_m':c1['production_result']['separation_m'],
    'returned_separation_minus_nominal_m':c1['production_result']['separation_m']-c1['thickness_m'],
    'independent_sampled_minimum_m':c1['cpu_fp64_reference']['minimum_refined_distance_m'],
    'sampled_minimum_is_lower_bound':c1['cpu_fp64_reference']['minimum_distance_is_lower_bound'],
    'nominal_crossing_claimed':False,
    'note':'The hit is within the implementation roundoff-expanded band. The independent sampled endpoint is only sub-nanometer inside nominal thickness and is not a lower-bound proof; do not call this a robust nominal-thickness crossing.'
  },
  'source_artifacts':{
    'baseline_native_result':str(BASELINE_RUN),'baseline_native_result_sha256':sha(BASELINE_RUN),
    'continuation_native_result':str(CONTINUED_RUN),'continuation_native_result_sha256':sha(CONTINUED_RUN),
    'adjudication_report':str(PKG.parent/'source103-pair-ccd-continuation-372-adjudication-001/report.json'),
    'adjudication_report_sha256':sha(PKG.parent/'source103-pair-ccd-continuation-372-adjudication-001/report.json')
  }
}
(PKG/'evidence/captured-ee-ray.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
print('overlay written',PKG)
