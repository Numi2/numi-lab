from pathlib import Path
import json,hashlib,shutil,subprocess,datetime,re
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-contact-compaction-fixture-249-001';s=q/'snapshots/source-primary-cache-fd-action-253-001';d=q/'analysis/source253-primary-cache-fd-action-001';d.mkdir(exist_ok=False);s.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(base/'frozen-source.json')=='a06587ac4ce299370ee68646d57d49aef3b9f69cd3051d07db9de2912d7c8ba2'
f=json.loads((base/'frozen-source.json').read_text());cfg=json.loads((base/'configure-support-source.json').read_text())
oldmap={**cfg['files'],**f['source_sha256']}
for n,h in oldmap.items():
 p=base/n;assert sha(p)==h,n;z=s/n;z.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,z)
shutil.copy2(base/'configure-support-source.json',s/'configure-support-source.json')
header='matter/include/numi/matter/detail/fem_primary_operator_cache_mode.hpp'
(s/header).write_text("""#pragma once
#include <cstdint>
namespace numi::matter::detail {
// One shared policy for primary Arnoldi, true residual, center action and FD.
// RAP mode2 and CGLS transpose/forward modes remain separate algorithms.
constexpr std::uint32_t femPrimaryOperatorCacheMode(
    const std::uint32_t executionBudgetFlag) noexcept {
    return executionBudgetFlag == 1u ? 1u : 0u;
}
}
""")
p=s/'matter/src/runtime.mm';t=p.read_text();needle='#include "numi/matter/detail/shipping_contact_page_plan.hpp"';assert t.count(needle)==1;t=t.replace(needle,needle+'\n#include "numi/matter/detail/fem_primary_operator_cache_mode.hpp"')
old='state.mixedSolverValue.executionBudgets.w == 1u ? 1u : 0u';n=t.count(old);assert n==4,n;t=t.replace(old,'numi::matter::detail::femPrimaryOperatorCacheMode(\n                            state.mixedSolverValue.executionBudgets.w)')
old='''[encoder setBuffer:state.dummy offset:0u atIndex:23u];
                    [encoder setBuffer:state.dummy offset:0u atIndex:24u];
                    const std::uint32_t useCache = 0u;'''
new='''[encoder setBuffer:state.femAlgorithmicTangentCache != nil
                        ? state.femAlgorithmicTangentCache : state.dummy
                                 offset:0u atIndex:23u];
                    [encoder setBuffer:state.femAlgorithmicTangentCacheValid != nil
                        ? state.femAlgorithmicTangentCacheValid : state.dummy
                                 offset:0u atIndex:24u];
                    // Match the primary Arnoldi/true-residual operator. A
                    // separately evaluated direct tangent is still captured
                    // in the local constitutive probe, with an explicit label.
                    const std::uint32_t useCache =
                        numi::matter::detail::femPrimaryOperatorCacheMode(
                            state.mixedSolverValue.executionBudgets.w);'''
assert t.count(old)==1;t=t.replace(old,new)
old=r'header << "],\"cached_algorithmic_tangent_capture\":{";'
new=r'''header << "],\"operator_action_policy\":{"
                                       << "\"selection\":\"matches_primary_FGMRES_cache_mode\""
                                       << ",\"cache_mode\":"
                                       << (capturedCachedTangentPolicyEnabled ? 1u : 0u)
                                       << ",\"element_force_source\":\"same_operator_dispatch_as_nodal_action\""
                                       << ",\"local_stress_tangent_source\":\"independent_direct_constitutive_probe\""
                                       << "},\"cached_algorithmic_tangent_capture\":{";'''
assert t.count(old)==1,t.count(old);t=t.replace(old,new)
p.write_text(t)
p=s/'matter/tools/cardboard_cached_operator_audit.py';t=p.read_text()
anchor='def run(args: argparse.Namespace) -> dict:'
helper='''def capture_operator_action_policy(header: dict) -> dict:
    """Legacy captures omitted the primary cache binding and used mode zero."""
    policy = header.get("operator_action_policy")
    if policy is None:
        return {"production_operator_action_bound": False,
                "cache_mode": None,
                "classification": "legacy action lacks primary-cache binding; cached-production parity unestablished"}
    if not isinstance(policy, dict):
        raise ValueError("operator action policy must be an object")
    expected = {"selection", "cache_mode", "element_force_source",
                "local_stress_tangent_source"}
    if set(policy) != expected:
        raise ValueError("operator action policy fields are incomplete or unknown")
    enabled = header.get("cached_algorithmic_tangent_capture", {}).get("policy_enabled")
    if type(enabled) is not bool:
        raise ValueError("operator action requires a Boolean primary cache policy")
    mode = policy.get("cache_mode")
    if type(mode) is not int or mode != int(enabled):
        raise ValueError("captured operator cache mode differs from the primary cache policy")
    if (policy["selection"] != "matches_primary_FGMRES_cache_mode" or
            policy["element_force_source"] != "same_operator_dispatch_as_nodal_action" or
            policy["local_stress_tangent_source"] != "independent_direct_constitutive_probe"):
        raise ValueError("unsupported operator action policy")
    return {"production_operator_action_bound": True, "cache_mode": mode,
            "classification": "primary production cache-mode action; measured comparison without noise-floor qualification"}


'''
assert t.count(anchor)==1;t=t.replace(anchor,helper+anchor)
old='''    action_vectors = meta.pop("action_component_arrays")'''
new='''    action_policy = capture_operator_action_policy(header)
    meta["operator_action_policy"] = action_policy
    meta["action_parity"]["classification"] = action_policy["classification"]
    action_vectors = meta.pop("action_component_arrays")'''
assert t.count(old)==1;t=t.replace(old,new)
p.write_text(t)
test='matter/tools/test_cardboard_primary_cache_action.py'
(s/test).write_text("""from pathlib import Path
import copy,subprocess,tempfile,unittest
import cardboard_cached_operator_audit as audit

ROOT=Path(__file__).resolve().parents[2]
class PrimaryCacheActionTests(unittest.TestCase):
    def header(self,enabled):
        return {"cached_algorithmic_tangent_capture":{"policy_enabled":enabled},
                "operator_action_policy":{"selection":"matches_primary_FGMRES_cache_mode",
                    "cache_mode":int(enabled),"element_force_source":"same_operator_dispatch_as_nodal_action",
                    "local_stress_tangent_source":"independent_direct_constitutive_probe"}}
    def test_enabled_and_disabled_primary_policies(self):
        for enabled in (False,True):
            self.assertTrue(audit.capture_operator_action_policy(self.header(enabled))["production_operator_action_bound"])
    def test_legacy_has_no_production_cache_claim(self):
        self.assertFalse(audit.capture_operator_action_policy({})["production_operator_action_bound"])
    def test_mismatch_and_boolean_mode_rejected(self):
        for mode in (0,True,2,"1"):
            h=self.header(True);h["operator_action_policy"]["cache_mode"]=mode
            with self.assertRaises(ValueError):audit.capture_operator_action_policy(h)
    def test_unknown_or_missing_fields_rejected(self):
        h=self.header(True);del h["operator_action_policy"]["element_force_source"]
        with self.assertRaises(ValueError):audit.capture_operator_action_policy(h)
        h=self.header(True);h["operator_action_policy"]["unknown"]=1
        with self.assertRaises(ValueError):audit.capture_operator_action_policy(h)
    def test_actual_fd_dispatch_uses_same_primary_helper(self):
        runtime=(ROOT/"matter/src/runtime.mm").read_text()
        fd=runtime.split("const auto encodeFEMOperatorFDTrace =",1)[1].split("const auto capturedNodeCaptureSlots",1)[0]
        self.assertIn("numi::matter::detail::femPrimaryOperatorCacheMode(",fd)
        self.assertIn("state.femAlgorithmicTangentCacheValid : state.dummy",fd)
        self.assertNotIn("const std::uint32_t useCache = 0u;",fd)
        self.assertEqual(runtime.count("numi::matter::detail::femPrimaryOperatorCacheMode("),5)
    def test_compiled_policy_preserves_primary_semantics(self):
        code='#include "numi/matter/detail/fem_primary_operator_cache_mode.hpp"\\nusing numi::matter::detail::femPrimaryOperatorCacheMode;\\nstatic_assert(femPrimaryOperatorCacheMode(0)==0);\\nstatic_assert(femPrimaryOperatorCacheMode(1)==1);\\nstatic_assert(femPrimaryOperatorCacheMode(2)==0);\\nstatic_assert(femPrimaryOperatorCacheMode(0xffffffffu)==0);\\nint main(){}\\n'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"check.cpp";p.write_text(code)
            subprocess.run(["/usr/bin/clang++","-std=c++20","-I",str(ROOT/"matter/include"),str(p),"-o",str(Path(d)/"check")],check=True)
            subprocess.run([str(Path(d)/"check")],check=True)
if __name__=="__main__":unittest.main()
""")
newnames=set(f['source_sha256'])|{header,test}
g={**f,'base_snapshot':str(base),'base_frozen_sha256':sha(base/'frozen-source.json'),'source253_scope':'FD observer matches primary cache selection; no change to primary real-number operator, material recipe or tolerances','source_sha256':{n:sha(s/n) for n in sorted(newnames)}}
(s/'frozen-source.json').write_text(json.dumps(g,indent=2,sort_keys=True)+'\n')
changed=[n for n in newnames if n not in f['source_sha256'] or sha(s/n)!=f['source_sha256'][n]]
patch=[]
for n in sorted(changed):
 a=base/n;b=s/n
 r=subprocess.run(['diff','-u','--label','a/'+n,'--label','b/'+n,str(a) if a.exists() else '/dev/null',str(b)],capture_output=True,text=True)
 assert r.returncode in (0,1);patch.append(r.stdout)
(d/'source253.patch').write_text(''.join(patch))
report={'schema':'numi.cardboard.source253-primary-cache-fd-action.v1','base':str(base),'source':str(s),'base_frozen_sha256':sha(base/'frozen-source.json'),'frozen_sha256':sha(s/'frozen-source.json'),'changed_files':sorted(changed),'shared_primary_policy_call_sites':5,'primary_policy_real_number_behavior_unchanged':True,'FD_observer_numerical_action_changed_to_actual_primary_cache_mode':True,'historical_captures_must_not_qualify_cached_production_action':True,'native_validation_pending':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(d/'manifest.json').write_text(json.dumps({'inputs':{str(base/'frozen-source.json'):sha(base/'frozen-source.json')},'artifacts':{str(p):sha(p) for p in [s/'frozen-source.json',d/'report.json',d/'source253.patch']}},indent=2)+'\n')
print(json.dumps(report))
