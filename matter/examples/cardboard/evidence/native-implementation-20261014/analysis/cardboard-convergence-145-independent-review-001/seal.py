import hashlib,json,pathlib,re
B=pathlib.Path('/Users/home/cardboard-implementation-20261014')
PKG=B/'analysis/cardboard-convergence-tool-overlay-145-001'
REV=B/'analysis/cardboard-convergence-145-independent-review-001'
SRC=B/'snapshots/source-cgls-primary-policy-143'
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
pkg_manifest=json.loads((PKG/'manifest.json').read_text())
manifest=json.loads((PKG/'report.json').read_text())
source_hashes=manifest['source143_sha256']
overlay_hashes=manifest['overlay_sha256']
patch_sha=manifest['patch_check']['patch_sha256']
assert manifest['schedule_contract']['default_post_release_steps']==0
assert manifest['schedule_contract']['post_release_requires_release'] is True
assert manifest['schedule_contract']['released_nodes'].startswith('right grip remains free')
assert manifest['schedule_contract']['shipping_settle'].startswith('separate shipping')
assert manifest['source_contract_checks']['option_default_zero']
assert manifest['source_contract_checks']['source143_did_not_gain_new_option']
assert manifest['source_contract_checks']['all_phase_arguments_are_controlled']
assert manifest['source_contract_checks']['strict_decrease_preserved']
assert manifest['patch_check']['applies_to_source143_snapshot'] and manifest['patch_check']['patch_applied_output_matches_overlay']
checks=[]
for name in ('convergence','resolution','admission','shipping'):
 text=(REV/(name+'.log')).read_text()
 match=re.search(r'Ran (\d+) tests?',text)
 assert match and ('OK' in text)
 checks.append({'suite':name,'test_count':int(match.group(1)),'exit_code':0,'log_sha256':sha(REV/(name+'.log'))})
assert sum(x['test_count'] for x in checks)==68
report={
 'schema':'numi.cardboard.post_release_145_independent_review.v1',
 'verdict':'PASS_SOFTWARE_SCOPE_ONLY',
 'overlay':'post-release-relaxation end-grip schedule/tooling overlay',
 'review':{
  'default_compatibility':{
   'new_count_default':0,
   'total_steps_add_post_count':True,
   'when_count_zero':'old loading/hold/unload/relax/release step sequence remains; no post-release accepted steps are added',
   'metadata_note':'manifest now emits the explicit zero count, so serialization shape changes even when trajectory schedule does not',
   'legacy_analyzer':'missing release/post-release counts default to zero and do not infer unrecorded histories'
  },
  'phase_and_boundary':{
   'phase_order':['loading','hold','unloading','relaxation','release','post_release_relaxation'],
   'post_phase_requires_positive_release':True,
   'release_and_post_release_right_grip_free':True,
   'left_grip_remains_fixed':True,
   'post_phase_rejected_for_shipping_crease_and_pin_adhesion_modes':True,
   'shipping_uses_existing_released_assembly_settle':True,
   'phase_endpoint_state_write':'each accepted step writes material state at final step or phase boundary; rejected steps preserve diagnostic state output'
  },
  'history_and_analysis_gates':{
   'native_analyzer_requires_completed_status_and_exact_accepted_step_count':True,
   'all_per_step_environment_rows_and_acceptance_certificates_required':True,
   'explicit_requested_post_phase_missing_in_either_run_is_inconclusive':True,
   'free_grip_constraint_and_fixed_node_count_checked':True,
   'strictly_decreasing_resolution_differences_preserved':True,
   'complete_resolution_matrix_requires_positive_counts_for_all_six_phases':True,
   'retained_failed_runs_cannot_pass_from_accepted_prefix':True
  },
  'patch':{'source143_hashes_match_package_manifest':True,'patch_applies_and_recreates_overlay':True,'patch_sha256':patch_sha,'changed_paths':manifest['patch_check']['changed_paths']},
 },
 'cpu_validation':{'tests':checks,'total_tests':68,'all_passed':True},
 'limits':['No native compile or GPU run for this overlay.','Synthetic-reader fixtures test software/history contracts only.','Numerical convergence, absolute force accuracy, held-out material prediction, physical coupon validation, and physical box validation remain unqualified.'],
 'blocking_findings':[]
}
(REV/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
readme='''# Independent review of the post-release relaxation overlay\n\n**Verdict: PASS_SOFTWARE_SCOPE_ONLY.** The default post-release count remains zero. With that count, the existing end-grip step sequence has no additional post-release steps; the new manifest explicitly serializes the zero count. A positive count is bounded, requires a positive release phase, and is refused for shipping, crease, and pin-adhesion modes. Release and post-release relaxation both keep the right grip free while retaining the left grip constraint. Shipping continues to use its separately existing settle phase.\n\nThe reader defaults absent release/post-release counts in legacy manifests to zero, verifies complete status, exact accepted-step count, per-step environment observations and certificates, and phase/boundary metadata. An explicitly requested post-release comparison is inconclusive if either history lacks that phase. The six-phase resolution planner demands positive counts for every phase, and retains strict-decrease tests and failed-run controls.\n\nAll 68 CPU tests pass (19 convergence, 13 resolution, 10 admission/schedule, 26 unchanged source-143 shipping-stage tests). The patch is bound to source143 and dry-runs/recreates all five overlay files. No native compile, GPU run, convergence qualification, or physical qualification is claimed.\n'''
(REV/'README.md').write_text(readme)
inputs={}
for rel,h in source_hashes.items():
 p=SRC/rel
 assert p.is_file() and sha(p)==h,rel
 inputs[str(p)]=h
for rel,h in overlay_hashes.items():
 p=PKG/rel
 assert p.is_file() and sha(p)==h,rel
 inputs[str(p)]=h
for rel in ('post-release-relaxation.patch','run_audit.py','report.json','manifest.json'):
 p=PKG/rel
 inputs[str(p)]=sha(p)
for log in REV.glob('*.log'): inputs[str(log)]=sha(log)
files={p.name:sha(p) for p in REV.iterdir() if p.is_file() and p.name not in ('manifest.json','seal.py')}
seal={'schema':'numi.cardboard.post_release_145_independent_review_manifest.v1','review_report_sha256':sha(REV/'report.json'),'readme_sha256':sha(REV/'README.md'),'seal_script_sha256':sha(REV/'seal.py'),'overlay_package_manifest_sha256':sha(PKG/'manifest.json'),'bound_inputs':inputs,'files':files,'test_total':68,'verdict':'PASS_SOFTWARE_SCOPE_ONLY'}
(REV/'manifest.json').write_text(json.dumps(seal,indent=2,sort_keys=True)+'\n')
print(json.dumps({'review':str(REV),'manifest_sha256':sha(REV/'manifest.json'),'overlay_manifest_sha256':sha(PKG/'manifest.json'),'patch_sha256':patch_sha,'tests':68,'verdict':report['verdict']},indent=2))
