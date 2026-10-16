from pathlib import Path
import json,hashlib,shutil,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/publication039-reproduction-001';d.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();files={}
def put(p,n):
 p=Path(p);assert p.is_file() and not p.is_symlink(),p;target=d/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);assert sha(p)==sha(target);files[str(n)]={'external_path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
scripts=['compose_build_source226_001.py','compose_build_source228_001.py','compose_build_source228_002.py','compose_build_source228_003.py','compose_build_source228_004.py','prepare_build_source229_001.py','compose_build_source230_001.py','build_source231_upstream_compatibility_001.py','run_native555_control19_capture_001.py','run_native558_strict_projection_001.py','run_native559_562_components_002.py','run_native563_strict_projection_fixed_001.py','run_native564_control19_small_epsilon_001.py','run_native565_control19_small_epsilon_001.py','run_native566_first_failure_off_001.py','run_native567_first_failure_on_001.py','run_native568_570_composed230_components_001.py','run_native571_composed230_regression_001.py','run_native573_composed231_regression_001.py','adjudicate_native555_parity_001.py','run_native555_cached_operator_001.py','adjudicate_native565_parity_001.py','adjudicate_native565_fine_descriptive_001.py','adjudicate_native571_parity_001.py']
for n in scripts:put(q/n,Path('commands')/n)
snapshots=['source-pure-fem-armijo-cgls-arena-215-002','source216-integrated-configure-support-006','source-local-projection-state-certification-220-endpoint-identity-repair-003','source-first-failure-dispatch-224-001','source-shipping-contact-pages-225-002','source-composed-pure-8192-on222-226-001','source-composed-pure-proxy-228-004','source-diagnostic-epsilon-replay-229-001','source-composed-pure-proxy-replay-230-001','source230-upstream-compatibility-231-001']
for n in snapshots:
 s=q/'snapshots'/n;put(s/'frozen-source.json',Path('source-maps')/(n+'.json'))
 if (s/'configure-support-source.json').exists():put(s/'configure-support-source.json',Path('source-maps')/(n+'-configure-support.json'))
builds=['build-source-pure-fem-armijo-cgls-arena-215-provenance-002','build-source216-integrated-006','build-source224-first-failure-002','build-source-shipping-contact-pages-225-004','build-source226-pure-arenas-001','build-source228-integrated-004','build-source229-diagnostic-epsilon-001','build-source230-integrated-001','build-source231-upstream-compatibility-001']
for n in builds:
 b=q/n;put(b/'source-build.json',Path('builds')/n/'source-build.json');j=json.loads((b/'source-build.json').read_text());assert j['success']
 for p in b.glob('provenance-command-*.log'):put(p,Path('builds')/n/p.name)
 if n in ['build-source224-first-failure-002','build-source-shipping-contact-pages-225-004','build-source229-diagnostic-epsilon-001','build-source231-upstream-compatibility-001']:
  for p,h in j['artifacts_sha256'].items():
   assert sha(p)==h;put(p,Path('builds')/n/Path(p).relative_to(b))
j=json.loads((q/'analysis/source-local-projection-state-certification-220-003/report.json').read_text())
for a in j['build']['artifacts'].values():
 p=Path(a['path']);assert sha(p)==a['sha256'];put(p,Path('builds')/'build-source220-endpoint-identity-repair-003'/p.relative_to(Path(j['build']['path'])))
(d/'README.md').write_text("""Reference039 executable reproduction

Canonical source231 composes source230 checked pure-FEM8192 workspaces, exact prescribed-BOX/capsule/free-BOX admission and bounded diagnostic epsilon replay controls with the three disjoint Human files already on main. Source230 completed the tiny seven-control all-phase regression with all32production files exactly matching source210; source231 has a separate fresh build and regression. These are bounded component/regression results, not full coupon or shipping-box convergence.

Original545 and actual diagnostic555/565 retain the same control19 failure; complete resumed histories and rejected state compare exactly. Fine native finite differences and tighter FP64 endpoint targets remain inconclusive. Strict local-state candidate220 fixes ten targeted material tangent rows in563, while its full native coupon remains a separate future experiment. Source224's567 recorder failed to produce a usable sidecar; terminal status does not attribute a kernel.

Shipping ordinal pages225 are a planner/component check only. Full cohesive production paging227 and contact paging, full-box arena admission, physical scoring/assembly, capacity and impact qualification remain open. Retain all failed attempts and bind every result to its own source map, binary and input package. Reproductions under different absolute paths require new source-bound commands and receipts.
""")
report={'schema':'numi.cardboard.publication039-reproduction.v1','files':files,'canonical_source':'source231','unqualified_excluded':['source227 active production paging','source232 active strict-policy integration','source233 active recorder repair'],'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');(d/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file()}},indent=2)+'\n');print(json.dumps({'files':len(files),'report_sha256':sha(d/'report.json')}))
