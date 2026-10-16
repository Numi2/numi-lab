from pathlib import Path
import json,hashlib,shutil,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/publication039-reproduction-extension-002';d.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();files={}
def put(p,n):
 p=Path(p);assert p.is_file() and not p.is_symlink(),p;z=d/n;z.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,z);assert sha(z)==sha(p);files[str(n)]={'external_path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
for n in ['adjudicate_native573_parity_001.py','run_native574_cohesive_pages_001.py','run_native575_cohesive_pages_fixed_001.py','build_source234_upstream_compatibility_001.py','archive_publication039_001.py']:put(q/n,Path('commands')/n)
s=q/'snapshots/source231-upstream-compatibility-234-001'
for n in ['frozen-source.json','configure-support-source.json']:put(s/n,Path('source234')/n)
b=q/'build-source234-upstream-compatibility-001';j=json.loads((b/'source-build.json').read_text());assert j['success'];put(b/'source-build.json',Path('source234/build/source-build.json'))
for p in b.glob('provenance-command-*.log'):put(p,Path('source234/build')/p.name)
for p,h in j['artifacts_sha256'].items():assert sha(p)==h;put(p,Path('source234/build')/Path(p).relative_to(b))
for version in ['001','002']:
 name='source-production-cohesive-pages-227-'+version;s=q/'snapshots'/name;build=q/('build-'+name);package=q/'analysis'/name
 put(s/'frozen-source.json',Path(name)/'frozen-source.json');put(package/'source-build.json',Path(name)/'source-build.json');j=json.loads((package/'source-build.json').read_text())
 for n,a in j['artifacts'].items():assert sha(build/n)==a['sha256'];put(build/n,Path(name)/'build'/n)
p=q/'analysis/source224-550-observer-pair-adjudication-tool-001/report.json';put(p,Path('observer-pair/retained-report.json'))
(d/'README.md').write_text("""Reference039 extension

Source234 fresh full integrated build and all nine CPU compatibility checks pass on main938e3034. All selected source231 inputs remain byte-exact; two Human-only upstream files are included and the changed configure-support header is rebound. Native573 separately completes the tiny seven-control history with all32production files byte-identical to210.

Retain native574's failed cohesive helper. Source227-002 repairs only helper bind-before-dispatch order and immutable page fixture buffers; native575 passes the actual production accumulation/apply kernel helper. This remains zero FEM transactions, not a complete paged nonlinear or shipping-world solve.

Native566/567 production outputs and stdout match exactly, while567 has only an ambiguous recorder failure and no usable sidecar. The retained550 raw before binary differs by eight bytes; do not call that cross-source raw parity. No first dispatch attribution is available.
""")
report={'schema':'numi.cardboard.publication039-reproduction-extension.v1','files':files,'native575_result':'PASS_BOUNDED_KERNEL_HELPER_ONLY','all_six_full_sequence_axes_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');(d/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file()}},indent=2)+'\n');print(json.dumps({'files':len(files),'report_sha256':sha(d/'report.json')}))
