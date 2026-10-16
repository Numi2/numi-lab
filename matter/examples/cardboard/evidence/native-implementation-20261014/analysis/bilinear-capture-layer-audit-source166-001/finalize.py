from pathlib import Path
import difflib, hashlib, json, os, subprocess, sys
import numpy as np

B=Path('/Users/home/cardboard-implementation-20261014')
P=B/'analysis/bilinear-capture-layer-audit-source166-001'
S=B/'snapshots/source-right-grip-telemetry-157'
T=P/'candidate/matter/tools'
V=P/'validation'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(T)+':'+str(S/'matter/tools'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
command=[sys.executable,'-m','unittest','test_cardboard_bilinear_capture_layers']
with (V/'contract-tests.stdout.log').open('x') as out,(V/'contract-tests.stderr.log').open('x') as err:
    result=subprocess.run(command,env=env,stdout=out,stderr=err)
assert result.returncode==0
old=np.load(B/'analysis/source125-full-tangent-fp64-audit-147-001/matrices.npz')
current=np.load(V/'regression419-004/arrays.npz')
assert np.array_equal(current['native_C'],old['native_C_Pa'])
error=np.linalg.norm(current['reference_C']-old['independent_C_Pa'],axis=(1,2))
scale=np.linalg.norm(old['independent_C_Pa'],axis=(1,2))
delta=float(np.max(error/scale));assert delta<=1e-12,delta
report=json.loads((V/'regression419-004/report.json').read_text())
assert report['bindings']['capture_run_fingerprints_match']
assert report['states'][0]['cached_material_matrix_comparison_Pa']['all_rows_at_relative_gate']
assert report['fp64_directional_sweep']['all_rows_have_resolved_pair']
assert not any(report['six_qualification_axes'].values())
assert all(0<s['projection']['valid_rows']<1728 for s in report['states'][1:])
assert report['fp64_directional_sweep']['native_branch_derivative_qualified'] is False
for index in range(1,5):
    relation=report['states'][index]['realized_endpoint_vs_ideal_F_perturbation']
    assert relation['max_relative_l2']<4e-5
validation={'schema':'numi.cardboard.capture-layer-reader-validation.v1','contracts_exit_code':result.returncode,'contracts_command':command,'reference_147_matrices_sha256':sha(B/'analysis/source125-full-tangent-fp64-audit-147-001/matrices.npz'),'regression004_manifest_sha256':sha(V/'regression419-004/manifest.json'),'native_C_exactly_matches_147':True,'independent_C_max_relative_matrix_change_vs_147':delta,'native_matrix_column_relative_error_max':report['states'][0]['cached_material_matrix_comparison_Pa']['max_relative_l2'],'all_reference_directions_have_adjacent_fp64_pair':True,'independent_realized_endpoint_projection_failures_retained':[1728-s['projection']['valid_rows'] for s in report['states'][1:]],'warning_count_retained':len(report['warnings']),'native_execution_performed':False,'qualification':False}
(V/'report.json').write_text(json.dumps(validation,indent=2)+'\n')
(P/'README.md').write_text('''# Bilinear capture layer reader166

This default-off CPU diagnostic reads a source-bound two-environment, contact-free, zero-gravity, rate-independent kraft/NSSC/starch coupon capture. It validates all recorded source/binary/input hashes through the existing diagnostic reader, both environments\' complete node/tet identities, capture selection and world/execution fingerprints, tangent layout, positive masses, constraint tags and topology. It rejects damage/cohesive/other material laws rather than reusing this oracle for them.

The independent FP64 law is copied exactly from267; the resilient projector and tightened residual/correction checks retain the previously evaluated147 algorithms. `reference-provenance.json` binds their actual source. The only helper dispatch change makes the seed-projector call local to the new module. No production compiler, bytecode or local projector is used in the independent constitutive evaluation.

At center and each captured FP32 high+low endpoint it reprojects from the same accepted history, retains independent residual/correction/iteration failures, evaluates native-state residuals, stress and inferred complementarity predicates, then compares all nine cached material columns, frame actions, element actions and assembled nodal action. Mass and stiffness terms remain separate under K=-df/dx and A=M+dt²K. Realized endpoint changes are explicitly compared with ±epsilon*dt*grad(delta_v); their quantization error is recorded.

The bounded ideal FP64 sweep independently checks the captured paper direction on its own reference branch, preserving all failed rows and requiring two adjacent declared grid points at relative1e-6. Branch-preserving one-sided actions are reported separately. Inferred lambda/yield signs at tiny margins are not native internal branch traces; native/reference predicate disagreement is retained, with raw margins, and never claimed to prove a physical active-set transition. All warnings retain their messages and source locations. This does not measure a native FP32 absolute noise floor, qualify a native smooth branch, or establish nonlinear/contact/mesh/time convergence.

Regression419-004 reproduces every native cached matrix exactly and every independent147 matrix to relative1e-12 or better. Its largest native material-column error is6.135899309e-5; all1728 independent paper directions have a branch-preserving adjacent FP64 pair. The four realized native endpoints have18/14/14/14 unresolved independent-reference projections; those rows remain inconclusive. Prototype001/002/003 sources/results are retained rather than rewritten. The final contracts cover missing/duplicate identities, fingerprint mismatch, epsilon bounds, unresolved near-zero rows and shared-face force cancellation. No new Matter step or GPU run was executed for this reader package; all six scientific qualification axes remain open.

Run after a source-bound capture has terminated, with the tools directory on PYTHONPATH:

    python3 cardboard_bilinear_capture_layers.py --case CASE --output FRESH_OUTPUT

The final output includes complete arrays, row validity, failure reasons, warnings, hashes and separate local/assembled comparisons. The existing diagnostic input binder also supports the separately validated161 diagnostic-run schema when integrated with that overlay.
''')
files=sorted(p for p in T.iterdir() if p.is_file())
patch=''.join(''.join(difflib.unified_diff([],p.read_text().splitlines(keepends=True),fromfile='/dev/null',tofile='b/matter/tools/'+p.name)) for p in files)
(P/'overlay.patch').write_text(patch)
# New-file-only patch must match every proposed byte on exact source157.
import tempfile
with tempfile.TemporaryDirectory(prefix='cardboard166-patch-',dir=B) as tmp:
    tree=Path(tmp);(tree/'matter/tools').mkdir(parents=True)
    applied=subprocess.run(['patch','--batch','-p1','-i',str(P/'overlay.patch')],cwd=tree,capture_output=True,text=True)
    (V/'patch-apply.stdout.log').write_text(applied.stdout);(V/'patch-apply.stderr.log').write_text(applied.stderr)
    assert applied.returncode==0
    for p in files:assert (tree/'matter/tools'/p.name).read_bytes()==p.read_bytes()
(P/'finalize.py').write_bytes(Path(__file__).read_bytes())
(P/'source-overlay.json').write_text(json.dumps({'schema':'numi.cardboard.source-overlay.v1','base_snapshot':str(S),'overlay_id':'bilinear-capture-layer-audit-source166-001','patch_sha256':sha(P/'overlay.patch'),'files':[{'path':'matter/tools/'+p.name,'base_sha256':None,'overlay_sha256':sha(p)} for p in files],'native_execution_performed':False,'qualification':False},indent=2)+'\n')
(P/'manifest.json').write_text(json.dumps({'schema':'numi.cardboard.analysis-files.v1','files_sha256':{str(p.relative_to(P)):sha(p) for p in sorted(P.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}},indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(P/'manifest.json'),'patch_sha256':sha(P/'overlay.patch'),'validation':validation}))
