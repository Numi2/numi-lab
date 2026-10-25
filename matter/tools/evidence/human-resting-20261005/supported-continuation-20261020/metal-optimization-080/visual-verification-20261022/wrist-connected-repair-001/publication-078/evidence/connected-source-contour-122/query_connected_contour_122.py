from pathlib import Path
import json,hashlib,subprocess,csv,collections
D=Path('/Users/numi/Work/numi-human-connected-surface-069');E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
C=D/'connected-source-contour-122/run-001/candidate.npz';Q=C.parent/'query';assert not Q.exists();Q.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
py='/Users/n/numi-human-prep-venv-20261005/bin/python';bank=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight';b=Path('/tmp/generic-query-overlay-build-042-work/matter/numi-matter-human-skin-fat-static-query')
assert sha(b)=='5dc7b5ce2a96f58bd03ccc85e38645d9da2ec8be105a532c461a3e0cc6d25460'
commands=[[py,str(E/'E/generic-static-query-042/prepare_generic_query_points.py'),'--candidate-npz',str(C),'--input-sha256',sha(C),'--field','points_world_f32','--output',str(Q/'points.csv')],[str(b),str(bank/'exact-emitted.triangles.csv'),str(bank/'exact-emitted.bodies.csv'),str(Q/'points.csv'),str(Q/'pairs.csv'),str(Q/'nodes.csv'),'185']]
for i,cmd in enumerate(commands):
 cp=subprocess.run(cmd,capture_output=True,text=True);(Q/f'command-{i}.log').write_text(cp.stdout+cp.stderr);assert cp.returncode==0
rows=list(csv.DictReader((Q/'pairs.csv').open()));ns=list(csv.DictReader((Q/'nodes.csv').open()));assert len(ns)==7360 and all(int(r['ambiguous_banks'])==0 and int(r['invalid_banks'])==0 for r in ns)
inside=[r for r in rows if r['classification']=='inside'];groups=collections.Counter('OID'[int(r['query_row'])//2411]if int(r['query_row'])<7233 else'generated'for r in inside)
report={'candidate_sha256':sha(C),'commands':commands,'binary_sha256':sha(b),'points_sha256':sha(Q/'points.csv'),'pairs_sha256':sha(Q/'pairs.csv'),'nodes_sha256':sha(Q/'nodes.csv'),'inside_node_bank_pairs_by_layer':dict(groups),'inside_nodes':len({int(r['query_row'])for r in inside}),'inside_witnesses':inside,'accepted_physical_steps':0}
(Q/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in('commands','inside_witnesses')},indent=2))
