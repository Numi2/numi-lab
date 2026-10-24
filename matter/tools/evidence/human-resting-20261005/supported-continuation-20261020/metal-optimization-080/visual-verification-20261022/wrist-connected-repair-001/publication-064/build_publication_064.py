from pathlib import Path
import json,hashlib,shutil,sys,subprocess,os
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");R=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
P=E/'publication-064'
assert not P.exists()
P.mkdir();D=P/'payload';D.mkdir()
files=set()
def add(rel):
 p=E/rel
 assert p.is_file(),p
 files.add(p)
def tree(rel):
 p=E/rel; assert p.is_dir(),p
 for f in p.rglob('*'):
  if f.is_file() and not any(x in f.parts for x in ['__pycache__','.pytest_cache','build','build-001']) and f.suffix in ['.py','.json','.npz','.txt','.log']:
   assert f.stat().st_size<8_000_000,(f,f.stat().st_size)
   files.add(f)
for f in ['root-outer-escape-direction-016.json','root_construct_outward_bone_exits_017.py','root_construct_surface_normal_bone_exits_018.py','root_run_geometry_only_native_query_023.py','root_verify_static_query_023.py','root-independent-static-query-023.json']:add(f)
for f in ['outward-skin-bone-exits-017','surface-normal-bone-exits-018','layer-depth-o12-reconstruction-007-run-002','o12-outward-repair-019','o12-outward-repair-021','o12-outward-repair-022','route-audit-consolidation-019','o12-outward-repair-024','o12-outward-repair-024-lp-feasibility-001']:tree(f)
Q='native-source-only-rest-geometry-022-query-023'
for f in ['source-query-report.json','checked-runner.stdout.txt','checked-runner.stderr.txt','base-launch-manifest.json','command.json','candidate-node-override.csv']:add(Q+'/'+f)
for f in ['execution-receipt.json','launch-manifest.snapshot.json','stdout.log','stderr.log','all-bank-containment.csv','all-bank-containment.csv.nodes.csv','all-bank-containment.csv.banks.csv','exact-emitted.nodes.csv','exact-emitted.bodies.csv']:add(Q+'/source-preflight/'+f)
records=[]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
for f in sorted(files):
 rel=f.relative_to(E);out=D/rel;out.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(f,out)
 assert sha(f)==sha(out)
 records.append({'path':rel.as_posix(),'sha256':sha(out),'size_bytes':out.stat().st_size})
(P/'payload-files-064.json').write_text(json.dumps({'files':records},indent=2)+'\n')
meta={'schema':'numi.human.publication-manifest.v1','publication':'064','base_origin_main':'79dbbd107b36813cf9c9c84c4f92f8f9c821b479','native_or_physical_admission':False,'accepted_physical_steps':0,'status':'READY_FOR_ROOT_REVIEW_NOT_PUSHED','payload_file_count':len(records),'payload_bytes':sum(r['size_bytes']for r in records),'payload_file_list_sha256':sha(P/'payload-files-064.json'),'candidate_022':{'local_exit_plane_constraints_pass':20,'exact_route_hit_segment_bank_pairs':16,'proper_hits':20,'inside_node_bank_pairs':17,'invalid_or_ambiguous_node_banks':0,'reference_mass_admission':False},'audit_corrections':['Initial FEM node export is pre-override baseline; exact-emitted candidate nodes own 023 query.','New consolidated audit accidentally used a bank-local triangle index against the global array for reported normals/vertices. Corrected source reconstructs actual hit barycentrics. Counts unchanged.','Original root041/010/015 and direct017/018 used correct bank-local indexing.','Pre-fix consolidated source bytes 1e0575/e4dfe2 are unavailable; retained report source hashes do not imply byte recoverability.'],'inherited_dependencies':{'parent':'79dbbd107b36813cf9c9c84c4f92f8f9c821b479','bank_archive_publication':'061','baseline_depth_source_publication':'063','duplicate_146MB_triangle_export_not_copied':True}}
r24=json.loads((D/'o12-outward-repair-024/report.json').read_text())
meta['candidate_024_report_sha256']=sha(D/'o12-outward-repair-024/report.json')
meta['candidate_024_as_run_status']=r24.get('status')
(P/'publication-manifest-064.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))

