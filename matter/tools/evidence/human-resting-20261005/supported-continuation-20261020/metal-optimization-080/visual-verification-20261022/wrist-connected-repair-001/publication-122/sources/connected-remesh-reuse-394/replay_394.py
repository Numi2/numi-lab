#!/usr/bin/env python3
"""Replay the verified 368/378/383 connected flips on pinned391 coordinates."""
from pathlib import Path
import hashlib,json,importlib.util,subprocess,sys
import numpy as np
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 owner=D/'D/connected-remesh-reuse-381/replay_381.py'
 assert sha(owner)=='1eee4ccddb2db4545e7e46db22a9bd42b94a966d17051070d99e0d4dc2922ad2'
 state=json.loads((H/'coherent-support-continuation-391/run-001/selected-result.json').read_text());inp=Path(state['candidate'])
 assert sha(inp)==state['candidate_sha256']=='958bd94f4e4589aa7b25265daa4e9a3c9d589f68a26939e9ed5257a0e712c888'
 base383=H/'bulk-remesh-383/run-001/round-001';parent383=base383/'self-safe-subset-01.npz'
 assert sha(parent383)=='292888c5beefb481bfea35028c724661cf1cbd5f324c425b7a0fd78b20411683'
 opsreport=base383/'self-safe-subset-01-358-report.json';ops=json.loads(opsreport.read_text())['operations'];assert len(ops)==9
 out=H/'combined-clearance-394/run-001';out.mkdir(parents=True,exist_ok=False)
 cmd=[PY,str(owner),'--candidate-375',str(inp),'--expected-candidate-375-sha256',sha(inp),'--output-dir',str(out/'stage-381')]
 result=subprocess.run(cmd,capture_output=True,text=True);(out/'stage-381-execution.json').write_text(json.dumps(dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n');assert result.returncode==0,result.stderr
 spec=importlib.util.spec_from_file_location('verified_replay381_for394',owner);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 before=m._load_npz(out/'stage-381/candidate.npz');expected=m._load_npz(parent383)['source_O_faces'];flip=m.import_flip()
 after,report=m.replay(before,ops,sorted({int(r)for op in ops for r in op['face_rows']}),flip,expected,'383-round1')
 first=json.loads((out/'stage-381/replay-381-report.json').read_text());allops=first['stage_368']['operations']+first['stage_378']['operations']+report['operations']
 after['same_parent_flip_349_face_rows']=np.asarray([r for op in allops for r in op['face_rows']],dtype=np.int32)
 after['same_parent_flip_349_pair_face_rows']=np.asarray([op['face_rows']for op in allops],dtype=np.int32).reshape(-1,2)
 after['same_parent_flip_349_parent_rows']=np.asarray([op['parent_row']for op in allops],dtype=np.int64)
 assert after['points_world_m'].tobytes()==m._load_npz(inp)['points_world_m'].tobytes()
 with(out/'candidate.npz').open('xb')as f:np.savez_compressed(f,**after)
 receipt=dict(status='PINNED_391_COORDINATES_WITH_REVALIDATED_383_FLIP_CHAIN_PROPOSAL',input_391=state['candidate'],input_391_sha256=sha(inp),pinned383_sha256=sha(parent383),pinned383_operations_sha256=sha(opsreport),owner381_sha256=sha(owner),stage383=report,aggregate_operation_count=len(allops),candidate_sha256=sha(out/'candidate.npz'),coordinates_391_bitwise_unchanged=True,topology_matches383=np.array_equal(after['source_O_faces'],expected),all_185_and_self_and_area_checks_required=True,native_admission=False,accepted_physical_steps=0)
 (out/'replay-394-report.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items()if k!='stage383'}))
if __name__=='__main__':main()
