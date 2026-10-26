"""Restore only exact newly-contained witness rows from a pinned outside reference.
Construction only; full arithmetic/self/all185 checks are still mandatory.
"""
from pathlib import Path
import argparse,json,hashlib,numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reference-state',required=True,type=Path);ap.add_argument('--trial-result',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
 st=json.loads(a.reference_state.read_text());tr=json.loads(a.trial_result.read_text());b=Path(st['candidate']);c=Path(tr['candidate']);assert sha(b)==st['candidate_sha256'];before=json.loads(Path(st['point_summary']).read_text());after=json.loads(Path(tr['point_summary']).read_text());assert before['candidate_sha256']==sha(b)and after['candidate_sha256']==sha(c)
 for r in [before,after]:assert r['complete_surface_audit']and r['exact_PLC_surface_pairs']==0
 key=lambda r:(int(r['output_node']),int(r['bank_body']),int(r['bank_stable_id']))
 old={key(r)for r in before['inside_witnesses']};new={key(r)for r in after['inside_witnesses']}-old;nodes=np.unique([q[0]for q in new]).astype(np.int64);assert len(nodes)and not any(q[0]in nodes for q in old)
 z=np.load(c,allow_pickle=False);ref=np.load(b,allow_pickle=False);assert np.array_equal(z['source_O_faces'],ref['source_O_faces']);assert not np.isin(nodes,z['typed_coarse_anchor_output_rows_018']).any();out={k:z[k]for k in z.files};out['points_world_m']=out['points_world_m'].copy();out['points_world_m'][nodes]=ref['points_world_m'][nodes]
 a.out.mkdir(exist_ok=False);p=a.out/'candidate.npz';np.savez_compressed(p,**out);r=dict(status='OUTSIDE_WITNESS_RESTORE_PENDING_EXACT_CHECKS',candidate_sha256=sha(p),parent_candidate_sha256=sha(c),point_source_candidate_sha256=sha(b),introduced_typed_pairs=sorted(new),construction_protection_rows=nodes.tolist(),source_sha256=sha(__file__),reference_state_sha256=sha(a.reference_state),trial_result_sha256=sha(a.trial_result),native_admission=False,accepted_physical_steps=0);(a.out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
