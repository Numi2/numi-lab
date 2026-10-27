#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,json,numpy as np
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def beq(a,b):return a.dtype==b.dtype and a.shape==b.shape and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()
def ids(p,m=None):
 r=read(p);assert r['complete'] and r['bank_count']==r['expected_bank_count']==185
 out=set()
 for bank in r['banks']:
  key=(int(bank['body']),int(bank['stable_bone_id']));q=bank.get('result') or {'count':0,'triangle_pairs':[]};assert q['count']==len(q['triangle_pairs'])
  for f,t in q['triangle_pairs']:out.add((key[0],key[1],int(m[int(f)]) if m is not None else int(f),int(t)))
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--d-root',default='/Users/numi/Work/numi-human-connected-surface-069/D');ap.add_argument('--out');a=ap.parse_args();D=Path(a.d_root);R=D/'surface-clearance-546/hard-face-feasibility-634'
 base=D/'publication-174/evidence/box-aware-sat-fit-637/run-001-from766/merit-631-001/candidate.npz';base159=D/'publication-174/evidence/box-aware-sat-fit-637/run-001-from766/merit-631-001/full-check/audit-159/report.json';p638=R/'local-refinement-638/from-764-001'
 statepaths=[p638/'state.json',p638/'joint-fit-001/restored-retention-582-001/next-714-001/retention-582-001/strict-from-refined/state.json',R/'fresh-fit-from715-001/retention-582-001/final/state.json',R/'fresh-fit-from703-001/retention-582-001/final/state.json',R/'fresh-fit-from703-001/merit-631-001/final/state.json']
 labels=['638','715','703','701','697'];out=[];cands=[];reports=[];points=[]
 for label,sp in zip(labels,statepaths):
  s=read(sp);c=Path(s['candidate']);assert sha(c)==s['candidate_sha256'] and not s['native_admission'] and s['accepted_physical_steps']==0
  if label=='638':pr=p638/'audit-130/summary.json';sr=p638/'audit-159/report.json';fr=p638/'self-arithmetic/report.json'
  else:
   pr=c.parent/'full-check/audit-130/summary.json';sr=c.parent/'full-check/audit-159/report.json';fr=sp.parent/'report.json'
   assert s['point_summary_sha256']==sha(pr) and s['surface_report_sha256']==sha(sr)
  q=read(pr);r=read(sr);assert q['candidate_sha256']==r['candidate_sha256']==sha(c) and q['complete_surface_audit'] and r['complete']
  assert r['bank_count']==r['expected_bank_count']==185 and q['exact_PLC_surface_pairs']==0
  inside=2 if label=='638' else 0;assert q['inside_node_bank_pairs']==inside
  if label!='638':
   f=read(fr);assert f['status']=='COMPLETE_EXACT_CONSTRUCTION_CHECKS' and f['candidate_sha256']==sha(c)
  pairs=sum((b.get('result') or {}).get('count',0) for b in r['banks'])
  if label=='638':assert pairs==770
  cands.append(c);reports.append(sr);points.append(pr)
  out.append({'label':label,'state_path':str(sp),'state_sha256':sha(sp),'candidate_path':str(c),'candidate_sha256':sha(c),'surface_report_sha256':sha(sr),'point_report_sha256':sha(pr),'final_report_sha256':sha(fr),'surface_pairs':pairs,'contained_pairs':q['inside_node_bank_pairs'],'self_pairs':q['exact_PLC_surface_pairs'],'banks':r['bank_count']})
 with np.load(base,allow_pickle=False) as bz,np.load(p638/'candidate.npz',allow_pickle=False) as z638,np.load(cands[1],allow_pickle=False) as z715,np.load(cands[2],allow_pickle=False) as z703,np.load(cands[3],allow_pickle=False) as z701,np.load(cands[4],allow_pickle=False) as z697:
  m=z697['refinement_face_parent_face_row'];assert beq(m,z715['refinement_face_parent_face_row']) and beq(m,z703['refinement_face_parent_face_row']) and beq(m,z701['refinement_face_parent_face_row']) and beq(m,z638['refinement_face_parent_face_row'])
  assert np.all(m>=0) and np.all(m<len(bz['source_O_faces']))
  keys=set(z715.files);assert keys==set(z703.files)==set(z701.files)==set(z697.files)
  assert {k for k in keys if not beq(z715[k],z697[k])}=={'points_world_m'}
  anchors=[k for k in keys if 'anchor' in k.lower()];assert anchors and all(beq(z715[k],z697[k]) for k in anchors)
  assert z697['typed_coarse_anchor_output_rows_018'].shape==(237,)
 parent=ids(base159);raw=[ids(p) for p in reports[1:]];mapped=[ids(p,m) for p in reports[1:]]
 assert len(parent)==764 and [len(x) for x in raw]==[715,703,701,697]
 assert (len(raw[0]-raw[1]),len(raw[1]-raw[0]))==(30,18)
 assert (len(raw[1]-raw[3]),len(raw[3]-raw[1]))==(13,7)
 assert [len(x) for x in mapped]==[707,695,695,691]
 assert (len(parent-mapped[3]),len(mapped[3]-parent))==(202,129)
 b701=read(cands[3].parent/'final/report.json')['per_bank_deltas'];b697=read(cands[4].parent/'final/report.json')['per_bank_deltas']
 assert b701=={'43/16':2,'57/89':-4,'59/95':-2,'65/99':2};assert b697=={'57/89':-4,'59/95':-2}
 a703=read(cands[2].parent/'final/report.json')['full_f32_area_gate'];a697=read(cands[4].parent/'final/report.json')['full_f32_area_gate'];assert a703['pass'] and a697['pass'] and a703['area_floor']==a697['area_floor']==0.1
 result={'schema':'numi.human.connected_surface_p175_independent_verification.v1','status':'PASS_CONSTRUCTION_EVIDENCE_ONLY','p174_manifest_sha256':sha(D/'publication-174/publication-manifest.json'),'baseline_764_candidate_sha256':sha(base),'refinement_face_count':len(m),'states':out,'pair_identity_comparison':{'baseline_parent_pairs':len(parent),'mapped_distinct_715_703_701_697':[len(x) for x in mapped],'same_topology_removed_715_to_703':30,'same_topology_introduced_715_to_703':18,'same_topology_removed_703_to_697':13,'same_topology_introduced_703_to_697':7,'mapped_764_to_697_removed':202,'mapped_764_to_697_introduced':129},'701_per_bank_deltas':b701,'697_per_bank_deltas':b697,'area_gate_minima':{'703_current':a703['minimum_current_oriented_ratio'],'703_fixed037':a703['minimum_fixed_037_reference_magnitude_ratio'],'697_current':a697['minimum_current_oriented_ratio'],'697_fixed037':a697['minimum_fixed_037_reference_magnitude_ratio'],'floor':0.1},'limitations':['638 intermediate has two contained node-bank pairs and is not a strict endpoint.','Single rest-pose checks do not qualify closed-solid containment, volume cells, mass/thickness, dynamics, multiple poses or native execution.','The fixed-037 area ratio for 703/697 is about 0.1000096383 against the 0.1 floor.','All states have native_admission=false and accepted_physical_steps=0.']}
 text=json.dumps(result,indent=2,sort_keys=True)+'\n'
 if a.out:Path(a.out).write_text(text)
 print(text)
if __name__=='__main__':main()
