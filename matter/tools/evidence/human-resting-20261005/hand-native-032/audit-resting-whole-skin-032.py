from pathlib import Path
import sys,json,hashlib,time,collections
import numpy as np
E=Path('/Users/n/numi-human-resting-evidence-20261005');R=E/'source017-release-stability-031'
sys.path.insert(0,'/Users/n/numi-human-resting-launcher-source-001/src')
sys.path.insert(0,'/Users/n/numi-human-resting-vascular-structure-source-013/matter/tools')
from numilab_human.skin_source_payload_preflight import decode_payload
from numilab_human.surface_topology_audit import exact_embedding
from numilab_human.compiled_quotient_embeddedness import coordinate_quotient
import accepted_mrvpack_surface_audit as a
skinpath=Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/skin-boundaries-001/bodyparts3d-myosim-skinned-shell.nhskin')
skin=decode_payload(skinpath.read_bytes());owners=skin['bindings_u'][:,0][skin['full_weights'].argmax(axis=1)]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
step=int(sys.argv[2]);p=R/'accepted-geometry'/f'step-{step}.mrvpack'
out=R/f'whole-skin-step-{step}.json';assert not out.exists()
raw,st,off,s=a.read_pack(p);receipt=a.validate_accepted_receipt(p,p.with_suffix('.receipt.json'),step,raw,off,s)
f=np.asarray(s[(51007,1)]['faces']);lo=int(f.min());hi=int(f.max());v=np.ndarray((hi-lo+1,20),dtype='<f4',buffer=raw,offset=off+lo*80)[:,:3].copy();f=f-lo
assert np.array_equal(f,skin['indices'].reshape(-1,3));raw.close();st.close()
t=time.monotonic();qv,qf,_=coordinate_quotient(v.tolist(),f.tolist());exact=exact_embedding(qv,qf)
dominant=owners[f];hand=((dominant>=45)&(dominant<=76)|(dominant>=95)&(dominant<=126)).any(axis=1)
hand_pairs=[p for p in exact['triangle_pairs'] if hand[p[0]] or hand[p[1]]]
other_pairs=[p for p in exact['triangle_pairs'] if not (hand[p[0]] or hand[p[1]])]
report={'scope':'Complete exact outer-skin self-pair audit of one accepted native state. Dominant-owner hand involvement is diagnostic classification, never an exemption. This does not assess internal organs or skin material strain.',
 'script_sha256':sha(__file__),'skin_sha256':sha(skinpath),'native_pack_sha256':sha(p),'accepted_receipt':receipt,'step':step,
 'exact':exact,'hand_involved_pair_count':len(hand_pairs),'other_pair_count':len(other_pairs),'other_pairs':other_pairs,
 'other_pair_dominant_body_ids':[[sorted(set(map(int,dominant[i]))) for i in pair] for pair in other_pairs[:64]],'elapsed_s':time.monotonic()-t}
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['step','hand_involved_pair_count','other_pair_count','elapsed_s']}),flush=True)
