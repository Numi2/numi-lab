"""Bounded same-count connected star repair; construction only, full final audits mandatory."""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import star_candidates569 as star
from cached_counter569 import CachedCounter
N=Path(__file__).resolve().parent.parent;D=N.parent
def load(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jwrite(p,o):Path(p).write_text(json.dumps(o,indent=2,sort_keys=True,allow_nan=False)+"\n")
def canontri(t):
 a=tuple(map(int,t));return min(a,a[1:]+a[:1],a[2:]+a[:2])
def run(a):
 out=Path(a.output);out.mkdir(exist_ok=False,parents=True);start=time.perf_counter();rejected=Counter();events=[];ops=[]
 def event(stage,**kw):
  e={"stage":stage,"seconds":time.perf_counter()-start,**kw};events.append(e)
  with (out/"progress.jsonl").open("a")as f:f.write(json.dumps(e,sort_keys=True)+"\n")
  print(json.dumps(e,sort_keys=True),flush=True)
 source_paths=[Path(__file__),Path(star.__file__),Path(sys.modules[CachedCounter.__module__].__file__),Path(a.payload_helper)]
 source_pins={str(p):sha(p)for p in source_paths}
 state_path=Path(a.state);s=json.loads(state_path.read_text())
 for k in ("candidate","surface_report","point_summary"):
  if sha(s[k])!=s[k+"_sha256"]:raise ValueError("state pin mismatch "+k)
 base_path=Path(s["candidate"]);base_sha=s["candidate_sha256"]
 with np.load(base_path,allow_pickle=False)as z:base={k:np.array(z[k],copy=True)for k in z.files}
 current={k:np.array(v,copy=True)for k,v in base.items()}
 x=np.asarray(base["points_world_m"]);f=np.asarray(base["source_O_faces"]);ref=base["area_reference_037_points_world_m_f64"]
 if x.dtype!=np.float32 or not np.array_equal(f,base["area_reference_037_source_O_faces"]):raise ValueError("emission dtype/reference mismatch")
 h=json.loads(Path(s["point_summary"]).read_text());ar=json.loads(Path(s["surface_report"]).read_text())
 exact_sha="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d"
 if h.get("candidate_sha256")!=base_sha or h.get("complete_surface_audit")is not True or h.get("exact_PLC_surface_pairs")!=0 or h.get("surface_predicate_sha256")!=exact_sha:raise ValueError("complete zero-self input required")
 if np.load(Path(s["point_summary"]).parent/"surface-self-pairs.npy").shape!=(0,2):raise ValueError("nonzero self-pair file")
 helper=load("payload569",a.payload_helper)
 m=load("owners025",N/"topology/topology-bone-aware-025.py")
 for k,p in [("owner370",m.OWNER370),("owner402",m.OWNER402),("check408",m.CHECK408),("search409",m.SEARCH409),("owner541",m.OWNER541)]:
  if sha(p)!=m.EXPECTED[k]:raise ValueError("owner pin mismatch "+k)
  source_pins[str(p)]=sha(p)
 sel=load("sel569",m.OWNER370);_,surface_owner,bank_helper,exact159,runtime=sel.load_pinned_runtime()
 raw=bank_helper.read_bank_inputs(sel.BANK_DIR_DEFAULT,sel.EVIDENCE_ROOT_DEFAULT);banks,keys=sel.typed_banks_from_159_inputs(raw)
 pairs,targets,counts=sel._pair_rows_and_counts(ar,candidate_sha256=base_sha,face_count=len(f),banks=banks,expected_bank_keys=keys)
 pairs={k:set(v)for k,v in pairs.items()};initial=sum(counts.values())
 original=sel.Exact159PatchCounter(x,banks,exact159,surface_owner.records);counter=CachedCounter(original)
 owner402=load("owner402569",m.OWNER402);q=load("q569",m.OWNER541)
 policy=load("policy569",N/"topology-policy-root-v6/quality_policy025_variant.py")
 source_pins[str(Path(policy.__file__))]=sha(policy.__file__)
 checker=load("check569",m.CHECK408);search=load("search569",m.SEARCH409)
 if sha(search.PRED)!=exact_sha:raise ValueError("predicate source mismatch")
 ex=search.exact_owner();records,_=ex.float32_surface_records(x,f,source_seam_groups=());records=ex._immutable_self_pair_records(records)
 frozenx=x.copy();frozenf=f.copy();frozenx.setflags(write=False);frozenf.setflags(write=False)
 selfstate=checker.State(ex,frozenx,frozenf,records,ex._prepare_surface_aabb(records),frozenset(),(),base_sha,exact_sha)
 anchors=set(map(int,base["typed_coarse_anchor_output_rows_018"]))
 if len(anchors)!=237:raise ValueError("anchor inventory")
 # Validate cache algorithm against the immutable exact owner on a real crossing sample.
 sample=sorted(targets)[:16];fast=counter.count_faces(sample,f[sample]);slow=original.count_faces(sample,f[sample])
 if fast!=slow:raise ValueError("cached AABB counter parity failure")
 event("prepared",pairs=initial,parity_sample_faces=len(sample),banks=len(keys))
 tri_count={};quality_cache={};exact_calls=0;quality_calls=0;deadline=False
 points_mm=x.astype(np.float64)*1000;ref_mm=np.asarray(ref,dtype=np.float64)*1000
 def qualities(nf):
  missing={canontri(t):t for t in nf if canontri(t)not in quality_cache}
  if missing:
   kk=list(missing);ff=np.asarray([missing[k]for k in kk],dtype=np.int64)
   vals=q.face_quality(points_mm,ref_mm,ff,ff)
   for j,k in enumerate(kk):quality_cache[k]={v:float(vals[v][j])for v in policy.QUALITY_KEYS}
  return {v:np.asarray([quality_cache[canontri(t)][v]for t in nf])for v in policy.QUALITY_KEYS}
 def exact_counts(nf):
  nonlocal exact_calls
  vals=np.zeros(len(keys),dtype=np.int64)
  for t in nf:
   # Exact cross-surface intersections are orientation invariant.
   key=tuple(sorted(map(int,t)))
   if key not in tri_count:
    rr=counter.count_faces(np.asarray([0],dtype=np.int64),np.asarray([t],dtype=np.int64));exact_calls+=1
    tri_count[key]=np.asarray([rr[k]["count"]for k in keys],dtype=np.int64)
   vals+=tri_count[key]
  return vals
 rounds=[]
 for round_no in range(1,a.rounds+1):
  f=current["source_O_faces"];em=owner402.edge_map(f);incidence=defaultdict(list)
  for row,t in enumerate(f):
   for v in t:incidence[int(v)].append(row)
  targets={r for ps in pairs.values()for r,_ in ps};face_counts={k:Counter(r for r,_ in ps)for k,ps in pairs.items()}
  vertex_scores=Counter()
  for row in targets:
   for v in f[row]:vertex_scores[int(v)]+=sum(face_counts[k].get(row,0)for k in keys)
  centers=sorted(vertex_scores,key=lambda v:(-vertex_scores[v],v))[:a.max_centers];used_nodes=set();round_ops=0
  current_quality=q.face_quality(points_mm,ref_mm,f,f)
  event("round_started",round=round_no,centers=len(centers),pairs=sum(map(len,pairs.values())))
  for order,center in enumerate(centers):
   if time.perf_counter()-start>a.seconds:deadline=True;break
   if center in anchors:rejected["anchored_center"]+=1;continue
   rows=sorted(incidence[center]);nodes=set(map(int,current["source_O_faces"][rows].flat))
   if nodes&used_nodes:rejected["overlap_in_batch"]+=1;continue
   if not 4<=len(rows)<=a.max_degree:rejected["degree_outside_bound"]+=1;continue
   if len(np.unique(current["face_kind"][rows]))!=1:rejected["mixed_material"]+=1;continue
   try:candidates=star.star_candidates(x,ref,current["source_O_faces"],rows,center,a.max_degree)
   except ValueError:rejected["non_disk_star"]+=1;continue
   old=np.asarray([sum(face_counts[k].get(r,0)for r in rows)for k in keys],dtype=np.int64)
   before={v:current_quality[v][rows]for v in policy.QUALITY_KEYS}
   eligible=[]
   for nf in candidates:
    if star.external_edge_conflict(current["source_O_faces"],rows,nf,em):rejected["external_edge"]+=1;continue
    qual,why=policy.evaluate_quality_arrays(q,before,qualities(nf));quality_calls+=1
    if qual is None:rejected["quality_"+why]+=1;continue
    eligible.append((nf,qual))
   eligible.sort(key=lambda v:(-v[1]["after"]["minimum_angle_deg"],-v[1]["after"]["minimum_normal_cosine"],tuple(v[0].flat)))
   scored=[]
   for nf,qual in eligible[:a.max_candidates_per_center]:
    if time.perf_counter()-start>a.seconds:deadline=True;break
    new=exact_counts(nf)
    if np.any(new>old)or int(new.sum())>=int(old.sum()):rejected["exact_no_bankwise_reduction"]+=1;continue
    scored.append((int(old.sum()-new.sum()),nf,qual,new))
    if not np.any(new):break
   scored.sort(key=lambda p:(-p[0],-p[2]["after"]["minimum_angle_deg"],tuple(p[1].flat)))
   for reduction,nf,qual,new in scored:
    # Re-evaluate the entire selected patch using both implementations.
    fast=counter.count_faces(rows,nf);slow=original.count_faces(rows,nf)
    if fast!=slow or not np.array_equal(new,np.asarray([slow[k]["count"]for k in keys])):raise ValueError("selected patch exact-cache parity")
    op={"face_rows":rows,"new_faces":nf.tolist()}
    trial=helper.apply_patch(current,op)
    inc=checker.evaluate(selfstate,trial["points_world_m"],trial["source_O_faces"])
    if inc.report["introduced_pairs"]:rejected["exact_self_intersection"]+=1;continue
    owner402.validate_candidate(trial)
    if any(len(v)>2 for v in owner402.edge_map(trial["source_O_faces"]).values()):raise ValueError("nonmanifold edge")
    selfstate=checker.commit(selfstate,inc,require_reduction=False,candidate_sha256=base_sha)
    op.update(center=center,round=round_no,old_faces=current["source_O_faces"][rows].tolist(),quality=qual,
      exact159_delta={"old":int(old.sum()),"new":int(new.sum()),"reduction":reduction,"per_bank":[{"body":k[0],"bone":k[1],"old":int(old[j]),"new":int(new[j])}for j,k in enumerate(keys)if old[j]or new[j]],"cache_matches_original370":True},
      self_screen=inc.report)
    ops.append(op);current=trial;used_nodes.update(nodes);round_ops+=1
    for k in keys:pairs[k]={p for p in pairs[k]if p[0]not in rows}|set(map(tuple,slow[k]["triangle_pairs"]))
    event("accepted_patch",round=round_no,center=center,degree=len(rows),reduction=reduction,pairs=sum(map(len,pairs.values())),accepted=len(ops))
    break
   if (order+1)%16==0:event("scan_progress",round=round_no,centers_examined=order+1,accepted=round_ops,exact_triangle_calls=exact_calls,rejections=dict(rejected))
   if deadline:break
  rounds.append({"round":round_no,"accepted":round_ops,"pairs":sum(map(len,pairs.values())),"deadline":deadline})
  if not round_ops or deadline:break
 for p,hsh in source_pins.items():
  if sha(p)!=hsh:raise ValueError("source changed during run "+p)
 # Verify all coordinates, identities and point-domain data are preserved.
 if not m.identical(current["points_world_m"],base["points_world_m"]):raise ValueError("coordinates changed")
 if set(np.asarray(current["source_O_faces"]).flat)!=set(np.asarray(base["source_O_faces"]).flat):raise ValueError("used point set changed")
 for key in base:
  if key.startswith("source_P") or key.startswith("source_Q") or key.startswith("source_prolongation") or key.startswith("typed_coarse_anchor"):
   if not m.identical(current[key],base[key]):raise ValueError("point/anchor source changed "+key)
 np.savez_compressed(out/"candidate.npz",**current)
 report={"status":"CONNECTED_STAR_CONSTRUCTION_PENDING_FULL_AUDITS"if ops else"NO_PASSING_CONNECTED_STAR",
   "input_state":str(state_path),"input_state_sha256":sha(state_path),"input_candidate":str(base_path),"input_candidate_sha256":base_sha,
   "output_candidate":str(out/"candidate.npz"),"output_candidate_sha256":sha(out/"candidate.npz"),
   "before_pairs":initial,"after_pairs_composed":sum(map(len,pairs.values())),"reduction":initial-sum(map(len,pairs.values())),
   "rounds":rounds,"accepted_operations":ops,"rejections":dict(rejected),"source_pins":source_pins,"runtime_source_pins":runtime,
   "exact_triangle_calls":exact_calls,"quality_proposal_count":quality_calls,"seconds":time.perf_counter()-start,
   "coordinates_unchanged":True,"point_ids_unchanged":True,"typed_anchors":237,"typed_banks":185,
   "self_pair_count_incremental":len(selfstate.pairs),"native_admission":False,"accepted_physical_steps":0,
   "limitations":["Construction only; full130/159, refreshed affine ancestry and skin/fat physical compiler remain mandatory.","Pair counts for unchanged faces reuse the pinned complete base audit.","No motion, mass, physical contact or native qualification claimed."]}
 jwrite(out/"report.json",report);event("complete",accepted=len(ops),before_pairs=initial,after_pairs=report["after_pairs_composed"])
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--state",required=True);p.add_argument("--output",required=True);p.add_argument("--payload-helper",required=True)
 p.add_argument("--rounds",type=int,default=4);p.add_argument("--seconds",type=float,default=160);p.add_argument("--max-centers",type=int,default=1000)
 p.add_argument("--max-degree",type=int,default=9);p.add_argument("--max-candidates-per-center",type=int,default=128)
 run(p.parse_args())
if __name__=="__main__":main()

