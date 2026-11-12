from pathlib import Path
import collections,gzip,hashlib,json,math
A=Path("artifacts/crow-v15-ground-impulse-attribution-20261112");B=Path("artifacts/crow-v15-tiny-swing-toe-screen-20261112");values={};reports=[]
def load(path,observer):
 hashes={k:hashlib.sha256() for k in ("CROW_SUPPORT_SAMPLE", "CROW_TWO_TOE_SUBSTEP")};counts=collections.Counter();samples={};gx={};errors=[]
 with gzip.open(path,"rt") as f:
  for line in f:
   if "CROW_SUBSTEP_ERROR" in line:errors.append(line.strip())
   key=line.split(" ",1)[0]
   if key in hashes:
    hashes[key].update(line.encode());counts[key]+=1
    if key=="CROW_SUPPORT_SAMPLE":
     x=json.loads(line.split(" ",1)[1]);samples[x["lane"],x["step"]]=x
    else:
     z=line.split()
     if int(z[4])>int(z[5]):errors.append("overflow")
   elif key=="CROW_GROUND_X":
    z=line.split();t,k,e=map(int,z[1:4]);
    if 1<=t<=300:
     assert (e,t,k) not in gx;gx[e,t,k]=list(map(float,z[4:]));assert len(gx[e,t,k])==4
 return {"hashes":{k:v.hexdigest() for k,v in hashes.items()},"counts":dict(counts),"samples":samples,"ground":gx,"errors":errors}
def endpoint_contact(x):
 out=[0.,0.,0.,0.]
 for c in x["contacts"]:
  aa=c["a_name"];bb=c["b_name"];imp=c["impulses"]
  if imp[0]<=0:continue
  name=aa if bb=="locomotion_ground" else bb if aa=="locomotion_ground" else "";sgn=-1 if bb=="locomotion_ground" else 1
  side=0 if name=="crow_left_foot" or name.startswith("crow_left_") and "toe" in name else 1 if name=="crow_right_foot" or name.startswith("crow_right_") and "toe" in name else None
  if side is None:continue
  j=sgn*sum(c[k][0]*imp[i] for i,k in enumerate(("normal","tangent_u","tangent_v")));out[side]+=j
  if "rear_toe" in name:out[2+side]+=j
 return out
for name,d,ref in (("sham","sham-20261127-observer","sham-20261127-dev2"),("candidate","candidate-20261127-rebound","candidate-20261127-rebound")):
 observed=load(A/d/"run.stderr.gz",True);retained=load(B/ref/"run.stderr.gz",False);assert observed["hashes"]==retained["hashes"] and observed["counts"]==retained["counts"] and not observed["errors"] and not retained["errors"]
 gx=observed["ground"];assert len(gx)==8*300*16
 errs=[]
 for e in range(8):
  for t in range(201,301):
   v=endpoint_contact(observed["samples"][e,t]);w=gx[e,t,15];errs.extend(abs(a-b) for a,b in zip(v,w))
 maxerr=max(errs);assert maxerr<1e-6,maxerr
 values[name]=gx;reports.append({"arm":name,"exact_retained_sample_and_substep_hashes":observed["hashes"],"counts":observed["counts"],"ground_records":len(gx),"maximum_endpoint_reconstruction_error_Ns":maxerr})
rows=[]
for e in range(8):
 out={"lane":e}
 for n in values:
  g=values[n];ts=[]
  for t in range(201,301):
   impulses=[sum(g[e,t,k][i] for k in range(16)) for i in range(4)];ts.append({"interval":t,"left_x_Ns":impulses[0],"right_x_Ns":impulses[1],"left_rear_x_Ns":impulses[2],"right_rear_x_Ns":impulses[3]})
  out[n]={"summed_x_Ns":[sum(x[k] for x in ts) for k in ("left_x_Ns","right_x_Ns","left_rear_x_Ns","right_rear_x_Ns")],"intervals":ts}
 out["net_ground_x_difference_Ns"]=sum(out["candidate"]["summed_x_Ns"][:2])-sum(out["sham"]["summed_x_Ns"][:2]);rows.append(out)
result={"validation":reports,"rows":rows,"scope":"Substep contact impulse of ground on modeled feet, one previously used development seed and two-second interval. Rear-toe impulse is a subset of foot total. Does not include non-ground loads; paired differences do not establish a unique causal mechanism."};(A/"analysis.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"validation":reports,"net_difference_Ns":[x["net_ground_x_difference_Ns"] for x in rows]},indent=2))
