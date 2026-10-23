from pathlib import Path
import re,json,hashlib,csv,math
r=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
e=r/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
p=e/"joint-layer-frame-consistency-066";p.mkdir(exist_ok=True)
f=e/"elbow-refinement-current-main-016-owner-orientation-005-001/apps/NumiHumanWholeBodySkinReduction.hpp";s=f.read_text()
def arr(name):
 a=s.split(name+"=",1)[1].split(";",1)[0]
 return [[float(v.strip().removesuffix("f")) for v in x.split(",")] for x in re.findall(r"(?:std::array<double,3>)?\{([^{}]+)\}",a)]
frames=[]
a=s.split("kWholeBodySkinDonorFrames=",1)[1].split(";",1)[0]
for row in re.findall(r"WholeBodySkinDonorFrameV1\{(\d+)u,\{([^{}]+)\},\{([^{}]+)\},\{([^{}]+)\},\{([^{}]+)\}\}",a):
 frames.append([int(row[0])]+[[float(x.removesuffix("f")) for x in a.split(",")] for a in row[1:]])
wd,ld,iw,il,dw,dl=[arr(n) for n in ["kWholeBodySkinMechanicalWorldDeltas","kWholeBodySkinMechanicalDonorLocalDeltas","kWholeBodySkinInnerWorld","kWholeBodySkinInnerLocal","kWholeBodyFatDeepWorld","kWholeBodyFatDeepLocal"]]
assert all(len(a)==2411 for a in [frames,wd,ld,iw,il,dw,dl])
rows=[]
for n,(donor,o,m,w,normal) in enumerate(frames):
 finalo=[w[k]+wd[n][k] for k in range(3)];finall=[o[k]+ld[n][k] for k in range(3)]
 def local(point):return [finall[k]+sum(m[3*k+j]*(point[j]-finalo[j]) for j in range(3)) for k in range(3)]
 ei=math.dist(il[n],local(iw[n]));ed=math.dist(dl[n],local(dw[n]));eg=math.dist(ld[n],[sum(m[3*k+j]*wd[n][j] for j in range(3)) for k in range(3)])
 rows.append({"row":n,"donor":donor,"inner_relation_error_m":ei,"deep_relation_error_m":ed,"mechanical_delta_frame_error_m":eg})
def stats(a,key):
 v=sorted(x[key] for x in a);w=max(a,key=lambda x:x[key]);return {"count":len(v),"p50_m":v[len(v)//2],"p95_m":v[int((len(v)-1)*.95)],"max_m":v[-1],"witness":w}
out={"source_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"scope":"Frozen D16 layer-local relation audit, not current-source registration or physical motion; float frame literals parsed as emitted decimal values.","old_2332":{k:stats(rows[:2332],k) for k in rows[0] if k.endswith("_m")},"projected_79":{k:stats(rows[2332:],k) for k in rows[0] if k.endswith("_m")}}
(p/"d16-wholebody-layer-frame-relations.json").write_text(json.dumps(out,indent=2)+"\n")
with (p/"d16-wholebody-layer-frame-relations.csv").open("w") as g:
 w=csv.DictWriter(g,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(out))
