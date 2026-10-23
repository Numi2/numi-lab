#!/usr/bin/env python3
from pathlib import Path
import re,json,hashlib,numpy as np
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001";HERE=Path(__file__).resolve().parent
CURRENT=E/"native-ground-aligned-placement-008/clean-main-app-headers/apps/NumiHumanWholeBodySkinReduction.hpp"
OLD=Path("/Users/n/numi-human-separation-20261018/C/wholebody-integration-295/build-wholebody-016/source/apps/NumiHumanWholeBodySkinReduction.hpp")
D17=E/"elbow-refinement-current-main-017-single-residual-prolongation-001/apps/NumiHumanWholeBodySkinReduction.hpp"
NPZ=E/"ankle-volume-restoration-fresh017/candidate-layers-017.npz"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def vecs(s,n):
 z=s.split(n+"=",1)[1].split(";",1)[0];return np.array([[float(v.strip().removesuffix("f")) for v in x.split(",")] for x in re.findall(r"(?:std::array<double,3>)?\{([^{}]+)\}",z)])
def parse(p):
 s=p.read_text();fr=re.findall(r"WholeBodySkinDonorFrameV1\{(\d+)u,\{([^{}]+)\},\{([^{}]+)\},\{([^{}]+)\},\{([^{}]+)\}\}",s)
 nums=lambda t:[float(v.strip().removesuffix("f")) for v in t.split(",")]
 O=np.array([nums(x[3]) for x in fr])+vecs(s,"kWholeBodySkinMechanicalWorldDeltas")
 z=s.split("kWholeBodySkinFaces=",1)[1].split(";",1)[0];F=np.array([[int(v) for v in x] for x in re.findall(r"Face\s*\{\s*(\d+)u?\s*,\s*(\d+)u?\s*,\s*(\d+)u?\s*\}",z)])
 return O,vecs(s,"kWholeBodySkinInnerWorld"),vecs(s,"kWholeBodyFatDeepWorld"),F

def quality(O,I,D,F):
 n=len(O);a,b,c=np.sort(F,axis=1).T
 ts=np.concatenate([np.stack((a,b,c,n+c),1),np.stack((a,b,n+b,n+c),1),np.stack((a,n+a,n+b,n+c),1)],axis=0)
 tf=np.concatenate([np.stack((n+a,n+b,n+c,2*n+c),1),np.stack((n+a,n+b,2*n+b,2*n+c),1),np.stack((n+a,2*n+a,2*n+b,2*n+c),1)],axis=0)
 X=np.concatenate((O,I,D)).astype(np.float32).astype(float);T=np.concatenate((ts,tf));p=X[T];det=np.einsum("ij,ij->i",np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),p[:,3]-p[:,0]);longest=np.max(np.stack([np.linalg.norm(p[:,i]-p[:,j],axis=1) for i in range(4) for j in range(i+1,4)]),axis=0)
 normalized=np.abs(det)/longest**3;bad=np.flatnonzero(normalized<1e-5)
 return {"tet_count":len(T),"absolute_normalized_det_min":float(normalized.min()),"below_1e_5_count":len(bad),"note":"Sorted prism connectivity; absolute determinants measure quality, not orientation admission. Native owner remains authoritative.","witnesses":[{"tet":int(i),"face":int(i%len(F)),"material":"skin" if i<len(ts) else "fat","normalized_abs_det":float(normalized[i]),"nodes":T[i].tolist()} for i in bad[:10]]}
def diff(a,b):
 d=np.linalg.norm(a-b,axis=1);return {"rows":len(a),"bit_exact_float64":bool(np.array_equal(a,b)),"float32_changed_rows":int(np.sum(np.any(a.astype(np.float32)!=b.astype(np.float32),axis=1))),"error_m_quantile":[float(v) for v in np.percentile(d,[50,95,100])],"rows_over_10um":int(np.sum(d>1e-5)),"maximum_row":int(np.argmax(d))}
def main():
 assert sha(CURRENT)=="60735d7457e2a8737d60e20cf155aa2fda4f97478072ddef44346bbdd903fe68" and sha(OLD)=="a0e6d69706611257556c74ab7815ca2605f1f3c5f5530d9dd3472ecf482f5200"
 cur=parse(CURRENT);old=parse(OLD);d17=parse(D17);a=np.load(NPZ)
 assert np.array_equal(cur[3],old[3])
 comparisons={name:diff(cur[i],old[i]) for i,name in enumerate(["O","I","D"])}
 inherited={name:diff(cur[i],d17[i][:2332]) for i,name in enumerate(["O","I","D"])}
 candidate={name:diff(cur[i],a[key]) for i,(name,key) in enumerate([("O","outer_world_m"),("I","skin_inner_world_m"),("D","fat_deep_world_m")])}
 for i in [1,2]:assert np.array_equal(cur[i],d17[i][:2332])
 out={"source_hashes":{str(p):sha(p) for p in [CURRENT,OLD,D17,NPZ,Path(__file__)]},"conclusion":"Ankle NPZ generation used older295 rest layers; current native607 layers differ despite the same294 surface payload. This is a frozen-rest-baseline mismatch; no attribution to evolved endpoint data is established.","current_vs_old_rest_layers":comparisons,"current_vs_D17_old_rows":inherited,"current_vs_candidate017":candidate,"quality_current607":quality(*cur),"quality_old295":quality(*old),"quality_candidate017":quality(a["outer_world_m"],a["skin_inner_world_m"],a["fat_deep_world_m"],a["faces"]),"scope":"Read-only preparation comparison, not physical or anatomy qualification."}
 dest=HERE/"rest-layer-baseline-consistency.json";assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:v for k,v in out.items() if k.startswith("quality") or k=="current_vs_old_rest_layers"}))
if __name__=="__main__":main()
