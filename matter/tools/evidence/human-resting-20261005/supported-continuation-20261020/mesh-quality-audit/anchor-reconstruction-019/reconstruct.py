#!/usr/bin/env python3
"""CPU-only reproduction of old/current whole-body skin-fat support anchors."""
import argparse, hashlib, json, math, re, struct, subprocess
from pathlib import Path

COMMIT="c927871bafc1194f8ef314fc0687e4d123d11dd8"
HEADER="apps/NumiHumanWholeBodySkinReduction.hpp"
ASSET=Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin")
FLOAT=re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[fF]?")

def hbytes(b): return hashlib.sha256(b).hexdigest()
def hfile(p): return hbytes(Path(p).read_bytes())
def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def block(t,n):
    m=re.search(re.escape(n)+r"\s*=\s*\{",t)
    if not m: raise ValueError("missing array "+n)
    e=[p for p in (t.find("\n};",m.end()),t.find("\n}};",m.end())) if p>=0]
    if not e: raise ValueError("unclosed array "+n)
    return t[m.end():min(e)]
def nums(s): return [float(x.rstrip("fF")) for x in FLOAT.findall(s)]
def ints(t,n): return [int(x.rstrip("uU")) for x in re.findall(r"\d+u?",block(t,n))]
def tri(t,n):
    rows=[tuple(nums(x)) for x in re.findall(r"std::array<double,3>\s*\{([^{}]+)\}",block(t,n))]
    if any(len(x)!=3 for x in rows): raise ValueError("bad triples "+n)
    return rows
def header(t):
    faces=[tuple(map(int,x)) for x in re.findall(r"Face\{\s*(\d+)u,\s*(\d+)u,\s*(\d+)u\s*\}",block(t,"kWholeBodySkinFaces"))]
    src=ints(t,"kWholeBodySkinSourceVertices")
    pat=re.compile(r"WholeBodySkinDonorFrameV1\{\s*\d+u,\s*\{([^{}]+)\},\s*\{([^{}]+)\},\s*\{([^{}]+)\},\s*\{([^{}]+)\}\s*\}")
    reg=[tuple(nums(m.group(3))) for m in pat.finditer(block(t,"kWholeBodySkinDonorFrames"))]
    delta=tri(t,"kWholeBodySkinMechanicalWorldDeltas"); interface=tri(t,"kWholeBodySkinInnerWorld"); deep=tri(t,"kWholeBodyFatDeepWorld")
    n=len(src)
    if len(faces)!=4619 or n!=2332 or any(len(x)!=n for x in (reg,delta,interface,deep)): raise ValueError("unexpected reduction dimensions")
    outer=[tuple(reg[i][a]+delta[i][a] for a in range(3)) for i in range(n)]
    return faces,src,outer,interface,deep
def skin(path):
    b=Path(path).read_bytes()
    magic,abi,nb,nv,ni,fp,srcsha=struct.unpack_from("<8s5I32s",b)
    if abi!=5: raise ValueError(f"expected ABI5, got {abi} {magic!r}")
    wo=60+nb*36+nv*56+ni*4
    if len(b)!=wo+nv*nb*4: raise ValueError("skin payload byte size mismatch")
    bodies=[struct.unpack_from("<I",b,60+i*36)[0] for i in range(nb)]
    w=struct.unpack_from(f"<{nv*nb}f",b,wo)
    rows=[w[i*nb:(i+1)*nb] for i in range(nv)]
    return bodies,rows,hbytes(b)
def sub(a,b): return tuple(a[i]-b[i] for i in range(3))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def dot(a,b): return sum(a[i]*b[i] for i in range(3))
def norm(a): return math.sqrt(dot(a,a))
def quality(p,ids):
    q=[p[i] for i in ids]
    six=abs(dot(cross(sub(q[1],q[0]),sub(q[2],q[0])),sub(q[3],q[0])))
    alts=[]
    for k in range(4):
        f=[q[i] for i in range(4) if i!=k]
        area=norm(cross(sub(f[1],f[0]),sub(f[2],f[0])))
        if not math.isfinite(area) or area<=0: raise ValueError("degenerate tet face")
        alts.append(six/area)
    v=min(alts)
    if not math.isfinite(v) or v<=0: raise ValueError("degenerate tet")
    return v
def reconstruct(parsed,bodies,weights):
    faces,src,outer,interface,deep=parsed; n=len(src); p=outer+interface+deep
    inc=[math.inf]*(3*n)
    for face in faces:
        a,b,c=sorted(face)
        for tet in ((n+a,n+b,n+c,2*n+c),(n+a,n+b,2*n+b,2*n+c),(n+a,2*n+a,2*n+b,2*n+c)):
            q=quality(p,tet)
            for node in tet: inc[node]=min(inc[node],q)
    by={}
    low=0
    for local,sv in enumerate(src):
        row=weights[sv]
        bi=max(range(len(row)),key=lambda i:row[i]) # first maximum matches C++ strict >
        body,conf=bodies[bi],max(row)
        alt=inc[2*n+local]
        if alt<5e-5: low+=1; continue
        by.setdefault(body,[]).append({"local_vertex":local,"source_vertex":sv,"donor_body_index":body,
            "min_incident_altitude_m":alt,"max_weight_confidence":conf,"_point":outer[local]})
    for c in by.values(): c.sort(key=lambda x:(-x["min_incident_altitude_m"],-x["max_weight_confidence"],x["local_vertex"]))
    selected=[]; per={}; budget=(3*n)//32
    for rnd in range(3):
        for body in sorted(by):
            if len(selected)>=budget: break
            chosen=per.setdefault(body,[])
            if len(chosen)>rnd or len(by[body])<=rnd: continue
            for c in by[body]:
                if c["max_weight_confidence"]<.85: continue
                pt=c["_point"]
                if all(norm(sub(pt,old))>=.025 for old in chosen):
                    selected.append({k:v for k,v in c.items() if k!="_point"}); chosen.append(pt); break
    if not selected or len(selected)>=budget: raise ValueError("unexpected budget truncation")
    ids=[[x["local_vertex"],x["source_vertex"],x["donor_body_index"]] for x in selected]
    return {"selected_count":len(selected),"low_quality_site_count":low,"anchor_budget_lower_bound":budget,
      "site_list_ordered_sha256":hbytes(canon(ids)),"site_identity_set_sha256":hbytes(canon(sorted(ids))),
      "site_records_sha256":hbytes(canon(selected)),"selected_sites":selected}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[7])
    ap.add_argument("--skin-asset",type=Path,default=ASSET); ap.add_argument("--baseline-commit",default=COMMIT)
    ap.add_argument("--output",type=Path,default=Path(__file__).with_name("receipt.json")); a=ap.parse_args()
    root=a.root.resolve(); asset=a.skin_asset.resolve(); current=(root/HEADER).read_bytes()
    old=subprocess.run(["git","show",f"{a.baseline_commit}:{HEADER}"],cwd=root,check=True,stdout=subprocess.PIPE).stdout
    oldapp=subprocess.run(["git","show",f"{a.baseline_commit}:apps/numilab_human_myosim_visual_probe.mm"],cwd=root,check=True,stdout=subprocess.PIPE).stdout
    app=root/"apps/numilab_human_myosim_visual_probe.mm"; prep=root/"apps/NumiHumanCoarseTissuePreparation.hpp"; integ=root/"apps/NumiHumanSkinFEMIntegration.hpp"
    bodies,weights,assetsha=skin(asset); before=reconstruct(header(old.decode()),bodies,weights); now=reconstruct(header(current.decode()),bodies,weights)
    oldids={(x["local_vertex"],x["source_vertex"],x["donor_body_index"]) for x in before["selected_sites"]}
    newids={(x["local_vertex"],x["source_vertex"],x["donor_body_index"]) for x in now["selected_sites"]}
    added,lost=sorted(newids-oldids),sorted(oldids-newids); witness=(692,8467,139)
    def site(r): return next((x for x in r["selected_sites"] if (x["local_vertex"],x["source_vertex"],x["donor_body_index"])==witness),None)
    mt=re.search(r'kWholeBodySkinSourcePayloadSHA256\[\]\s*=\s*"([0-9a-f]{64})"',current.decode())
    receipt={
      "schema":"whole_body_skin_fat_anchor_reconstruction_v1",
      "scope":"CPU reconstruction from old/current frozen reduction headers and the hashed full-weight shell asset; no native solver/GPU.",
      "baseline_commit":a.baseline_commit,
      "working_tree_head":subprocess.run(["git","rev-parse","HEAD"],cwd=root,check=True,stdout=subprocess.PIPE,text=True).stdout.strip(),
      "selector_policy":{"source":"apps/numilab_human_myosim_visual_probe.mm:19824-19912",
        "tetrahedron_quality":"per-tet minimum of four vertex altitudes; node receives minimum quality among incident fat prism tets",
        "prism_split_source":"apps/NumiHumanCoarseTissuePreparation.hpp:592-599",
        "dominant_donor_source":"apps/NumiHumanSkinFEMIntegration.hpp:74-90",
        "minimum_incident_altitude_m":5e-5,"minimum_weight_confidence":0.85,"minimum_spacing_m":0.025,"max_sites_per_donor":3,
        "order":"3 rounds; donors ascending; within donor altitude descending, confidence descending, local vertex ascending",
        "body_label":"numeric donor_body_index only; anatomical label not inferred"},
      "inputs":{"baseline_header":{"git_object":f"{a.baseline_commit}:{HEADER}","sha256":hbytes(old)},
        "current_header":{"path":str(root/HEADER),"sha256":hbytes(current)},
        "skinned_shell_asset":{"path":str(asset),"sha256":assetsha,"payload_abi":5,"layout":"60-byte header; 36-byte bindings; 56-byte vertices; uint32 indices; float32 full weights"},
        "baseline_selector_source_sha256":hbytes(oldapp),"current_selector_source_sha256":hfile(app),
        "prism_builder_source_sha256":hfile(prep),"dominant_donor_source_sha256":hfile(integ),
        "header_embedded_payload_sha256":mt.group(1) if mt else None},
      "baseline":before,"current":now,
      "comparison":{"old_site_count":len(oldids),"current_site_count":len(newids),
        "lost_site_identities":[list(x) for x in lost],"added_site_identities":[list(x) for x in added],
        "baseline_sites_all_retained":not lost,"failure_site_identity":list(witness),
        "failure_site_baseline":site(before),"failure_site_current":site(now)},
      "reconstruction_script_sha256":hfile(Path(__file__).resolve())}
    if len(oldids)!=20 or len(newids)!=22 or lost or set(added)!={(802,14269,136),(813,15140,138)}:
        raise ValueError("unexpected reconstructed identity sets: "+json.dumps(receipt["comparison"]))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(receipt,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
    print(json.dumps({"output":str(a.output.resolve()),"baseline_count":len(oldids),"current_count":len(newids),
      "lost":[list(x) for x in lost],"added":[list(x) for x in added],
      "failure_site_baseline_altitude_m":site(before)["min_incident_altitude_m"],
      "failure_site_current_altitude_m":site(now)["min_incident_altitude_m"],
      "baseline_site_list_sha256":before["site_list_ordered_sha256"],
      "current_site_list_sha256":now["site_list_ordered_sha256"],"asset_sha256":assetsha},indent=2,sort_keys=True))
if __name__=="__main__": main()
