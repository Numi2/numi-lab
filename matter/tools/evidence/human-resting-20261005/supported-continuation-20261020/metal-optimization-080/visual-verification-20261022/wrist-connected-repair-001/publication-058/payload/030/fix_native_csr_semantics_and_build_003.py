from pathlib import Path
import numpy as np,json,hashlib,shutil,subprocess,time,shlex,difflib
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");N=E/"native-candidate020-conforming-prism-030";A=N/"apps";C=E/"native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
shutil.copytree(A,N/"apps-attempt-002")
shutil.copy2(N/"candidate-identity.json",N/"candidate-identity-attempt-002.json")
shutil.copy2(N/"app-integration.diff",N/"app-integration-attempt-002.diff")
a=np.load(C/"candidate.npz");x=a["points_world_f32"].astype(np.float64);t=a["tetrahedra"];o=a["parent_csr_offsets"];pr=a["parent_csr_old_tet_rows"];ov=a["parent_csr_overlap_volumes_m3"]
def volumes(t):
 p=x[t];return np.linalg.det(np.stack([p[:,1]-p[:,0],p[:,2]-p[:,0],p[:,3]-p[:,0]],axis=2))/6
cv=volumes(t);pv=volumes(a["old_tetrahedra"])
assert np.all(cv>0) and np.all(pv>0)
childRows=np.repeat(np.arange(len(cv)),np.diff(o))
fractions=ov/cv[childRows]
rowSums=np.bincount(childRows,weights=fractions,minlength=len(cv))
parentCovered=np.bincount(pr,weights=ov,minlength=len(pv))
assert max(abs(rowSums-1))<1e-8
assert max(abs(parentCovered/pv-1))<1e-8
dominant=np.array([pr[o[i]+np.argmax(ov[o[i]:o[i+1]])] for i in range(len(t))])
hp=A/"NumiHumanConformingPrism030.hpp";h=hp.read_text()
start=h.index("inline constexpr std::array<double,");end=h.index('inline constexpr const char* receiptSHA256=',start)
h=h[:start]+"inline constexpr std::array<double,"+str(len(fractions))+"> parentFractions{{\n"+",\n".join("    "+format(float(v),".17g") for v in fractions)+"\n}};\n"+"inline constexpr std::array<std::uint32_t,"+str(len(dominant))+"> dominantParent{{\n"+",\n".join("    "+str(int(v))+"u" for v in dominant)+"\n}};\n"+h[end:]
hp.write_text(h)
receipt={"status":"PASS","source_npz_sha256":sha(C/"candidate.npz"),"input_semantics":"parent_csr_old_volume_fractions = overlap/old parent volume; column-normalized transport","native_semantics":"child ancestry = overlap/new child volume; denominator from exact signed child determinant; no normalization","maximum_child_fraction_sum_error":float(max(abs(rowSums-1))),"maximum_parent_volume_relative_error":float(max(abs(parentCovered/pv-1))),"dominant_parent_policy":"maximum absolute overlap volume; deterministic first parent on ties","native_header_sha256":sha(hp)}
rp=N/"native-csr-semantics-conversion.json";rp.write_text(json.dumps(receipt,indent=2)+"\n")
ip=N/"candidate-identity.json";oldsha=sha(ip);identity=json.loads(ip.read_text())
identity.update(topology_header_sha256=sha(hp),native_csr_semantics_receipt_sha256=sha(rp),dominant_cell_region_policy="largest absolute exact-overlap parent; node donor frames and mass remain registered source")
ip.write_text(json.dumps(identity,indent=2,sort_keys=True)+"\n")
p=A/"numilab_human_myosim_visual_probe.mm";s=p.read_text();assert s.count(oldsha)==1;s=s.replace(oldsha,sha(ip));p.write_text(s)
base=(E/"native-candidate020-mechanical-source-024/apps"/p.name).read_text()
(N/"app-integration.diff").write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="024/app.mm",tofile="030/app.mm")))
B=N/"build-003";B.mkdir()
for label in ("compile","link"):
 cmd=json.loads((N/"build-002"/(label+"-command.json")).read_text())
 cmd=[str(B/Path(x).name) if x.startswith(str(N/"build-002")+"/") else x for x in cmd]
 (B/(label+"-command.json")).write_text(json.dumps(cmd,indent=2))
 now=time.monotonic()
 with (B/(label+".log")).open("wb") as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 (B/(label+"-result.json")).write_text(json.dumps({"returncode":r.returncode,"wall_seconds":time.monotonic()-now}))
 print(label,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
deps=shlex.split((B/"candidate.d").read_text().split(":",1)[1].replace("\\\n",""))
for p in A.iterdir():
 if p.is_file() and p.suffix==".hpp":assert [Path(z) for z in deps if z.endswith("/"+p.name)]==[p],p
(B/"dependency-certification.json").write_text(json.dumps({"status":"PASS_EXACT_OVERLAY","input_hashes":{z:sha(z) for z in deps if Path(z).is_file() and z.startswith("/Users/")},"binary_sha256":sha(B/"numi-human-native"),"physical_qualification":False},indent=2))
s=(N/"run_source_preflight.py").read_text().replace("build-001","build-003").replace("source-preflight-001","source-preflight-003")
(N/"run_source_preflight_003.py").write_text(s)
raise SystemExit(subprocess.run(["python3",str(N/"run_source_preflight_003.py")]).returncode)
