from pathlib import Path
import json, hashlib, importlib.util, mmap
E=Path("/Users/n/numi-human-resting-evidence-20261005")
BASE=E/"final-native-scene-preflight-936"
OUT=E/"native-retired-alias-geometry-identity-018"
OLD=BASE/"skin-927-current-lung-924/native-run"
NEW=BASE/"skin-927-lung-924-viewer-018-v014-final-attempt3/native-run"
READER=E/"cardiac-wall-native-self-audit-001/accepted_mrvpack_surface_audit.py"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4194304),b""):h.update(b)
 return h.hexdigest()
spec=importlib.util.spec_from_file_location("pack_reader",READER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def check(run,step):
 rp=run/"accepted-geometry"/f"step-{step}.receipt.json"
 r=json.loads(rp.read_text());p=Path(r["accepted_pack_path"])
 assert p.resolve()==(run/"accepted-geometry"/f"step-{step}.mrvpack").resolve()
 assert r["accepted_step"]==step and r["physical_endpoint"]=="accepted"
 digest=sha(p);assert digest==r["pack_file_sha256"]
 with p.open("rb") as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
  header=m.HEADER.unpack_from(data)
  assert header[:2]==(b"MRVPACK2",2)
  rows=[m.DIRECTORY.unpack_from(data,m.HEADER.size+i*m.DIRECTORY.size) for i in range(header[2])]
  assert len({x[0] for x in rows})==len(rows)
  hashes={}
  for row in rows:
   typ,flags,offset,size,count,stride,*_=row
   if typ not in (2,3,4,5):continue
   assert offset+size<=len(data) and size==count*stride
   hashes[str(typ)]={"sha256":hashlib.sha256(data[offset:offset+size]).hexdigest(),"bytes":size,"count":count,"stride":stride}
  assert set(hashes)=={"2","3","4","5"}
  assert hashes["2"]["sha256"]==r["captured_vertex_buffer_sha256"]
 return {"receipt_path":str(rp),"receipt_sha256":sha(rp),"pack_path":str(p),"pack_sha256":digest,"accepted_step":step,"accepted_time_s":r["accepted_time_s"],"accepted_body_state_sha256":r["accepted_body_state_sha256"],"accepted_respiration_state_sha256":r["accepted_respiration_state_sha256"],"sections":hashes}
rows=[]
for step in (0,4991,5375,5759,6111,6495,7743,10000):
 a=check(OLD,step);b=check(NEW,step)
 same=a["sections"]==b["sections"]
 assert same
 assert a["accepted_body_state_sha256"]==b["accepted_body_state_sha256"]
 assert a["accepted_respiration_state_sha256"]==b["accepted_respiration_state_sha256"]
 rows.append({"step":step,"all_geometry_sections_byte_identical":same,"reference":a,"candidate":b})
result={"schema":"numi.human.native-geometry-regression.v1","status":"passed","reference_run":str(OLD),"candidate_run":str(NEW),"script_sha256":sha(__file__),"reader_path":str(READER),"reader_sha256":sha(READER),"section_meanings":{"2":"vertices including positions normals tangents","3":"indices","4":"primitives","5":"instances"},"captures":rows,"limits":["Eight stored accepted captures only; no continuous-time or later-time geometry claim.","Live layer selection uses the GPU instance-layer mask. Exported packs retain the retired alias and base flags.","Whole-pack hashes differ because their metadata binds different run paths; section equality is the geometry claim."]}
report=OUT/"comparison.json"
assert not report.exists()
report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":"passed","captures":len(rows),"geometry_sections_per_capture":4,"report":str(report),"sha256":sha(report)},indent=2))
