from pathlib import Path
import gzip, hashlib, importlib.util, json, tempfile
import numpy as np
ROOT=Path(__file__).resolve().parent
path=ROOT/"compare_remeshed_native_310.py"
spec=importlib.util.spec_from_file_location("metric",path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
controls=[]
def passed(name):controls.append(name)
rest=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
F=np.diag([1.02,1.,1.])
def state(r,t):
 n=len(r);forces=np.zeros((n,4,3));forces[:4,0,0]=.5
 return dict(rest=r,u=r@(F-np.eye(3)).T,mass=np.ones(n),region=np.full(n,3,np.int32),obj=np.zeros(n,np.int16),objects={"known":0},forces=forces,tets=t,treg=np.full(len(t),3,np.int32),tobj=np.zeros(len(t),np.int16))
a=state(rest,np.array([[0,1,2,3]],np.int32))
b=state(np.vstack([rest,rest.mean(0)]),np.array([[4,1,2,3],[0,4,2,3],[0,1,4,3],[0,1,2,4]],np.int32))
x=m.strain_stats(a)["region:3"];y=m.strain_stats(b)["region:3"];E=(1.02**2-1)/2
for name,got,want in [("unit_tet_volume",x["volume_m3"],1/6),("affine_green_strain",x["volume_weighted_strain_norm"],E/6),("strain_squared_measure",x["volume_weighted_strain_squared"],E**2/6),("affine_J",x["minimum_J"],1.02)]:
 assert np.isclose(got,want,rtol=0,atol=1e-14),(name,got,want);passed(name)
assert all(np.isclose(x[k],y[k],rtol=0,atol=1e-14) for k in x if k!="cells");passed("tet_subdivision_invariance")
assert m.force_stats(a)==m.force_stats(b);passed("integrated_force_subdivision_invariance")
for z in (a,b):
 z["region"][:]=-1;z["treg"][:]=-1
 assert "object:known" in m.strain_stats(z) and "object:known" in m.force_stats(z)
passed("unlabeled_int16_namespace")
bad=state(rest,np.array([[0,0,2,3]],np.int32))
try:m.strain_stats(bad)
except AssertionError:passed("singular_cell_rejected")
else:raise AssertionError("singular cell admitted")
bad=state(rest,np.array([[0,1,2,3]],np.int32));bad["u"][:,0]=-2*bad["rest"][:,0]
try:m.strain_stats(bad)
except AssertionError:passed("inverted_state_rejected")
else:raise AssertionError("inverted state admitted")
values=np.array([1.7359761,-1.7365108,.0000017384878,-.000004978282,1.1920929e-7,1.e-18],np.float32)
assert np.array_equal(values,np.array([m.f32(format(v,".10g")) for v in values],np.float32));passed("native_f32_text_roundtrip")
# Exercise the real capture parser, including its duplicate/truncation checks.
header="resting_FEM_endpoint_capture=begin version=2 coordinate_unit=m force_unit=N strain_unit=1 nodes=4 tetrahedra=1\n"
lines=[header]
for i,r in enumerate(rest):
 row=dict(object="known",local=i,global_=i,reference_mode=0,constraint_tag=0,mass_kg=1)
 row["global"]=row.pop("global_")
 for j,c in enumerate("xyz"):row["rest_"+c]=r[j];row["accepted_"+c]=r[j]
 for channel in m.CHANNELS:
  for c in "xyz":row[channel+"_"+c+"_N"]=0
 lines.append("resting_FEM_node,"+",".join(str(k)+"="+str(v) for k,v in row.items())+"\n")
lines+=["resting_FEM_tet,object=known,local=0,global=0,n0=0,n1=1,n2=2,n3=3,J=1\n","resting_FEM_endpoint_capture=end\n","resting_integrated_body=completed simulated_s=0 wall_s=1\n"]
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/"fixture.gz"
 def emit(rows):
  with gzip.open(p,"wt") as f:f.writelines(rows)
 emit(lines);z=m.capture(p);assert np.array_equal(z["rest"],rest);passed("complete_capture_parser")
 for name,rows in [("duplicate_node_rejected",lines[:2]+[lines[1]]+lines[2:]),("truncated_capture_rejected",lines[:-2]+lines[-1:])]:
  emit(rows)
  try:m.capture(p)
  except AssertionError:passed(name)
  else:raise AssertionError(name)
receipt=dict(status="PASS_ANALYTIC_POSTPASS_CONTROLS",controls=controls,instrument_path=str(path),instrument_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physical_qualification=False)
(ROOT/"joint-boundary-native-310-postpass-controls.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
