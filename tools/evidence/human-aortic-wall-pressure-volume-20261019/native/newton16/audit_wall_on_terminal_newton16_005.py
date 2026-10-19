
import csv,hashlib,json,math
from pathlib import Path

ROOT=Path("/Users/n/numi-human-separation-20261018/C")
RUN=ROOT/"aortic-wall-pressure-volume-258/native-32step-paired-newton16-005"
APP=ROOT/"aortic-wall-pressure-volume-258/build-newton16-app-001/source/apps/numilab_human_myosim_visual_probe.mm"
WALL=ROOT/"aortic-wall-pressure-volume-258/build-newton16-app-001/source/apps/NumiHumanRestingAorticWall.hpp"
COUPLING=ROOT/"aortic-wall-pressure-volume-258/build-newton16-app-001/source/apps/NumiHumanVesselWallCoupling.hpp"
MSL=ROOT/"integrated-general-tissue-258/production-merged-007/MatterSource.metal"
LIB=ROOT/"integrated-general-tissue-258/production-merged-007/build-production-001/libmetalrobo.dylib"
OBJ="Human/BodyParts3D/stable-9/reduced-cavity-wall-v1"

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,msg):
    if not ok: raise RuntimeError(msg)

log=RUN/"wall_on/native.log"
csvpath=RUN/"wall_on/output/resting-coupled.csv"
invpath=RUN/"wall_on/invocation.json"
offlog=RUN/"wall_off/native.log"
audit=RUN/"wall_on_terminal_volume_audit.json"
inv=json.loads(invpath.read_text())
require(inv["process_exit"]==0,"wall-on native did not pass")
env=inv["environment"]
require(env.get("NUMI_HUMAN_RESTING_AORTIC_WALL")=="1","wall-on gate mismatch")
require(env.get("NUMI_HUMAN_RESTING_FGMRES_BUDGET")=="0","not cooked-default FGMRES run")
require(env.get("NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS")=="16","Newton16 opt-in mismatch")
require(env.get("NUMI_MATTER_TEST_SOURCE")==str(MSL),"invocation MSL path mismatch")
require(env.get("NUMI_MATTER_TEST_SOURCE_SHA256")==sha(MSL),"invocation MSL SHA mismatch")
dyld=[line for line in log.read_text(errors="replace").splitlines() if "libmetalrobo.dylib" in line and Path(line.rsplit(" ",1)[-1]).exists() and Path(line.rsplit(" ",1)[-1]).samefile(LIB)]
require(bool(dyld),"actual production dylib not observed in DYLD output")

text=log.read_text(errors="replace")
begin=next(x for x in text.splitlines() if x.startswith("resting_FEM_endpoint_capture=begin "))
capture=dict(t.split("=",1) for t in begin.split()[1:])
require(int(capture["steps"])==32,"unexpected endpoint frame count")
summary_line=next(x for x in text.splitlines() if x.startswith("resting_FEM_object_capture=summary object="+OBJ+" "))
sm=dict(t.split("=",1) for t in summary_line.split()[1:])
first=int(sm["first_node"])
nodes={};tets={}
for line in text.splitlines():
    if line.startswith("resting_FEM_node,object="+OBJ+","):
        d=dict(t.split("=",1) for t in line.split(",")[1:])
        i=int(d["local"])
        nodes[i]=((float(d["initial_x"]),float(d["initial_y"]),float(d["initial_z"])),
                  (float(d["accepted_x"]),float(d["accepted_y"]),float(d["accepted_z"])))
    elif line.startswith("resting_FEM_tet,object="+OBJ+","):
        d=dict(t.split("=",1) for t in line.split(",")[1:])
        i=int(d["local"])
        tets[i]=tuple(int(d["n"+str(j)])-first for j in range(4))
require(len(nodes)==532 and len(tets)==1584,"captured mesh topology changed")
n,axial=24,8
inner={48*z+2*i for z in range(axial+1) for i in range(n)}
cap={tuple(sorted((432,2*i,2*((i+1)%n)))) for i in range(n)}
cap|={tuple(sorted((433,48*axial+2*i,48*axial+2*((i+1)%n)))) for i in range(n)}
boundary={}
for i in range(1584):
    a,b,c,d=tets[i]
    for f in ((b,c,d),(a,d,c),(a,b,d),(a,c,b)):
        key=tuple(sorted(f))
        count,oriented=boundary.get(key,(0,f))
        boundary[key]=(count+1,oriented)
faces=[]
for key,(count,face) in boundary.items():
    if count==1 and (all(v in inner for v in key) or key in cap):
        faces.append((face[0],face[2],face[1]))
require(len(faces)==432,"cavity face boundary mismatch")
def signed_volume(frame):
    origin=nodes[faces[0][0]][frame]
    total=0.0
    for face in faces:
        a,b,c=[tuple(nodes[i][frame][k]-origin[k] for k in range(3)) for i in face]
        cross=(b[1]*c[2]-b[2]*c[1],b[2]*c[0]-b[0]*c[2],b[0]*c[1]-b[1]*c[0])
        total+=(a[0]*cross[0]+a[1]*cross[1]+a[2]*cross[2])/6.0
    return total
v0=signed_volume(0);v1=signed_volume(1)
rows=list(csv.DictReader(csvpath.open()))
require(len(rows)==32,"expected 32 accepted CV rows")
require([int(r["step"]) for r in rows]==list(range(1,33)),"CV step sequence mismatch")
offtext=offlog.read_text(errors="replace")
failure=next((x for x in offtext.splitlines() if x.startswith("human_joint_failure ")),None)
require(failure is not None,"expected retained wall-off failure not found")
off_loaded=[line for line in offtext.splitlines() if "libmetalrobo.dylib" in line and Path(line.rsplit(" ",1)[-1]).exists() and Path(line.rsplit(" ",1)[-1]).samefile(LIB)]
result={
 "schema":"numi.aortic_wall_pv_wall_on_endpoint_audit.v1",
 "interpretation":"One 32-step wall-on accepted state with offline signed cavity-volume reconstruction and CV aortic-pressure association. Wall-off paired case failed at segment_start=14, so this endpoint audit is descriptive and does not establish an ON/OFF causal effect or pressure-volume calibration.",
 "run_plan_sha256":sha(RUN/"run-plan.json"),
 "invocation_sha256":sha(invpath),
 "binary_path":inv["argv"][0],
 "binary_sha256":inv["assets_sha256"][str(Path(inv["argv"][0]).resolve())],
 "production_msl_path":str(MSL),
 "production_msl_sha256":sha(MSL),
 "production_library_path":str(LIB),
 "production_library_sha256":sha(LIB),
 "observed_dyld_line":dyld[-1],
 "app_sha256":sha(APP),
 "aortic_wall_header_path":str(WALL),
 "aortic_wall_header_sha256":sha(WALL),
 "vessel_wall_coupling_header_path":str(COUPLING),
 "vessel_wall_coupling_header_sha256":sha(COUPLING),
 "controls":{"wall":"on","MPM":"off","FGMRES":"cooked default 64 (environment 0)","steps":32,"step_seconds":0.00025},
 "wall":{"object":OBJ,"nodes":len(nodes),"tetrahedra":len(tets),"pressure_boundary_faces":len(faces),"terminal_max_displacement_m":float(sm["max_displacement_m"]),"terminal_min_J":float(sm["min_J"]),"terminal_max_J":float(sm["max_J"])},
 "cavity_geometry":{"signed_initial_volume_ml":v0*1e6,"signed_terminal_volume_ml":v1*1e6,"initial_orientation_positive":v0>0,"terminal_orientation_positive":v1>0,"volume_change_ml":(abs(v1)-abs(v0))*1e6,"volume_change_fraction":abs(v1)/abs(v0)-1},
 "cv_series":{"rows":len(rows),"initial_time_s":float(rows[0]["time_s"]),"terminal_time_s":float(rows[-1]["time_s"]),"initial_aortic_pressure_mmhg":float(rows[0]["aorta_mmhg"]),"terminal_aortic_pressure_mmhg":float(rows[-1]["aorta_mmhg"]),"initial_total_blood_ml":float(rows[0]["blood_ml"]),"terminal_total_blood_ml":float(rows[-1]["blood_ml"]),"csv_sha256":sha(csvpath)},
 "wall_off":{"native_exit":json.loads((RUN/"wall_off/invocation.json").read_text())["process_exit"],"failure_line":failure,"production_library_observed":bool(off_loaded)},
 "source_binding_check":{"CV_compartment_stable_id":7,"wall_cavity_stable_id":9007,"same_CV_volume_state_reused":True,"initial_geometry_admission_tolerance_relative":2e-4,"application_check":"/Users/n/numi-human-separation-20261018/C/aortic-wall-pressure-volume-258/build-newton16-app-001/source/apps/NumiHumanVesselWallCoupling.hpp:100-176"},
 "native_log_sha256":sha(log),
 "source_visual_verification":False
}
audit.write_text(json.dumps(result,indent=2)+"\n")
result["audit_script_sha256"]=sha(Path(__file__))
print(json.dumps(result,indent=2))
