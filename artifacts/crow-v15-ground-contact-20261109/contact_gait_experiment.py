from pathlib import Path
import difflib,gzip,hashlib,json,math,os,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-ground-contact-20261109";B=R/"build/crow-v15-ground-contact-20261109";original=(R/"src/metal/LocomotionTask.metal").read_text()
def once(s,a,b):
 if s.count(a)!=1:raise RuntimeError(f"expected one marker, found {s.count(a)}: {a[:80]}")
 return s.replace(a,b)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for arm in ("feedback_zero","feedback_live"):
 s=original
 marker="""        // Keep the training-only teacher active through the final target.
"""
 gait="""        // Isolated walking experiment: one-step-lag accepted foot-load gating.
        if (fixedFlightController && state.episode.z == 1u && action >= 7u && action <= 12u) {
            const float seconds = float(state.episode.x)*dispatch.timing.x;
            const uint leg = (action-7u)/3u, joint = (action-7u)%3u;
            const float support = leg == 0u ? state.nutrition.y : state.nutrition.x;
            const float carrier = sin(kTwoPi*seconds/0.50f)*(leg == 0u ? 1.0f : -1.0f);
            const float swing = seconds >= 2.0f && seconds < 5.0f && support > 0.05f ? carrier : 0.0f;
            const float lift = max(swing,0.0f);
            navigationTeacherRequested = joint == 0u ? -0.30f*swing : joint == 1u ? 0.40f*lift : -0.15f*lift;
        }
"""
 s=once(s,marker,gait+marker)
 marker="""    taskStates[environment] = state;

    const uint transitionIndex =
"""
 values="float2(0.0f)" if arm=="feedback_zero" else "float2(compactContact[compactBase+contactGroups[0].members.w], compactContact[compactBase+contactGroups[1].members.w])"
 injection="""    // The experimental band uses two otherwise unused diagnostic components;
    // the navigation actor and observation publishing have already run.
    if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION) != 0u && state.episode.z == 1u && program.counts0.w >= 2u) {
        state.nutrition.xy = VALUES;
    }
""".replace("VALUES",values)
 s=once(s,marker,injection+marker)
 (A/(arm+".patch")).write_text("".join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile="a/src/metal/LocomotionTask.metal",tofile="b/"+arm+".metal")))
 src=B/(arm+".metal");air=B/(arm+".air");lib=B/(arm+".metallib");src.write_text(s)
 cmds=[["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(src),"-o",str(air)], ["xcrun","-sdk","macosx","metallib",*[str(p) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name!="MetalRoboLocomotionTask.air"],str(air),"-o",str(lib)]]
 for cmd in cmds:subprocess.run(cmd,cwd=R,check=True)
 cmd=json.loads((A/"bindings.json").read_text())["argv"]
 for k,v in [("--metallib",str(lib)),("--minimum-difficulty-band","1"),("--maximum-difficulty-band","1"),("--state-trace",str(B/(arm+".tsv")))]:cmd[cmd.index(k)+1]=v
 cmd += ["--crow-replay-pack",str(B/(arm+".json"))]
 (A/(arm+"-bindings.json")).write_text(json.dumps({"argv":cmd,"plan_sha256":h(A/"contact-gait-plan.md"),"patch_sha256":h(A/(arm+".patch")),"experimental_source_sha256":h(src),"experimental_shader_sha256":h(lib),"observer_sha256":h(B/"libmetalrobo.dylib"),"frozen_flight_source_sha256":h(R/"src/metal/CrowFlightControllerV1.metal"),"baseline_bindings":"bindings.json"},indent=2)+"\n")
 env={**os.environ,"DYLD_LIBRARY_PATH":str(B),"CROW_CONTACT_TRACE":"1","CROW_SUPPORT_TRACE":"1","PYTHONDONTWRITEBYTECODE":"1"}
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240)
 (A/(arm+".stdout")).write_text(p.stdout)
 with (A/(arm+".stderr.gz")).open("wb") as f:
  with gzip.GzipFile(fileobj=f,mode="wb",mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(arm+" failed: "+p.stderr[-3000:])
 raw=(B/(arm+".json")).read_bytes()
 with (A/(arm+".replay.json.gz")).open("wb") as f:
  with gzip.GzipFile(fileobj=f,mode="wb",mtime=0) as z:z.write(raw)
 frames=json.loads(raw)["payload"]["frames"]; data=json.loads(p.stdout)
 samples=[json.loads(x.split(" ",1)[1]) for x in p.stderr.splitlines() if x.startswith("CROW_SUPPORT_SAMPLE ")]
 rows=[]
 for lane in range(2):
  loads={}
  for sample in samples:
   if sample["lane"]!=lane:continue
   flags=loads.setdefault(sample["step"],set())
   for c in sample["contacts"]:
    a,b=c["a_name"],c["b_name"]
    if "locomotion_ground" not in (a,b):continue
    isA=a in ("crow_left_foot","crow_right_foot");isB=b in ("crow_left_foot","crow_right_foot")
    if (isA or isB) and (-1 if isA else 1)*c["normal"][2]*c["impulses"][0]>0:flags.add(a if isA else b)
  left="crow_left_foot";right="crow_right_foot"
  row={"lane":lane,"left_loaded_samples":sum(left in x for x in loads.values()),"right_loaded_samples":sum(right in x for x in loads.values()),"left_only_samples":sum(x=={left} for x in loads.values()),"right_only_samples":sum(x=={right} for x in loads.values())}
  if lane==1:
   a,b=frames[0]["q"],frames[-1]["q"];row["net_planar_displacement_m"]=math.hypot(b[0]-a[0],b[1]-a[1])
  rows.append(row)
 result={"arm":arm,"rows":rows,"failed_environment_steps":data.get("failed_environment_steps"),"first_episode_navigation":data.get("first_episode_navigation"),"scope":"exploratory endpoint samples on exposed layout, not qualification"}
 (A/(arm+"-summary.json")).write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result,indent=2),flush=True)
