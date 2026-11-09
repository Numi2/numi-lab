from pathlib import Path
import gzip,hashlib,json,math,os,statistics,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-ground-contact-20261109";B=R/"build/crow-v15-ground-contact-20261109"
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
shader=B/"feedback_live.metallib";binding=json.loads((A/"feedback_live-bindings.json").read_text());assert h(shader)==binding["experimental_shader_sha256"]
for course,seed in [("training",20261122),("reserved-final-i",20261123),("reserved-final-j",20261124)]:
 name="replicate-"+str(seed);cmd=binding["argv"].copy()
 for k,v in [("--birdflow-navigation-course",course),("--seed",str(seed)),("--state-trace",str(B/(name+".tsv"))),("--crow-replay-pack",str(B/(name+".json")))]:cmd[cmd.index(k)+1]=v
 (A/(name+"-bindings.json")).write_text(json.dumps({"argv":cmd,"plan_sha256":h(A/"gait-replication-plan.md"),"shader_sha256":h(shader),"source_binding":"feedback_live-bindings.json"},indent=2)+"\n")
 env={**os.environ,"DYLD_LIBRARY_PATH":str(B),"CROW_CONTACT_TRACE":"1","CROW_SUPPORT_TRACE":"1","PYTHONDONTWRITEBYTECODE":"1"}
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240)
 (A/(name+".stdout")).write_text(p.stdout)
 with (A/(name+".stderr.gz")).open("wb") as f:
  with gzip.GzipFile(fileobj=f,mode="wb",mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(name+": "+p.stderr[-3000:])
 raw=(B/(name+".json")).read_bytes()
 with (A/(name+".replay.json.gz")).open("wb") as f:
  with gzip.GzipFile(fileobj=f,mode="wb",mtime=0) as z:z.write(raw)
 replay=json.loads(raw)["payload"];frames=replay["frames"];data=json.loads(p.stdout);loads={};nonfoot=0
 for l in p.stderr.splitlines():
  if not l.startswith("CROW_SUPPORT_SAMPLE "):continue
  s=json.loads(l.split(" ",1)[1]);
  if s["lane"]!=1:continue
  flags=loads.setdefault(s["step"],set())
  for c in s["contacts"]:
   a,b=c["a_name"],c["b_name"]
   if "locomotion_ground" not in (a,b):continue
   normal=c["impulses"][0]
   if normal<=0:continue
   foot=next((x for x in (a,b) if x in ("crow_left_foot","crow_right_foot")),None)
   if foot and (-1 if a==foot else 1)*c["normal"][2]*normal>0:flags.add(foot)
   elif not foot:nonfoot+=1
 a,b=frames[0]["q"],frames[-1]["q"];D=math.hypot(b[0]-a[0],b[1]-a[1]);metric={}
 for namefoot in ("crow_left_foot","crow_right_foot"):
  idx=replay["body_names"].index(namefoot);stride=replay["body_state_stride"];speeds=[];zs=[]
  for f in frames[100:250]:
   body=f["body_states"][idx*stride:(idx+1)*stride];zs.append(body[2])
   if namefoot in loads.get(f["step"],set()):speeds.append(math.hypot(body[7],body[8]))
  speeds.sort();metric[namefoot]={"loaded_samples":sum(namefoot in x for x in loads.values()),"only_loaded_samples":sum(x=={namefoot} for x in loads.values()),"center_height_min_max_m":[min(zs),max(zs)],"loaded_center_speed_median_m_s":statistics.median(speeds) if speeds else None,"loaded_center_speed_p95_m_s":speeds[math.ceil(.95*len(speeds))-1] if speeds else None,"loaded_center_speed_max_m_s":max(speeds) if speeds else None}
 report={"course":course,"seed":seed,"frames":len(frames),"net_planar_displacement_m":D,"root_height_min_max_m":[min(f["q"][2] for f in frames),max(f["q"][2] for f in frames)],"maximum_tilt_radians":max(f["tilt"] for f in frames),"feet":metric,"nonfoot_ground_contact_rows":nonfoot,"failed_environment_steps":data.get("failed_environment_steps"),"first_episode_navigation":data.get("first_episode_navigation"),"scope":"development screen; foot-center speed is not contact-point slip"}
 (A/(name+"-summary.json")).write_text(json.dumps(report,indent=2)+"\n");print(name,D,metric,flush=True)
