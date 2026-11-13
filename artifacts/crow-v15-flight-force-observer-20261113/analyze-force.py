from pathlib import Path
import gzip,json,math,collections,hashlib
A=Path("artifacts/crow-v15-flight-force-observer-20261113");P=A/"push-wing-20261126-rebound-force";Q=Path("artifacts/crow-v15-flight-pitch-feedback-20261113/push-wing-20261126-rebound-pitch");meta=json.loads((P/"run.stdout").read_text())["first_episode_navigation"];oldmeta=json.loads((Q/"run.stdout").read_text())["first_episode_navigation"];refanalysis=json.loads((Q.parent/"analysis.json").read_text());recontact={x["lane"]:x["recontact"] for x in refanalysis["rows"]};data=[]
for d in (Q,P):
 state={};mask=[];force={};effort={};contacts={};errors=[];duplicate=[]
 with gzip.open(d/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);key=x["lane"],x["step"]
    if key in state:duplicate.append(["state",key])
    state[key]=x;continue
   z=l.split()
   if not z:continue
   if z[0]=="CROW_SCENE_MASK":mask.append((z[1],int(z[2]),int(z[3])))
   if z[0]=="CROW_AERO_OBSERVER_ERROR":errors.append(l.strip())
   if z[0]=="CROW_TWO_TOE_SUBSTEP":
    t,k,e=map(int,z[1:4]);key=(e,t,k)
    if key in contacts:duplicate.append([z[0],key])
    contacts[key]=[sum(map(float,z[8:10])),float(z[10])]
   if z[0] in ("CROW_AERO_FORCE","CROW_WING_EFFORT"):
    t,k,e=map(int,z[1:4]);key=(e,t,k);table=force if z[0]=="CROW_AERO_FORCE" else effort
    if key in table:duplicate.append([z[0],key])
    table[key]=list(map(float,z[4:]))
 data.append({"state":state,"mask":mask,"force":force,"effort":effort,"contacts":contacts,"errors":errors,"duplicate":duplicate})
old,new=data;inputs=[json.loads((d/"inputs.json").read_text()) for d in (Q,P)];equalinputs=all(inputs[0][k]==inputs[1][k] for k in ("metallib_sha256","route_policy_sha256","landing_policy_sha256"));maskvalid=old["mask"]==new["mask"];rows=[]
for e in range(2,8):
 end=min(oldmeta["steps_by_environment"][e],900);oldend=oldmeta["steps_by_environment"][e];newend=meta["steps_by_environment"][e];missing=[];diff={k:[] for k in ("q","v","policy_action","contacts")}
 for t in range(1,end+1):
  if (e,t) not in old["state"] or (e,t) not in new["state"]:missing.append(t);continue
  for k in diff:
   if old["state"][e,t][k]!=new["state"][e,t][k]:diff[k].append(t)
 parity=oldend==newend and not missing and not any(diff.values());last=recontact[e];keys=[(e,t,k) for t in range(500,last or 500) for k in range(16)];missing_force=[k for k in keys if k not in new["force"]];missing_effort=[k for k in keys if k not in new["effort"]];missing_contact=[k for k in keys if k not in new["contacts"]];clean_contact=not missing_contact and all(max(new["contacts"][k])<=1e-7 for k in keys);good=bool(keys) and not missing_force and not missing_effort and clean_contact and not new["errors"] and not new["duplicate"]
 if good:
  force=[new["force"][k] for k in keys];eff=[new["effort"][k] for k in keys];good=all(len(z)==13 and int(z[-1])==4 and all(math.isfinite(x) for x in z) for z in force) and all(len(z)==14 and all(math.isfinite(x) for x in z) for z in eff)
  if good:mean=[sum(z[i] for z in force)/len(force) for i in (1,2,3)];imp=[x*len(keys)*.00125 for x in mean];peak=[max(abs(z[i]) for z in eff) for i in (0,7)]
  else:mean=imp=peak=None
 else:mean=imp=peak=None
 prekeys=[(e,t,k) for t in range(450,501) for k in range(16)];pregood=all(k in new["force"] and len(new["force"][k])==13 and int(new["force"][k][-1])==4 for k in prekeys);premean=[sum(new["force"][k][i] for k in prekeys)/len(prekeys) for i in (1,2,3)] if pregood else None
 row={"prewindow_450_500_mean_aero_force_n":premean,"prewindow_complete":pregood,"lane":e,"original_first_episode_end":oldend,"rerun_first_episode_end":newend,"first_foot_recontact":last,"state_parity_full_original_episode":parity,"missing_state_ticks":missing,"state_mismatches":diff,"airborne_force_substeps":len(keys),"missing_force_substeps":missing_force,"missing_effort_substeps":missing_effort,"missing_contact_substeps":missing_contact,"airborne_without_recorded_normal_impulse":clean_contact,"observer_complete_and_finite":good,"world_mean_aero_force_newtons":mean,"world_aero_impulse_ns":imp,"max_abs_flap_drive_effort":peak,"reference_body_weight_newtons":.45006*9.81,"root_vx_500":new["state"].get((e,500),{}).get("v",[None])[0],"root_vx_before_recontact":new["state"].get((e,(last or 1)-1),{}).get("v",[None])[0]};row["valid_matched_observation"]=bool(equalinputs and maskvalid and parity and good);rows.append(row)
out={"same_shader_and_policy_bytes":equalinputs,"same_scene_mask":maskvalid,"observer_errors":new["errors"],"duplicate_count":len(new["duplicate"]),"rows":rows,"scope":"Force increments are from the modeled aerodynamic kernel in world coordinates, not joint muscle work, measured animal force or a full independent momentum budget. Compare mean force to approximate 0.45006 kg model weight cautiously. Root speed is not COM speed."};(A/"force-analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
