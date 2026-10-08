#!/usr/bin/env python3
import csv, hashlib, json, math, struct
from pathlib import Path
run=Path("/Users/n/numi-human-resting-evidence-20261005/final-native-scene-preflight-936/skin-927-lung-924-viewer-018-v014-final-attempt3/native-run")
out=Path("/Users/n/numi-human-resting-evidence-20261005/native-viewer018-recording-review-960")
dense=Path("/Users/n/numi-human-resting-evidence-20261005/native-viewer018-recording-review-960-dense")
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def load(path): return json.loads(Path(path).read_text())
decode=load(out/"movie-continuity-full-decode.json")
samples=load(out/"even-content-samples.json")
dense_samples=load(dense/"even-content-samples.json")
meta=load(run/"run-metadata.json")
with open(run/"resting-coupled.csv",newline="") as f: trace=list(csv.DictReader(f))
with open(run/"resting-surface-audit.csv",newline="") as f: surfaces=list(csv.DictReader(f))
log=(run/"native.log").read_text(errors="replace").splitlines()
mask=[x for x in log if "resting_retired_inspection_source semantic=51010 stable_id=22" in x]
throughput=next((x for x in log if x.startswith("resting_integrated_throughput ")),None)
body=next((x for x in log if x.startswith("resting_integrated_body=")),None)
body_fields=dict(token.split("=",1) for token in body.split() if "=" in token) if body else {}
def changes(key):
    return [{"time_s":float(r["time_s"]),"step":int(r["step"]),"value":int(float(r[key]))} for i,r in enumerate(trace) if i and r[key]!=trace[i-1][key]]
def stats(key):
    v=[float(r[key]) for r in trace]; f=[x for x in v if math.isfinite(x)]
    return {"first":v[0],"last":v[-1],"min_finite":min(f) if f else None,"max_finite":max(f) if f else None}
def sample(rep,name): return next(x for x in rep["content_samples"] if x["file"]==name)
mapping={"wholebody":(samples,"content-even-06.png","Whole body","selected-scene-layers/wholebody.png"),"viscera":(dense_samples,"content-even-09.png","Organs","selected-scene-layers/viscera-organs.png"),"lungs":(samples,"content-even-03.png","Lungs","selected-scene-layers/lungs.png"),"heart":(samples,"content-even-04.png","Heart detail","selected-scene-layers/heart.png"),"vessels":(samples,"content-even-05.png","Vessels","selected-scene-layers/vessels.png")}
layers={}
for name,(rep,frame,ui,path) in mapping.items():
    p=out/path; raw=p.read_bytes(); w,h=struct.unpack(">II",raw[16:24]); s=sample(rep,frame)
    layers[name]={"file":str(p),"sha256":sha(p),"decoded_pts_s":s["decoded_pts_s"],"decoded_frame_index":s["decoded_frame_index"],"ui_layer_seen_in_frame":ui,"png_dimensions":[w,h],"nonblack_fraction":s["nonblack_fraction"]}
times=[float(x["time_s"]) for x in surfaces]; steps=[int(x["step"]) for x in surfaces]; dts=sorted(b-a for a,b in zip(times,times[1:]))
movie=run/"native-viewer.mov"; metadata=run/"run-metadata.json"; surface=run/"resting-surface-audit.csv"; tracepath=run/"resting-coupled.csv"; logfile=run/"native.log"
report={"schema":"numi.human.native-viewer-recording-review.v1","run_directory":str(run),"run_exit_code":meta.get("exit_code"),"launcher_wall_seconds":meta.get("wall_seconds"),"loaded_metal_runtime_verified":meta.get("loaded_metal_runtime",{}).get("verified"),"inputs_sha256":{"movie":sha(movie),"run_metadata":sha(metadata),"native_log":sha(logfile),"surface_audit_csv":sha(surface),"coupled_csv":sha(tracepath),"pinned_decoder":sha("/Users/n/numi-human-resting-evidence-20261005/native-current-recording-review-873/movie-continuity-even-samples.swift"), "dense_sample_decoder":sha(dense/"decoder-25-samples.swift")},"movie_decode":{k:decode[k] for k in ["status","frames_decoded","surface_audit_rows","nonblank_decoded_frames","nonblank_count_matches_surface_rows","first_decoded_frame_is_black","first_nonblank_pts_s","last_nonblank_pts_s","movie_duration_s","pts_first_s","pts_last_s","pts_monotonic","gap_median_s","gap_max_s","gap_after_startup_gt_1_5x_median_count","gap_after_startup_max_s","dimensions","second_pass_pts_sequence_exact_match"]},"surface_audit":{"rows":len(surfaces),"first_step":steps[0],"last_step":steps[-1],"first_time_s":times[0],"last_time_s":times[-1],"median_sim_time_interval_s":dts[len(dts)//2],"mapping_note":"decode order has one black startup image plus one nonblank image per surface-audit row; movie PTS is not the audit simulation-time coordinate"},"native_layer_mask":{"log_record":mask[0] if len(mask)==1 else mask,"semantic_id":51010,"stable_id":22,"layer_mask":0,"payload_retained":True,"geometry_retained":True,"note":"mask is on the native GPU inspection/render route; MRVPacks may still contain the alias geometry"},"native_timing":{"accepted_steps":10000,"requested_dt_s":0.002,"full_body_simulated_s":float(body_fields["simulated_s"]),"full_body_wall_s":float(body_fields["wall_s"]),"full_body_rtf":float(body_fields["real_time_factor"]),"launcher_wall_seconds":meta.get("wall_seconds"),"throughput_log":throughput,"body_completion_log":body},"physiology_trace":{"rows":len(trace),"start_time_s":float(trace[0]["time_s"]),"start_step":int(trace[0]["step"]),"end_time_s":float(trace[-1]["time_s"]),"end_step":int(trace[-1]["step"]),"completed_breath_counter":int(float(trace[-1]["breaths"])),"breath_counter_transitions":changes("breaths"),"complete_filling_ejection_cycles":int(float(trace[-1]["complete_filling_ejection_cycles"])),"cycle_counter_transitions":changes("complete_filling_ejection_cycles"),"summary":{k:stats(k) for k in ["lung_volume_ml","airflow_ml_s","SaO2","aortic_ejected_ml","pulmonary_ejected_ml","last_lv_stroke_ml","last_complete_breath_inspired_ml","root_assistance_n","root_assistance_nm"]},"scope":"native model counters, not clinical validation"},"representative_layer_frames":layers,"geometry_identity_reference":{"path":"/Users/n/numi-human-resting-evidence-20261005/native-retired-alias-geometry-identity-018/comparison.json","sha256":sha("/Users/n/numi-human-resting-evidence-20261005/native-retired-alias-geometry-identity-018/comparison.json")},"limits":["PTS is monotonic but variable; the decoder verifies complete readable output and nonblank count alignment, not constant frame rate.","The PNGs show only the native GPU layer route at the selected frames.","No claim that the native layer mask deletes alias geometry from arbitrary MRVPacks.","Rendered appearance, 20-second counters, and long-horizon physiological/anatomical acceptance are separate evidence."]}
p=out/"review-report.json"; p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(str(p))
print(json.dumps(report,indent=2,sort_keys=True))

