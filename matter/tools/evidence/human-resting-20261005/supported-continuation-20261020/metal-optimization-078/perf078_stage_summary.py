import pathlib,re,json,sys,collections
r=pathlib.Path("/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-078")/sys.argv[1]
t=(r/"stderr.log").read_text(); stages=collections.defaultdict(list)
for name,root,ns in re.findall(r"matter_gpu_stage=(\S+) step=(\d+) elapsed_ns=(\d+)",t):stages[name].append(int(ns)*1e-9)
s={"stages":{k:{"spans":len(v),"sum_s":sum(v),"max_s":max(v)} for k,v in stages.items()},"invalid_record_reports":re.findall(r"matter_gpu_timing_invalid_records count=(\d+)",t),"dropped_reports":re.findall(r"matter_gpu_timing_samples_dropped count=(\d+) capacity=(\d+)",t),"unavailable_lines":[l for l in t.splitlines() if "matter_gpu_timing_buffer_unavailable" in l or "matter_gpu_timing_resolution_failed" in l],"note":"Exact observed spans; no scaled estimate or end-to-end speedup inferred."}
(r/"stage-summary.json").write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
