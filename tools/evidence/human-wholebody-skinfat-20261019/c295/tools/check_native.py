import csv,hashlib,json,math,pathlib,re,sys
root=pathlib.Path(sys.argv[1]).resolve()
run=json.loads((root/"result.json").read_text())
assert run["exit_code"]==0 and not run["timed_out"],"native run did not finish"
inv=json.loads((root/"invocation.json").read_text())
loaded=[pathlib.Path(line.rsplit(" ",1)[-1]).resolve() for line in (root/"stderr.log").read_text().splitlines() if line.startswith("dyld[") and line.endswith("/libmetalrobo.dylib")]
assert loaded==[pathlib.Path(inv["library_path"]).resolve()],"loaded runtime differs from pinned runtime"
registered_sha=hashlib.sha256((root/"registered-skin.csv").read_bytes()).hexdigest()
assert registered_sha==inv["expected_registered_skin_csv_sha256"],"runtime source registration differs from exact admitted coordinate source"
out=root/"output"
rows=list(csv.DictReader((out/"resting-coupled.csv").open()))
assert rows,"no accepted coupled state"
last_time=0.; last_step=0
for row in rows:
    values={k:float(v) for k,v in row.items()}
    assert all(math.isfinite(v) for v in values.values()),"nonfinite accepted physiology"
    assert values["time_s"]>last_time and values["step"]>last_step,"accepted history is not monotonic"
    assert values["root_assistance_n"]==0 and values["root_assistance_nm"]==0,"hidden root assistance"
    assert abs(values["body_mass_residual_kg"])<1e-5,"body mass accounting changed"
    last_time=values["time_s"];last_step=values["step"]
assert last_step==inv["requested_steps"],"native trace did not reach requested accepted step"
assert abs(last_time-inv["requested_steps"]*inv["simulation_dt_s"])<1e-8,"accepted clock differs from requested physical time"
text=(root/"stdout.log").read_text()
objects=[]
for line in text.splitlines():
    if line.startswith("resting_FEM_object_capture=summary "):
        fields=dict(re.findall(r"(\w+)=(.*?)(?= \w+=|$)",line))
        for key in ("min_J","max_J","max_displacement_m","max_speed_m_s"):
            assert math.isfinite(float(fields[key])),(key,fields)
        assert float(fields["min_J"])>0,"inverted accepted tissue tet"
        objects.append(fields)
assert objects,"no accepted FEM endpoint observations"
whole=[o for o in objects if "skin" in o["object"] and int(o["tet_count"])>384]
assert whole,"regional skin only: whole-body shell not present"
bound=[dict(re.findall(r"(\w+)=(.*?)(?= \w+=|$)",line)) for line in text.splitlines() if line.startswith("whole_body_skin_fat_material_column_self_contact=bound")]
assert len(bound)==1,"whole-body self-contact specialization was not initialized exactly once"
assert int(bound[0]["nodes"])==int(whole[0]["node_count"]),"self-contact range differs from actual shell"
assert int(bound[0]["columns"])*3==int(bound[0]["nodes"]),"self-contact material-column layout differs"


prepared=[dict(re.findall(r"(\w+)=(.*?)(?= \w+=|$)",line)) for line in text.splitlines() if line.startswith("accepted_fem_boundary_render=prepared ")]
assert len(prepared)==1,"exact inner skin and deep-fat rendering was not initialized exactly once"
layer=prepared[0]
assert layer["self_contact"]=="material_column_bound","inner layers lack bound nonlocal self-contact"
assert layer["accepted_snapshot_owner"]=="Runtime::snapshot","layers do not share accepted snapshot owner"
assert layer["layer_masks"]=="128,256" and layer["rendered_layers"]=="2","inner layers are not independently inspectable"
n=int(layer["nodes_per_layer"])
assert 3*n==int(bound[0]["nodes"]),"render boundaries differ from physical shell"
ranges=[]
for key in ("inner_vertex_range","deep_fat_vertex_range"):
    match=re.fullmatch(r"\[(\d+),(\d+)\)",layer[key])
    assert match,"invalid accepted layer vertex range"
    a,b=map(int,match.groups());assert b-a==n,"render layer count differs from physical shell"
    ranges.append((a,b))
assert ranges[0][1]<=ranges[1][0] or ranges[1][1]<=ranges[0][0],"inner boundary output ranges overlap"

presented_steps=[int(m.group(1)) for m in re.finditer(r"^resting_render_profile step=(\d+) ",text,re.M)]
assert int(last_step) in presented_steps,"final accepted skin/fat presentation did not complete"
receipt={"scope":"short native numerical consistency only; not sustained stability or physiological qualification",
"loaded_pinned_runtime":True,"registered_source_csv_sha256":registered_sha,"accepted_trace_rows":len(rows),"accepted_final_step":last_step,"accepted_simulated_seconds":last_time,
"whole_body_shell":whole,"initialized_inner_boundary_render":layer,"inner_boundary_render_scope":"accepted endpoint GPU presentation completed; no visual inspection or sustained stability claim","accepted_rendered_steps":presented_steps,"FEM_object_count":len(objects),"all_FEM_min_J":min(float(o["min_J"]) for o in objects),
"root_assistance_zero":True,"body_mass_residual_within_1e-5_kg":True,"trace_sha256":hashlib.sha256((out/"resting-coupled.csv").read_bytes()).hexdigest()}
(root/"numerical-check.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
