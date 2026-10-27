import importlib.util
from pathlib import Path
import pytest
P=Path(__file__).with_name("prepare_matched613.py")
s=importlib.util.spec_from_file_location("prep613_tests",P)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_replaces_one_flag_value():
    a=["--trust-mm","2","--output","old"]
    m.replace_cli_value(a,"--trust-mm",4)
    assert a==["--trust-mm","4","--output","old"]
@pytest.mark.parametrize("a",[
    ["--trust-mm","1","--trust-mm","2"],
    ["--trust-mm"],
    ["--other","1"],
])
def test_duplicate_or_missing_flags_fail(a):
    with pytest.raises(ValueError):m.replace_cli_value(a,"--trust-mm","4")
def test_guard_and_ancestry_context_roles_are_separate(tmp_path):
    g=tmp_path/"guard.npz";p=tmp_path/"parent.npz"
    g.write_bytes(b"guard");p.write_bytes(b"parent")
    actual_guard,ancestry=m.context_roles(
        {"candidate_path":str(g),"candidate_sha256":m.sha(g)},
        {"candidate":str(p),"candidate_sha256":m.sha(p)})
    assert actual_guard==g and ancestry==p and actual_guard!=ancestry
def test_bad_guard_pin_fails(tmp_path):
    g=tmp_path/"g";p=tmp_path/"p";g.write_bytes(b"g");p.write_bytes(b"p")
    with pytest.raises(ValueError,match="guard context"):
        m.context_roles({"candidate_path":str(g),"candidate_sha256":"0"*64},
                        {"candidate":str(p),"candidate_sha256":m.sha(p)})


def test_empty_baseline_requires_all_outside_depth_and_complete_185():
    s={"candidate_sha256":"x"}
    p={"candidate_sha256":"x","complete_surface_audit":True,"exact_PLC_surface_pairs":0,"inside_node_bank_pairs":0}
    f={"candidate_sha256":"x","complete":True,"bank_count":185,"expected_bank_count":185}
    d={"candidate_sha256":"x","inside_row_count":0,"outside_row_count":3,"complete_bank_accounting_verified":True,"bank_count":185}
    assert m.validate_empty_baseline(s,p,f,d,"x")["contained_pairs"]==0
    d["inside_row_count"]=1
    with pytest.raises(ValueError):m.validate_empty_baseline(s,p,f,d,"x")

def test_normalize_constraint_context_keeps_guard_paths(tmp_path):
    c=tmp_path/"candidate";g=tmp_path/"guard.npz";gr=tmp_path/"guard.json";sf=tmp_path/"surface.json";seed=tmp_path/"seed.json"
    for p,b in ((c,b"c"),(g,b"g"),(gr,b"gr"),(sf,b"sf"),(seed,b"seed")):p.write_bytes(b)
    refresh=tmp_path/"refresh.json"
    refresh.write_text(__import__("json").dumps({"candidate_sha256":m.sha(c),"fixed_guard_payload_contains_no_current_crossings":True}))
    ctx={"candidate":str(c),"candidate_sha256":m.sha(c),"refresh_report":str(refresh),"refresh_report_sha256":m.sha(refresh),"surface_report":str(sf),"surface_report_sha256":m.sha(sf),"guard_npz":str(g),"guard_report":str(gr),"seed_report":str(seed)}
    cp=tmp_path/"ctx.json";cp.write_text(__import__("json").dumps(ctx))
    normalized,original=m.normalize_guard_context(cp)
    assert normalized["candidate_path"]==str(c)
    assert normalized["fixed_guard_npz_path"]==str(g)
    assert normalized["audit159_path"]==str(sf)

def test_fit_entry_replacement_is_explicit(tmp_path):
    actual=tmp_path/"fit.py";actual.write_text("# pinned")
    argv=[str(m.PYTHON),"/old/fit.py","--face-target-report","old.json"]
    m.set_fit_entry(argv,actual)
    assert argv[1]==str(actual)
    with pytest.raises(ValueError):m.set_fit_entry(["/other/python","old.py"],actual)
