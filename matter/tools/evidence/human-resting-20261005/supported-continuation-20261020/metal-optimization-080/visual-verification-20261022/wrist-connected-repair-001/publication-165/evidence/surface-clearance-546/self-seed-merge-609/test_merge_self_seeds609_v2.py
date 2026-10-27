from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("seed_merge609",HERE/"merge_self_seeds609_v2.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def test_canonical_union_allows_overlap():
    old=m.canonical_pairs(np.array([[3,1],[2,0]],dtype=np.int32),4,"old")
    new=m.canonical_pairs(np.array([[0,2],[3,0]],dtype=np.int64),4,"new")
    merged,overlap=m.merge_seed_pairs(old,new)
    assert old.tolist()==[[0,2],[1,3]]
    assert merged.tolist()==[[0,2],[0,3],[1,3]]
    assert overlap==[(0,2)]

@pytest.mark.parametrize("pairs,n",[
    (np.array([[0,0]],dtype=np.int64),3),(np.array([[-1,1]],dtype=np.int64),3),
    (np.array([[0,3]],dtype=np.int64),3),(np.array([[0,1],[1,0]],dtype=np.int64),3),
    (np.array([["x","y"]]),3),(np.array([0,1]),3),
])
def test_malformed_pair_payloads_rejected(pairs,n):
    with pytest.raises(ValueError):m.canonical_pairs(pairs,n,"bad")

def test_identity_face_remap_and_changed_topology_reject():
    s=importlib.util.spec_from_file_location("refresh546_test609",m.DEFAULT_REFRESH)
    refresh=importlib.util.module_from_spec(s);sys.modules[s.name]=refresh;s.loader.exec_module(refresh)
    faces=np.array([[0,1,2],[2,1,3],[2,3,4],[0,4,5]],dtype=np.int64)
    got=m.exact_face_remap(refresh,faces,faces.copy(),np.array([[3,0],[1,2]],dtype=np.int64),"fixture")
    assert got["mapped_pairs"].tolist()==[[0,3],[1,2]]
    changed=faces.copy();changed[0]=[0,1,5]
    with pytest.raises(ValueError,match="topology differs"):
        m.exact_face_remap(refresh,faces,changed,np.array([[0,2]],dtype=np.int64),"fixture")

def test_608_report_parity():
    root=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/refined-fit-604/matched-608-001")
    p=root/"self-seeds/seed-preflight.json"
    if not p.is_file():pytest.skip("608 metadata absent")
    r=json.loads(p.read_text())
    old=np.load(r["current_seed_source"]["pairs_path"],allow_pickle=False)
    new=np.load(r["failed_trial_source"]["pairs_path"],allow_pickle=False)
    merged=np.load(r["output_pairs_path"],allow_pickle=False)
    assert len(old)==r["current_seed_source"]["source_pair_count"]
    assert len(new)==r["failed_trial_source"]["source_pair_count"]
    assert len(merged)==r["merged_unique_pair_count"]
    assert len(merged)==len(old)+len(new)-r["deduplication"]["overlap_count"]
    assert r["zero_step_feasible"] and r["zero_step_max_rhs_mm"]<=2e-8


def test_pinned_replace_cli_value_preserves_flag_tokens_and_only_changes_value_slots():
    owner=importlib.util.spec_from_file_location("continue546_replace_test609",m.DEFAULT_CLI_OWNER)
    cli=importlib.util.module_from_spec(owner);sys.modules[owner.name]=cli;owner.loader.exec_module(cli)
    template=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/refined-fit-604/matched-608-001/fit-argv.json")
    if not template.is_file():pytest.skip("608 fit argv absent")
    original=json.loads(template.read_text())
    assert original.count("--self-pairs")==1 and original.count("--output")==1
    updated=m.replace_fit_argv_values(original,"/tmp/609-new-seeds.npy","/tmp/609-fit",cli)
    assert updated.count("--self-pairs")==1 and updated.count("--output")==1
    changed=[i for i,(a,b) in enumerate(zip(original,updated)) if a!=b]
    assert original.index("--self-pairs")==6 and original.index("--output")==61
    assert changed==[7,62]
    assert updated[6]=="--self-pairs" and updated[61]=="--output"
    bad=list(original)+["--self-pairs","duplicate.npy"]
    with pytest.raises(ValueError):
        m.replace_fit_argv_values(bad,"/tmp/new-seeds.npy","/tmp/new-fit",cli)
