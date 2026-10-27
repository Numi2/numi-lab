import build_joint_outward_targets597_v3 as m
def test_legacy500_report_without_status():
 r=m.terminal_summary597({"crossing_pairs":3,"passing_targets":3},"abc")
 assert r["status"]=="PASS_CERTIFIED_DIRECTION_TARGETS_ONLY"and r["report_sha256"]=="abc"
def test_incomplete_target_summary():
 r=m.terminal_summary597({"crossing_pairs":3,"passing_targets":2},"abc")
 assert r["status"]=="INCOMPLETE_TARGETS"
