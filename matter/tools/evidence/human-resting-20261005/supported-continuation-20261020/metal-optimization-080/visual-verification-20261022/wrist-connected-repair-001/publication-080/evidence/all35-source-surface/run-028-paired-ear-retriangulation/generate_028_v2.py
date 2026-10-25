from pathlib import Path
src=Path("/Users/numi/Work/numi-human-connected-surface-069/connected-source-patch-cut-123/run-024-paired-ear-retriangulation/build_024_v3.py")
out=Path("/Users/numi/Work/numi-human-connected-surface-069/connected-source-patch-cut-123/run-028-paired-ear-retriangulation/build_028_apply.py")
s=src.read_text()
repls=[
('run-023-merged-hybrid/hybrid-O-surface-131.npz','run-027-global-hybrid/hybrid-O-surface-133.npz'),
('run-023-merged-hybrid/hybrid-O-surface-131-report.json','run-027-global-hybrid/hybrid-O-surface-133-report.json'),
('run-024-paired-ear-retriangulation','run-028-paired-ear-retriangulation'),
('e70b2fb26447d824abfb20732b2820a3ed079c8fdecad27f580ca14e59f3ea3e','cc11c68d58ec492922c3c86247197e8d83bbaf1fec9f55cc1a6fc3fedbad5273'),
('895224a41a5ae955dfc26d875a15cc0f4faeb54e3fbfadbc454aaadee6d1f9d3','aba83b696f8fe0d8ee967ebd1077f94a49f291aedec94ff0d47ea44af5f54cda'),
('PAIR_FACE, EAR_FACE = 10677, 10678','PAIR_FACE, EAR_FACE = 10761, 10762'),
('PAIR = np.array([5808, 2878, 2879], dtype=np.int64)','PAIR = np.array([5847, 2870, 2871], dtype=np.int64)'),
('EAR = np.array([2879, 2878, 2321], dtype=np.int64)','EAR = np.array([2871, 2870, 2302], dtype=np.int64)'),
('NEW = np.array([[5808, 2878, 2321], [5808, 2321, 2879]], dtype=np.int64)','NEW = np.array([[5847, 2870, 2302], [5847, 2302, 2871]], dtype=np.int64)'),
('assert F.shape == (17423, 3) and X.shape == (8734, 3)','assert F.shape == (17507, 3) and X.shape == (8776, 3)'),
('assert set(new_edges) - set(old_edges) == {(2321, 5808)}','assert set(new_edges) - set(old_edges) == {(2302, 5847)}'),
('assert set(old_edges) - set(new_edges) == {(2878, 2879)}','assert set(old_edges) - set(new_edges) == {(2870, 2871)}'),
('hybrid-O-surface-132','hybrid-O-surface-134'),
('Original seed coverage remains partial: 160/187 seed faces, 34/35 groups; omitted component 1 is unresolved.','All 187/187 original seed faces and 35/35 groups are represented in the input hybrid candidate; full self/bone gates remain pending.'),
('"output_node_row": 2321, "edge_endpoint_rows": [2879, 2878]','"output_node_row": 2302, "edge_endpoint_rows": [2871, 2870]'),
]
for a,b in repls:
 if a not in s: raise SystemExit("missing replacement: "+a)
 s=s.replace(a,b)
with out.open("x") as f: f.write(s)
print(out)
