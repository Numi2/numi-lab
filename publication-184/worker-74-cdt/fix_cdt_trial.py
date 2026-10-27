from pathlib import Path
p=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b74-104/steiner-multinode-004-cdt-guided/cdt_patch_trial.py')
s=p.read_text()
needle="# Validate the supplied SciPy constrained-Delaunay arrays independently before any 3D node fitting."
insert="""# Build source-face barycentric ancestry for the fixed boundary identities.
node_bary={}
for fr in old_rows:
 for n,b in zip(F[fr],Z['refinement_face_parent_barycentric'][fr]):node_bary[int(n)]=np.asarray(b,np.float64)
if set(node_bary)!=set(loop):raise RuntimeError('source-face boundary identity mismatch')
corners=[next(n for n,b in node_bary.items() if np.array_equal(b,np.eye(3)[i])) for i in range(3)]
"""
if needle not in s:raise SystemExit('CDT validation marker missing')
s=s.replace(needle,insert+needle,1)
s=s.replace("all_edges={tuple(sorted((int(a),int(b))) for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])) for t in ct}","all_edges={tuple(sorted((int(a),int(b)))) for t in ct for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0]))}")
p.write_text(s)
print('fixed source ancestry setup and CDT edge set construction')

