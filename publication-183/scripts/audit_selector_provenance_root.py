#!/usr/bin/env python3
from pathlib import Path
import os,subprocess,sys
w=Path(__file__).resolve().parent;root=Path("/Users/numi/Work/numi-human-connected-surface-069");c=w/"same-parent-bone-selector653-provenance-001";py=sys.executable;slot="/tmp/numi-human-geometry-slot.py"
for name,script,extra in (("audit-130","audit_connected_outer_geometry_130.py",[]),("audit-159","audit_skin_bone_surfaces_159.py",["--max-seconds","600"])):
 args=[py,slot,py,str(root/script),str(c/"candidate.npz"),str(c/name),*extra];print("running",name,flush=True);subprocess.run(args,check=True)
