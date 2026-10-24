#!/usr/bin/env python3
import hashlib,json,pathlib
import numpy as np
P=pathlib.Path(__file__).resolve().parent
# Canonical 1 mm right triangle represented in mm then emitted in world meters.
source_mm=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],np.float64)
world_m=(source_mm*0.001).astype(np.float32).astype(np.float64)
edge_m=max(np.linalg.norm(world_m[1]-world_m[0]),np.linalg.norm(world_m[2]-world_m[0]),np.linalg.norm(world_m[2]-world_m[1]))
cross_m=np.cross(world_m[1]-world_m[0],world_m[2]-world_m[0])
altitude_mm=float(np.linalg.norm(cross_m)/edge_m*1000.0)
source_area_mm2=float(np.linalg.norm(np.cross(source_mm[1]-source_mm[0],source_mm[2]-source_mm[0])))
world_area_mm2=float(np.linalg.norm(cross_m)*1.0e6)
area_ratio=world_area_mm2/source_area_mm2
assert abs(altitude_mm-1.0)<1e-6,(altitude_mm)
assert abs(area_ratio-1.0)<1e-6,(area_ratio)
report={'schema':'numi.human.gate-unit-conversion-preflight.v1','fixture':'1 mm right triangle represented in mm and emitted world meters','source_area_mm2':source_area_mm2,'emitted_world_area_mm2':world_area_mm2,'area_ratio':area_ratio,'minimum_altitude_mm':altitude_mm,'status':'PASS'}
out=P/'gate-unit-preflight-008.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));print('sha256',hashlib.sha256(out.read_bytes()).hexdigest())
