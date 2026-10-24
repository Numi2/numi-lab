from pathlib import Path
import importlib.util,json
import numpy as np
D=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('f',D/'build_source_fit_006b.py');f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
sp=importlib.util.spec_from_file_location('c',f.CANON);c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
d=c.decode_nhtiss(f.BASE.read_bytes());r=next(x for x in d['records'] if int(x[6])==134);fv,vc,fi,ic=map(int,(r[2],r[3],r[4],r[5]));ix=d['indices'][fi:fi+ic].astype(np.int64).reshape(-1,3);F=ix-fv if ix.min()>=fv and ix.max()<fv+vc else ix
rows,pfaces,free,bedges=f.patch_rows(F,f.SEEDS[134],3);v=d['vertices'][fv:fv+vc]['p'].astype(np.float64);ids=np.asarray(pfaces[13],int);tri=v[ids]*1000;edges=np.array([tri[1]-tri[0],tri[2]-tri[0],tri[2]-tri[1]]);cross=np.cross(edges[0],edges[1]);area=np.linalg.norm(cross);alt=area/max(np.linalg.norm(edges,axis=1).max(),1e-30);print(json.dumps({'patch_face_ordinal':13,'source_face_row':int(rows[13]),'vertex_ids':ids.tolist(),'vertices_m':v[ids].tolist(),'min_altitude_mm':float(alt),'double_area_mm2':float(area),'vertices_free':[int(x) in set(map(int,free)) for x in ids],'free_vertices':free.tolist()},indent=2))
