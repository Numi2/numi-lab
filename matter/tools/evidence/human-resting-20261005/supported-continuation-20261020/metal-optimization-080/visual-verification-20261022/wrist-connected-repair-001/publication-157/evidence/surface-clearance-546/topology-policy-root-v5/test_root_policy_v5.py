from pathlib import Path
import importlib.util,sys
from types import SimpleNamespace
import numpy as np
HERE=Path(__file__).resolve().parent
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def test_actual_lazy_owner_and_five_argument_parity():
 helper=load('v5helper',HERE/'quality_policy025_variant.py')
 ownerpath=HERE.parent.parent/'focused-clearance-541/topology/repair_focused_clearance_541.py'
 def loader(name,path):return load(name,path)
 topology=SimpleNamespace(loadmod=loader)
 installed=helper.install_policy_hooks(topology,helper,ownerpath)
 assert installed==[]
 q=topology.loadmod('v5actual541',ownerpath)
 assert len(installed)==1 and installed[0]is q
 x=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
 f=np.array([[0,1,2],[0,2,3]],np.int64)
 c={'source_O_faces':f,'area_reference_037_source_O_faces':f}
 op={'face_rows':[0,1],'new_faces':[[0,1,3],[1,2,3]]}
 quality=q.face_quality(x,x,f,f)
 a=topology.evaluate_op_local(q,c,op,quality,x,x)
 b=q.evaluate_op(c,op,quality,x,x)
 assert a==b and a[1]=='eligible'
def test_report_writer_actual_json_roundtrip(tmp_path):
 runner=load('v5runner',HERE/'run_quality_policy025_variant.py')
 p=tmp_path/'report.json'
 runner.write_json_roundtrip(p,{'a':1,'text':'hello'})
 import json
 assert json.loads(p.read_text())=={'a':1,'text':'hello'}
 assert p.read_bytes().endswith(b'\n')

