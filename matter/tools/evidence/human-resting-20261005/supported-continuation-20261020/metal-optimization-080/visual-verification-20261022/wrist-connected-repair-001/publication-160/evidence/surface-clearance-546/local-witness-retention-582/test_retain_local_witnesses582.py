import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse

HERE=Path(__file__).resolve().parent
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
D=ROOT/'D/surface-clearance-546'
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
m=load_module('retention582',HERE/'retain_local_witnesses582.py')

def test_config_defaults_and_invalid_values():
 assert m.validate_config(.5,4)==(.5,4.)
 assert m.validate_config(.1,4)==(.1,4.)
 for edge,trust in [(0,4),(-.1,4),(float('nan'),4),(.5,0),(.5,float('inf'))]:
  with pytest.raises(ValueError):m.validate_config(edge,trust)

def test_trust_gate_uses_configured_limit():
 base=np.zeros((2,3),np.float32);trial=base.copy();trial[0,0]=.006
 EG=sparse.csr_matrix((0,6));rhs=np.zeros(0);free=np.array([0,1])
 low=m.trial_metrics(trial,base,base,free,EG,rhs,4.)
 high=m.trial_metrics(trial,base,base,free,EG,rhs,8.)
 assert low['trust_bad_nodes'].tolist()==[0]
 assert high['trust_bad_nodes'].tolist()==[]
 assert low['max_step_mm']==pytest.approx(6.0)

def test_edge_builder_receives_explicit_limit():
 class Owner:
  seen=[]
  @classmethod
  def build_edge_gradient_constraints(cls,base,faces,free,limit):
   cls.seen.append(limit)
   return sparse.csr_matrix((0,3*len(free))),np.zeros(0),None
 base=np.zeros((3,3));faces=np.array([[0,1,2]]);free=np.array([0,1,2])
 m.edge_constraints(Owner,base,faces,free,.1)
 m.edge_constraints(Owner,base,faces,free,.5)
 assert Owner.seen==[.1,.5]

def test_default_gate_parity_on_retained_578_result():
 report_path=D/'witness-restore-578/checked-001/report.json'
 report=json.loads(report_path.read_text())
 assert report['status']=='CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED'
 legacy=D/'local-witness-retention-567/retain_local_witnesses567.py'
 assert report['source_pins'][str(legacy)]==m.V567_SHA
 state_key=next(k for k in report['source_pins'] if k.endswith('/final/state.json'))
 state_path=Path(state_key)
 assert m.sha(state_path)==report['source_pins'][state_key]
 state=json.loads(state_path.read_text())
 base_path=Path(state['candidate'])
 assert m.sha(base_path)==state['candidate_sha256']
 assert m.sha(Path(report['target']))==report['target_sha256']
 candidate=Path(report['candidate_path'])
 assert m.sha(candidate)==report['candidate_sha256']
 b=m.load(base_path);target=m.load(Path(report['target']));trial=m.load(candidate)['points_world_m']
 x=b['points_world_m'];f=b['source_O_faces'];free=np.flatnonzero(np.any(x!=target['points_world_m'],axis=1))
 assert np.array_equal(x[b['typed_coarse_anchor_output_rows_018']],target['points_world_m'][b['typed_coarse_anchor_output_rows_018']])
 owner=m.mod('retention582owner',m.OWNER)
 EG,rhs=m.edge_constraints(owner,x.astype(float)*1000.,f,free,.5)
 R0=ROOT/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz'
 assert m.sha(R0)=='70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31'
 with np.load(base_path,allow_pickle=False) as z:source,_=owner.compose_source_reference(z,R0)
 got=m.trial_metrics(trial,x,source,free,EG,rhs,4.)
 legacy_last=report['iterations'][-1]
 assert len(got['edge_bad_rows'])==legacy_last['edge_bad_rows']==0
 assert got['max_step_mm']==pytest.approx(legacy_last['max_step_mm'],abs=1e-10)
 assert got['max_source_mm']==pytest.approx(legacy_last['max_source_mm'],abs=1e-10)
 assert float(got['edge_slack_mm'].min(initial=1e30))==pytest.approx(legacy_last['minimum_edge_slack_mm'],abs=1e-10)
 assert len(got['trust_bad_nodes'])==len(got['source_bad_nodes'])==0
 assert legacy_last['exact_self_pairs']==0
 assert report['full_final_gate']['pass'] is True
