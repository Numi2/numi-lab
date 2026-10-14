"""Critical shared-state contracts; scientific execution is separately qualified natively."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python/metalrobo'))
from wet_lab_shared import SharedWorkspace,Conflict,write

class Interaction(SharedWorkspace):
 def models(self):return [{'id':'frozen'}]
 def datasets(self):return [{'id':'measured'}]
 def enrich(self,state):return state
 def validate_binding(self,sel):
  if sel.get('conditionID')!='supported':raise ValueError('Unsupported condition')
 def qualify(self,d):d['coverage']={'canExecute':True};d['axes']=[]

class State(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.store=Interaction(self.root,{'assays':{}})
  self.selection={'assayID':'frozen','specimenID':'sample','populationID':'p','conditionID':'supported'}
  self.store.mutate({'action':'selection','expectedRevision':0,'selection':self.selection})
  self.ack=self.store.mutate({'action':'propose','expectedRevision':1,'genes':['FOS'],'targets':['DDIT3']});self.card=self.ack['drafts'][0]
 def tearDown(self):self.temp.cleanup()
 def test_ack_retains_discovery(self):self.assertEqual(self.ack['datasets'],self.store.state()['datasets']);self.assertEqual(self.ack['models'],self.store.state()['models'])
 def test_stale_agent_cannot_replace_human_edit(self):
  self.store.mutate({'action':'edit','expectedRevision':2,'id':self.card['id'],'genes':['JUNB'],'actor':'human'})
  with self.assertRaises(Conflict):self.store.mutate({'action':'edit','expectedRevision':2,'id':self.card['id'],'genes':['FOS'],'actor':'codex'})
  self.assertEqual(self.store.state()['drafts'][0]['genes'],['JUNB'])
 def test_population_edit_and_undo_share_card(self):
  updated={**self.selection,'populationID':'receiving'}
  self.store.mutate({'action':'edit','expectedRevision':2,'id':self.card['id'],'selection':updated})
  self.assertEqual(self.store.state()['selection']['populationID'],'receiving')
  self.store.mutate({'action':'undo','expectedRevision':3,'id':self.card['id']})
  self.assertEqual(self.store.state()['drafts'][0]['selection'],self.selection)
 def test_unknown_condition_does_not_commit(self):
  with self.assertRaises(ValueError):self.store.mutate({'action':'selection','expectedRevision':2,'selection':{**self.selection,'conditionID':'unknown'}})
  self.assertEqual(self.store.state()['revision'],2)
 def test_inspection_does_not_reveal(self):
  state=self.store.state();state['drafts'][0].update(status='sealed',record='f'*32);write(self.store.path,state)
  before=self.store.path.read_bytes()
  with patch('subprocess.Popen') as process:
   with self.assertRaises(ValueError):self.store.mutate({'action':'reveal','expectedRevision':2,'id':self.card['id']})
   process.assert_not_called()
  self.assertEqual(self.store.path.read_bytes(),before)
 def test_sealed_edit_is_immutable(self):
  state=self.store.state();state['drafts'][0]['status']='sealed';write(self.store.path,state)
  with self.assertRaises(ValueError):self.store.mutate({'action':'edit','expectedRevision':2,'id':self.card['id'],'genes':['JUNB']})

if __name__=='__main__':unittest.main()
