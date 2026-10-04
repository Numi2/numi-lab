"""State transitions that protect human edits and sealed experiments."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'metalrobo'))
from wet_lab_shared import SharedWorkspace,Conflict,write

class SharedTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.store=SharedWorkspace(self.tmp.name,{'assays':{}})
  self.store.qualify=lambda d:d.update(coverage={'canExecute':True})
  s=self.store.state();s['selection']={'assayID':'a','specimenID':'s','populationID':'p'};write(self.store.path,s)
 def tearDown(self):self.tmp.cleanup()
 def act(self,action,**kw):return self.store.mutate({'action':action,'expectedRevision':self.store.state()['revision'],**kw})
 def test_stale_edits_and_undo(self):
  s=self.act('propose',genes=['A'],targets=['T']);i=s['drafts'][0]['id'];self.act('edit',id=i,genes=['B'])
  with self.assertRaises(Conflict):self.store.mutate({'action':'edit','id':i,'genes':['C'],'expectedRevision':1})
  self.assertEqual(self.store.state()['drafts'][0]['genes'],['B']);self.assertEqual(self.act('undo',id=i)['drafts'][0]['genes'],['A'])
 def test_sealed_immutability_and_revision(self):
  s=self.act('propose',genes=['A'],targets=['T']);i=s['drafts'][0]['id'];s['drafts'][0].update(status='sealed',record='record');write(self.store.path,s)
  with self.assertRaises(ValueError):self.act('edit',id=i,genes=['B'])
  with self.assertRaises(ValueError):self.act('undo',id=i)
  s=self.act('revise',id=i);self.assertEqual(s['drafts'][0]['status'],'sealed');self.assertEqual(s['drafts'][1]['parent'],i);self.assertNotIn('record',s['drafts'][1])
 def test_reveal_not_implied(self):
  s=self.act('propose',genes=['A'],targets=['T']);i=s['drafts'][0]['id'];s['drafts'][0].update(status='sealed',record='record');write(self.store.path,s)
  with self.assertRaisesRegex(ValueError,'explicit user authorization'):self.act('reveal',id=i)
  self.assertFalse(self.store.state()['operations'])
 def test_recovery_refuses_live_worker(self):
  s=self.act('propose',genes=['A'],targets=['T']);i=s['drafts'][0]['id'];s['drafts'][0].update(status='interrupted',operation='op');s['operations']=[{'id':'op','status':'interrupted','action':'seal'}];write(self.store.path,s);self.store.alive=lambda op:True
  with self.assertRaisesRegex(ValueError,'still running'):self.act('recover',id=i)
 def test_model_condition_edits_undo_and_staleness(self):
  s=self.act('propose',genes=['A'],targets=['T'],modelIDs=['old','new'],conditionIDs=['endpoint']);i=s['drafts'][0]['id'];revision=s['revision']
  self.act('edit',id=i,modelIDs=['new'],conditionIDs=['unsupported'])
  with self.assertRaises(Conflict):self.store.mutate({'action':'seal','id':i,'expectedRevision':revision})
  card=self.act('undo',id=i)['drafts'][0];self.assertEqual(card['modelIDs'],['old','new']);self.assertEqual(card['conditionIDs'],['endpoint'])
 def test_revised_comparison_drops_old_axis_results(self):
  s=self.act('propose',genes=['A'],targets=['T']);i=s['drafts'][0]['id'];s['drafts'][0].update(status='sealed',record='r',axisResults=[{'record':'r'}]);write(self.store.path,s)
  s=self.act('revise',id=i);self.assertNotIn('axisResults',s['drafts'][1]);self.assertEqual(s['drafts'][0]['axisResults'],[{'record':'r'}])
 def test_restart_retains_draft(self):
  s=self.act('propose',genes=['A'],targets=['T']);other=SharedWorkspace(self.tmp.name,{'assays':{}});self.assertEqual(other.state(),s)

if __name__=='__main__':unittest.main()
