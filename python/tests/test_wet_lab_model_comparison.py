"""Model/condition compatibility is explicit and cannot silently fall back."""
import hashlib,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'metalrobo'))
from wet_lab_shared import SharedWorkspace,Conflict,write

class ComparisonTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);paths={}
  for model in ('old','corrected'):
   folder=self.root/model;folder.mkdir();write(folder/'features.json',['A'])
   a={'family':'learned-spatial-response','id':model,'title':model,'runtime':{'sha256':model},'models':{'neighborhood':{'sha256':model}},'targets':[{'target':'T'}],'populations':[{'id':'P'}],'specimens':[{'id':'S','sourceSHA256':'same-specimen','inferenceSupported':True}],'conditions':[{'id':'measured-endpoint'}]};write(folder/'assay.json',a);paths[model]=str(folder/'assay.json')
  def catalog(p):return json.loads(Path(p).read_text())
  def support(*args):
   if self.store.config.get('vivoRoot'):
    from investigation import population_support
    return population_support(*args)
   return {'candidates':[{'target':'T','role':'direct','canExecute':True}]}
  def inspect(p,s,o):return {'canExecute':all(x=='A' for x in o['genes']),'genes':[{'gene':x,'eligible':x=='A'} for x in o['genes']]}
  def validate(p,s):
   if s['condition']!='measured-endpoint':raise ValueError('Unsupported condition')
   if not inspect(p,s,s['objective'])['canExecute']:raise ValueError('Unsupported genes')
   if any(t!='T' for t in s['targets']) or not all(c['canExecute'] for c in support(p,s,s['objective']['genes'])['candidates']):raise ValueError('Unsupported population or target')
   return s
  self.modules=patch.dict(sys.modules,{'laboratory':types.SimpleNamespace(catalog=catalog,capabilities=lambda p:{**catalog(p),'prediction':True,'presentation':'tissue'},inspect_objective=inspect,validate_selection=validate,population_support=support,scoring_owner=lambda p:Path(__file__)),'wetlab':types.SimpleNamespace(sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest())});self.modules.start()
  self.store=SharedWorkspace(self.root,{'assays':paths});s=self.store.state();s['selection']={'assayID':'corrected','specimenID':'S','populationID':'P'};write(self.store.path,s)
 def tearDown(self):self.modules.stop();self.tmp.cleanup()
 def act(self,action,**kw):return self.store.mutate({'action':action,'expectedRevision':self.store.state()['revision'],**kw})
 def test_unsupported_condition_is_editable_with_supported_correction(self):
  d=self.act('propose',genes=['A'],targets=['T'],modelIDs=['old','corrected'],conditionIDs=['unmeasured'])['drafts'][0]
  self.assertFalse(d['coverage']['canExecute']);self.assertEqual(d['axes'][0]['corrections'],[{'conditionID':'measured-endpoint'}])
  with self.assertRaises(ValueError):self.act('seal',id=d['id'])
  fixed=self.act('edit',id=d['id'],conditionIDs=['measured-endpoint'])['drafts'][0];self.assertTrue(fixed['coverage']['canExecute']);self.assertEqual(len(fixed['axes']),2);self.assertNotEqual(fixed['axes'][0]['binding']['runtimeSHA256'],fixed['axes'][1]['binding']['runtimeSHA256'])
 def test_missing_model_does_not_use_regression(self):
  d=self.act('propose',genes=['A'],targets=['T'],modelIDs=['missing'])['drafts'][0];self.assertFalse(d['coverage']['canExecute']);self.assertEqual(d['axes'][0]['modelID'],'missing')
 def test_artifact_replacement_rejected_before_execution(self):
  d=self.act('propose',genes=['A'],targets=['T'])['drafts'][0];p=self.root/'corrected/assay.json';a=json.loads(p.read_text());a['runtime']['sha256']='replacement';write(p,a)
  with self.assertRaisesRegex(Conflict,'artifact changed'):self.act('seal',id=d['id'])
  self.assertFalse(self.store.state()['operations'])
 def test_unsupported_marker_requires_explicit_objective_revision(self):
  d=self.act('propose',genes=['A','missing'],targets=['T'])['drafts'][0];self.assertFalse(d['coverage']['canExecute']);self.assertEqual(d['genes'],['A','missing']);self.assertTrue(d['axes'][0]['corrections'][0]['requiresObjectiveEdit'])

 def test_unknown_target_intent_retained_until_explicit_edit(self):
  d=self.act('propose',genes=['A'],targets=['T','unknown'])['drafts'][0]
  self.assertEqual(d['targets'],['T','unknown']);self.assertFalse(d['coverage']['canExecute'])
  with self.assertRaises(ValueError):self.act('seal',id=d['id'])
  fixed=self.act('edit',id=d['id'],targets=['T'])['drafts'][0]
  self.assertTrue(fixed['coverage']['canExecute']);self.assertEqual(fixed['undo'][-1]['targets'],['T','unknown'])

 def test_registered_target_without_population_support_is_blocked(self):
  self.store.config['vivoRoot']='native-owner'
  support={'candidates':[{'target':'T','role':'direct','canExecute':False}]}
  with patch.dict(sys.modules,{'investigation':types.SimpleNamespace(population_support=lambda *args:support,model_difference=lambda d:{'available':False})}):
   d=self.act('propose',genes=['A'],targets=['T'])['drafts'][0]
   self.assertFalse(d['coverage']['canExecute']);self.assertEqual(d['axes'][0]['corrections'][0]['removeUnsupportedTargets'],['T'])
   with self.assertRaises(ValueError):self.act('seal',id=d['id'])

if __name__=='__main__':unittest.main()
