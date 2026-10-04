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
  self.modules=patch.dict(sys.modules,{'laboratory':types.SimpleNamespace(adapter_for_config=lambda p:types.SimpleNamespace(catalog=lambda p:json.loads(p.read_text()))),'objective_inspection':types.SimpleNamespace(inspect=lambda p,s,g:{'canExecute':all(x=='A' for x in g),'genes':[{'gene':x,'eligible':x=='A'} for x in g]}),'wetlab':types.SimpleNamespace(sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest())});self.modules.start()
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

if __name__=='__main__':unittest.main()
