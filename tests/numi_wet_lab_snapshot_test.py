"""Filesystem consistency and relocation; actual native replay is qualified separately."""
import contextlib,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python/metalrobo'))
import wet_lab_snapshot as snapshots
from wet_lab_shared import write

class Snapshot(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.home=self.base/'labs';self.owner=self.base/'owner';(self.owner/'Tools/VirtualWetLab').mkdir(parents=True);(self.owner/'Tools/VirtualWetLab/laboratory.py').write_text('# retained scientific owner\n')
  root=self.base/'workspace';(root/'shared').mkdir(parents=True);self.path=root/'shared/workspace.json';write(self.path,{'revision':4,'operations':[],'drafts':[{'id':'card','status':'sealed'}],'history':[]});(root/'record').mkdir();self.record=root/'record/seal.json';self.record.write_bytes(b'{"immutable":true}\n');(root/'artifact').mkdir();write(root/'artifact/assay.json',{'specimens':[]})
  self.store=SimpleNamespace(root=root,path=self.path,config={'vivoRoot':str(self.owner),'assays':{'assay':str(root/'artifact/assay.json')}},reconcile=lambda:None,lock=contextlib.nullcontext,alive=lambda op:False)
  self.home_patch=patch.object(snapshots,'HOME',self.home);self.home_patch.start()
 def tearDown(self):self.home_patch.stop();self.temp.cleanup()
 def test_revision_and_live_worker_rejected(self):
  with self.assertRaisesRegex(ValueError,'changed'):snapshots.snapshot(self.store,3)
  write(self.path,{'revision':4,'operations':[{'id':'op'}]});self.store.alive=lambda op:True
  with self.assertRaisesRegex(ValueError,'live operation'):snapshots.snapshot(self.store,4)
  self.assertFalse((self.home/'snapshots').exists())
 def test_restore_preserves_sealed_bytes_and_stops_running_work(self):
  state=json.loads(self.path.read_text());state['operations']=[{'id':'op','draftID':'card','status':'running','pid':99}];write(self.path,state)
  saved=snapshots.snapshot(self.store,4);expected=self.record.read_bytes();self.store.root.rename(self.base/'unavailable')
  result=snapshots.restore(saved['id'],'relocated');dest=self.home/'relocated';restored=json.loads((dest/'workspace/shared/workspace.json').read_text())
  self.assertEqual((dest/'workspace/record/seal.json').read_bytes(),expected);self.assertEqual(restored['operations'][0]['status'],'interrupted');self.assertNotIn('pid',restored['operations'][0]);self.assertTrue(result['freshSessionRequired']);self.assertTrue(json.loads((dest/'restore.json').read_text())['configuration']['assays']['assay'].startswith(str(dest)))
 def test_tampered_manifest_member_rejected(self):
  saved=snapshots.snapshot(self.store,4);folder=self.home/'snapshots'/saved['id'];(folder/'workspace/record/seal.json').write_text('changed')
  with self.assertRaisesRegex(ValueError,'hash mismatch'):snapshots.verify(folder)
 def test_unregistered_snapshot_file_rejected(self):
  saved=snapshots.snapshot(self.store,4);folder=self.home/'snapshots'/saved['id'];(folder/'unexpected').write_text('unregistered')
  with self.assertRaisesRegex(ValueError,'unregistered'):snapshots.verify(folder)

if __name__=='__main__':unittest.main()
