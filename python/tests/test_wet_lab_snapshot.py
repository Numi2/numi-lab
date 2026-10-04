"""Restoration preserves sealed bytes and never silently resumes execution."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'metalrobo'))
from wet_lab_shared import SharedWorkspace,write,read
import wet_lab_snapshot as snapshots

class SnapshotTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.home=patch.object(snapshots,'HOME',self.root/'labs');self.home.start();self.versions=patch('importlib.metadata.version',return_value='test-environment');self.versions.start()
  vivo=self.root/'vivo/Tools/VirtualWetLab';vivo.mkdir(parents=True);(vivo/'owner.py').write_text('# retained owner\n')
  workspace=self.root/'original';workspace.mkdir();self.store=SharedWorkspace(workspace,{'vivoRoot':str(self.root/'vivo'),'assays':{},'binary':None})
  (workspace/'sealed-record').mkdir();(workspace/'sealed-record/registration.json').write_text('{"absoluteProvenance":"/do/not/rewrite"}')
  state=read(self.store.path);state['drafts']=[{'id':'draft','status':'draft','genes':['FOS'],'targets':['T']}];write(self.store.path,state)
 def tearDown(self):self.versions.stop();self.home.stop();self.temp.cleanup()
 def test_archive_restore_without_original_or_snapshot(self):
  result=snapshots.snapshot(self.store,0);identifier=result['id'];snapshots.export(identifier)
  import shutil
  shutil.rmtree(self.store.root);shutil.rmtree(snapshots.HOME/'snapshots'/identifier)
  restored=snapshots.restore(identifier,'elsewhere');self.assertEqual(restored['status'],'restored')
  root=snapshots.HOME/'elsewhere/workspace';self.assertEqual(read(root/'shared/workspace.json')['drafts'][0]['genes'],['FOS']);self.assertEqual((root/'sealed-record/registration.json').read_text(),'{"absoluteProvenance":"/do/not/rewrite"}')
  self.assertFalse((root/'wet-lab-active.json').exists())
 def test_live_output_cannot_be_snapshotted(self):
  state=read(self.store.path);state['operations']=[{'id':'op','draftID':'draft','status':'running'}];write(self.store.path,state);self.store.alive=lambda op:True
  with self.assertRaisesRegex(ValueError,'Stop or finish'):snapshots.snapshot(self.store,0)
 def test_tamper_rejected(self):
  result=snapshots.snapshot(self.store,0);folder=snapshots.HOME/'snapshots'/result['id'];(folder/'workspace/sealed-record/registration.json').write_text('{}')
  with self.assertRaisesRegex(ValueError,'hash mismatch'):snapshots.restore(result['id'],'reject')
 def test_stale_snapshot_rejected(self):
  with self.assertRaisesRegex(ValueError,'Workspace changed'):snapshots.snapshot(self.store,1)
if __name__=='__main__':unittest.main()
