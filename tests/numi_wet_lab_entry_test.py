import json, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python/metalrobo'))
import wet_lab_entry as entry

class Entry(unittest.TestCase):
 def test_stale_discovery_does_not_reconnect_or_expose_token(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'active.json';path.write_text(json.dumps({'pid':99999999,'url':'http://127.0.0.1:1','token':'private'}))
   self.assertEqual(entry.readiness(path),(None,None))
 def test_foreign_url_rejected_without_network(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'active.json';path.write_text(json.dumps({'pid':1,'url':'https://example.com','token':'private'}))
   with patch('urllib.request.urlopen') as network:self.assertEqual(entry.readiness(path),(None,None));network.assert_not_called()
 def test_discovery_ignores_broken_labs(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'good').mkdir();(root/'bad').mkdir();(root/'good/restore.json').write_text(json.dumps({'workspace':'retained'}));(root/'bad/restore.json').write_text('{')
   with patch.object(entry,'HOME',root):self.assertEqual([x['name'] for x in entry.laboratories()],['good'])
 def test_ready_lab_reuses_current_credentials(self):
  with patch.object(entry,'laboratories',return_value=[]),patch.object(entry,'readiness',return_value=({'url':'local','pid':2},{'workspace':'retained'})):
   result=entry.open_laboratory();self.assertEqual(result['status'],'reconnected');self.assertNotIn('token',result)

if __name__=='__main__':unittest.main()
