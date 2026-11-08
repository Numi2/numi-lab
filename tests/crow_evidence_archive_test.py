"""Large-file capture reuses existing Git chunks without checkout duplication."""
import contextlib
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('crow_archive',Path(__file__).resolve().parents[1]/'tools/crow_evidence_archive.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class ChunkReuseTest(unittest.TestCase):
 def capture(self,payload):
  blobs={'one':gzip.compress(b'abcd',mtime=0),'two':gzip.compress(b'efgh',mtime=0)}
  entries={'artifacts/crow-old/blobs/one.gz':('100644','one'),'artifacts/crow-old/blobs/two.gz':('100644','two')}
  class Objects:
   def __enter__(self):return self
   def __exit__(self,*args):pass
   def read(self,oid):return blobs[oid]
  with tempfile.TemporaryDirectory()as temp:
   previous=Path.cwd()
   try:
    os.chdir(temp);source=Path('.numi/runs/american-crow');source.mkdir(parents=True)
    (source/'first').write_bytes(payload);(source/'second').write_bytes(payload)
    out=Path('artifacts/crow-new')
    with patch.object(a,'CHUNK',4),patch.object(a,'tree',return_value=entries),patch.object(a,'Objects',Objects),patch.object(a,'selected_roots',return_value=[source]),patch.object(a,'git',return_value=b'base\n'),contextlib.redirect_stdout(io.StringIO()):a.capture(out)
    manifest=json.loads((out/'manifest.json').read_text());new=list((out/'blobs').iterdir())
    for p in new:
     oid='new-'+p.name;blobs[oid]=p.read_bytes();entries[str(p)]=('100644',oid)
    for f in manifest['files']:self.assertEqual(b''.join(a.content(f,entries,Objects())),payload)
    self.assertEqual(manifest['files'][0]['backing'],manifest['files'][1]['backing'])
    return manifest,new and [p.name for p in new]
   finally:os.chdir(previous)
 def test_complete_large_file_reuses_existing_chunks(self):
  manifest,new=self.capture(b'abcdefgh')
  self.assertFalse(new);self.assertEqual(manifest['summary']['new_compressed_bytes'],0)
  chunks=manifest['files'][0]['backing']['chunks']
  self.assertEqual([x['path']for x in chunks],['artifacts/crow-old/blobs/one.gz','artifacts/crow-old/blobs/two.gz'])
 def test_only_new_tail_is_materialized_and_shared(self):
  manifest,new=self.capture(b'abcdefghij')
  self.assertEqual(len(new),1)
  self.assertEqual(manifest['summary']['new_compressed_bytes'],len(gzip.compress(b'ij',compresslevel=6,mtime=0)))
  self.assertEqual([x['bytes']for x in manifest['files'][0]['backing']['chunks']],[4,4,2])

if __name__=='__main__':unittest.main()
