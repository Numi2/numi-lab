import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from open_knee_reference_state import SOURCE_XPLT_SHA256, digest, load_displacement_reference
from compare_open_knee_accepted_displacement import read_checkpoint

class ReferenceAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.array = np.array([[1,2,3], [4,5,6]], dtype='<f4')
        self.metadata = dict(schema='numi.matter.open-knee-xplt-state.v1',
            source_xplt_sha256=SOURCE_XPLT_SHA256, continuation_time=0.05,
            state_index=3, node_count=2, mesh_node_count=2,
            fields=[dict(**{'class':'node'}, dictionary_id=1, shape=[2,3],
                float32_array_sha256=hashlib.sha256(self.array.tobytes()).hexdigest())])
    def reference(self, array=None, metadata=None):
        p = self.root/'reference.npz'
        np.savez(p, node_01=self.array if array is None else array,
                 metadata_json=json.dumps(self.metadata if metadata is None else metadata))
        return p, digest(p)
    def checkpoint(self, time=0.05, source=(1,2), cooked=(0,1), accepted=1):
        header = b'NOKACPT1'+struct.pack('<IId9I',1,accepted,time,2,0,0,0,0,0,0,0,0)
        nodes = b''.join(struct.pack('<4I3f',s,max(0,s-1),c,0,*self.array[i]*0.001)
                         for i,(s,c) in enumerate(zip(source,cooked)))
        p=self.root/'checkpoint.bin';p.write_bytes(header+nodes);return p
    def test_valid_reference_and_cli_units(self):
        p,h=self.reference();meta,a=load_displacement_reference(p,h)
        np.testing.assert_array_equal(a,self.array)
        c=self.checkpoint();out=self.root/'report.json'
        r=subprocess.run([sys.executable,str(Path(__file__).with_name('compare_open_knee_accepted_displacement.py')),str(c),str(p),'--expected-reference-sha256',h,'--output',str(out)],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        report=json.loads(out.read_text());self.assertLess(report['vector_displacement_error_max_m'],1e-9)
        self.assertEqual(report['matter_checkpoint_sha256'],digest(c))
    def test_digest_must_be_independently_frozen(self):
        p,h=self.reference()
        for invalid in ['0'*64,h.upper(),'self-reported']:
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):load_displacement_reference(p,invalid)
        p.write_bytes(p.read_bytes()+b'tamper')
        with self.assertRaises(ValueError):load_displacement_reference(p,h)
    def test_nonfinite_negative_or_boolean_time(self):
        for time in [float('nan'),float('inf'),-1,True]:
            with self.subTest(time=time):
                meta=dict(self.metadata,continuation_time=time);p,h=self.reference(metadata=meta)
                with self.assertRaises(ValueError):load_displacement_reference(p,h)
    def test_malformed_nonfinite_or_wrong_dtype_array(self):
        for a in [np.zeros((2,2),dtype='<f4'),np.zeros((0,3),dtype='<f4'),self.array.astype('f8'),np.full((2,3),np.nan,dtype='<f4')]:
            with self.subTest(shape=a.shape,dtype=str(a.dtype)):
                p,h=self.reference(array=a)
                with self.assertRaises(ValueError):load_displacement_reference(p,h)
    def test_manifest_identity_shape_hash_and_counts(self):
        for change in [dict(source_xplt_sha256='0'*64),dict(node_count=True),dict(mesh_node_count=3),dict(state_index=True),dict(fields=[]),dict(fields=self.metadata['fields']*2),dict(fields=[dict(self.metadata['fields'][0],shape=[1,3])]),dict(fields=[dict(self.metadata['fields'][0],float32_array_sha256='0'*64)])]:
            with self.subTest(change=change):
                p,h=self.reference(metadata=dict(self.metadata,**change))
                with self.assertRaises(ValueError):load_displacement_reference(p,h)
    def test_checkpoint_rejects_duplicate_identity_invalid_time_unaccepted(self):
        for kwargs in [dict(source=(0,2)),dict(source=(1,1)),dict(cooked=(0,0)),dict(time=-1),dict(time=float('nan')),dict(accepted=0)]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):read_checkpoint(self.checkpoint(**kwargs))
    def test_cli_time_mismatch_emits_no_report(self):
        p,h=self.reference();c=self.checkpoint(time=0.1);out=self.root/'report.json'
        r=subprocess.run([sys.executable,str(Path(__file__).with_name('compare_open_knee_accepted_displacement.py')),str(c),str(p),'--expected-reference-sha256',h,'--output',str(out)],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertFalse(out.exists())
        self.assertIn('continuation times differ',r.stderr)

if __name__=='__main__':unittest.main()
