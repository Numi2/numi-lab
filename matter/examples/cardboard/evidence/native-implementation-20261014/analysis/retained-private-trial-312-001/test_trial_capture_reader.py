from pathlib import Path
import copy
import hashlib
import json
import tempfile
import unittest
from trial_capture_reader import Capture, CAP

class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.raw = bytes.fromhex('0000803f00000040')
        self.binary = self.root / 'test.coupled.bin'
        self.binary.write_bytes(self.raw)
        sha = lambda b: hashlib.sha256(b).hexdigest()
        self.manifest = dict(schema='numi.matter.coupled_armijo_trial_capture.v1',
            byte_order='little_endian', selected_phase='both', root=1, iteration=2, trial=3,
            max_payload_bytes=CAP, max_arrays=192, array_count=1,
            payload=dict(file='/nonexistent/original/test.coupled.bin', byte_count=8, sha256=sha(self.raw)),
            arrays=[dict(name='sample', phase='pre', dtype='float32', count=2,
                element_bytes=4, offset=0, byte_count=8, sha256=sha(self.raw))])
        self.path = self.root / 'test.coupled.json'

    def tearDown(self):
        self.directory.cleanup()

    def write(self, m=None):
        self.path.write_text(json.dumps(self.manifest if m is None else m))
        return Capture(self.path)

    def test_adjacent_only(self):
        c = self.write()
        self.assertEqual(c.payload_path, self.binary.resolve())
        self.assertEqual(c.f32('sample', 'pre').tolist(), [1., 2.])

    def test_tampered_payload(self):
        self.binary.write_bytes(b'bad-data')
        with self.assertRaises(ValueError): self.write()

    def test_tampered_array_hash(self):
        self.manifest['arrays'][0]['sha256'] = '0' * 64
        with self.assertRaises(ValueError): self.write()

    def test_extent(self):
        self.manifest['arrays'][0]['count'] = 3
        with self.assertRaises(ValueError): self.write()

    def test_duplicate(self):
        self.manifest['arrays'] *= 2
        self.manifest['array_count'] = 2
        with self.assertRaises(ValueError): self.write()

    def test_overlap(self):
        other = copy.deepcopy(self.manifest['arrays'][0])
        other['name'] = 'second'
        self.manifest['arrays'].append(other)
        self.manifest['array_count'] = 2
        with self.assertRaises(ValueError): self.write()

    def test_scalar_stride(self):
        self.manifest['arrays'][0].update(count=1, element_bytes=8)
        with self.assertRaises(ValueError): self.write()

    def test_uint_vector_stride(self):
        self.raw = bytes.fromhex('01000000020000000300000004000000')
        self.binary.write_bytes(self.raw)
        self.manifest['payload'].update(byte_count=len(self.raw),
            sha256=hashlib.sha256(self.raw).hexdigest())
        self.manifest['arrays'][0].update(dtype='uint32x4', count=1, element_bytes=16,
            byte_count=len(self.raw), sha256=hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(self.write().raw('sample','pre'), self.raw)
        self.manifest['arrays'][0].update(count=2, element_bytes=8)
        with self.assertRaises(ValueError): self.write()

    def test_nonfinite_metadata(self):
        self.path.write_text(json.dumps(self.manifest)[:-1] + ',"extra":NaN}')
        with self.assertRaises(ValueError): Capture(self.path)

    def test_symlink(self):
        other = self.root / 'other'
        other.write_bytes(self.raw)
        self.binary.unlink()
        self.binary.symlink_to(other)
        with self.assertRaises(ValueError): self.write()

    def test_bool_count(self):
        self.manifest['arrays'][0]['count'] = True
        with self.assertRaises(ValueError): self.write()

if __name__ == '__main__': unittest.main()
