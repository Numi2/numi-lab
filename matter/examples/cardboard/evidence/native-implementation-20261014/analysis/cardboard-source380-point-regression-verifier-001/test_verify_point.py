import importlib.util, json, shutil, tempfile, unittest
from pathlib import Path

PKG=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('verify_point',PKG/'verify_point.py')
vp=importlib.util.module_from_spec(spec);spec.loader.exec_module(vp)
ROOT=Path('/Users/home/cardboard-implementation-20261014')
RUN=ROOT/'native/dimensionless-material674-point-390';OUT=RUN/'output'
ARGS={'fixture':Path('/Users/home/numi-cardboard-20261006/matter/examples/cardboard/evidence/native-implementation-20261014/analysis/source380-generalized-material-point-001/native_point_input_674_v2.json'),
'package':ROOT/'solver/explicit32-certified-coupon-380/output/compiled.nmatterpack',
'medium_asset':ROOT/'snapshots/source-scaled-material-merit-118/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter',
'reference_vectors':PKG/'fp64-reference-vectors-674.json','receipt':RUN/'receipt.json',
'result':OUT/'source380-point-run.json','points':OUT/'paper-points-source380_exact_compiled_budget32.jsonl',
'trace':OUT/'tet674-env0-source380_exact_compiled_budget32-trace.json',
'f_readback':ROOT/'native/exact-source380-material674-point-382/output/budget32.json',
'f_readback_manifest':ROOT/'native/exact-source380-material674-point-382/output/point-replay.jsonl'}
class VerifierTests(unittest.TestCase):
 def test_origin390_passes(self):
  r=vp.verify(**ARGS);self.assertEqual(r['status'],'pass');self.assertEqual(r['projection_iterations'],5);self.assertEqual(r['accepted_armijo_trials'],4)
 def _mutated(self,which,mutator):
  with tempfile.TemporaryDirectory() as td:
   td=Path(td);kw=dict(ARGS)
   src=Path(ARGS[which]);dst=td/src.name;shutil.copy2(src,dst);kw[which]=dst
   if which=='points':
    lines=[json.loads(line) for line in dst.read_text().splitlines() if line.strip()]
    obj=next(x for x in lines if x.get('record')=='point');mutator(obj)
    dst.write_text('\n'.join(json.dumps(x) for x in lines)+'\n')
   else:
    obj=json.loads(dst.read_text());mutator(obj);dst.write_text(json.dumps(obj))
   with self.assertRaises(vp.VerificationError):vp.verify(**kw)
 def test_rejects_wrong_package_binding(self):
  self._mutated('result',lambda x:x.__setitem__('compiled_package_sha256','0'*64))
 def test_rejects_trace_parity_mismatch(self):
  self._mutated('result',lambda x:x['direct_trace_vs_trace_off_batch_bit_mismatches'].__setitem__('tangent',1))
 def test_rejects_correction_gate_failure(self):
  def mutate(x):
   idx=8+(int(x['summary']['direct_iterations'])-1)*int(x['trace_layout']['iteration_slots'])+5
   import struct
   value=struct.unpack('<f',struct.pack('<f',3e-8))[0]
   x['trace_f32'][idx]=value
   x['trace_f32_bits'][idx]=f'0x{struct.unpack("<I",struct.pack("<f",value))[0]:08x}'
  self._mutated('trace',mutate)
 def test_rejects_native_tangent_error(self):
  def mutate(x):
   b=x['output_f32_bits'];b[26]='0x00000000';x['tangent_pa_per_s'][0]=0
  self._mutated('points',mutate)
 def test_rejects_f_readback_mismatch(self):
  def mutate(x):x['raw_160_float32_words'][3]='0x00000000'
  self._mutated('f_readback',mutate)
if __name__=='__main__':unittest.main(verbosity=2)
