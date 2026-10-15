#!/usr/bin/env python3
import importlib.util
import struct
import tempfile
from pathlib import Path
import numpy as np

script=Path(__file__).with_name('audit_v1_parser.py')
spec=importlib.util.spec_from_file_location('audit_capture_tested',script)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def fixture(version=2, bad_trailing=False):
    E,N,U,cap=2,3,12,64
    header=struct.pack('<8I',mod.MAGIC,version,E,N,U,cap,13 if version==2 else 12,0)
    sizes=[]
    for _ in mod.VECTOR_NAMES: sizes.append(U*16)
    sizes += [U*4,E*16,E*16,E*4,E*4]
    if version==2: sizes += [E*N*3*16]
    buf=bytearray(header)
    for size in sizes:
        at=mod.align16(len(buf)); buf.extend(b'\0'*(at-len(buf))); buf.extend(b'\0'*size)
    if bad_trailing: buf.extend(b'X')
    return bytes(buf)

def test_parse_v2_segments_and_offsets():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'x.bin'; p.write_bytes(fixture())
        meta,a=mod.parse_cgls(p)
        assert meta['version']==2 and meta['segment_count']==13
        assert a['right_scale_rows'].shape==(2,3,3,4)
        assert a['x'].shape==(12,4)
        assert meta['segments'][-1]['name']=='right_scale_rows'

def test_rejects_version1_for_right_scale_audit():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'x.bin'; p.write_bytes(fixture(version=1))
        try: mod.parse_cgls(p)
        except ValueError: pass
        else: raise AssertionError('v1 must be rejected by right-scale audit')

def test_rejects_trailing_bytes():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'x.bin'; p.write_bytes(fixture(bad_trailing=True))
        try: mod.parse_cgls(p)
        except ValueError: pass
        else: raise AssertionError('unexpected trailing bytes must reject')

def test_signed_zero_report_is_not_bitwise_identity():
    report=mod.floats_equal_report(np.array([-0.0],np.float32),np.array([0.0],np.float32))
    assert report['numeric_mismatch_count']==0
    assert report['fp32_word_mismatch_count']==1
    assert report['signed_zero_only_mismatch_count']==1

if __name__=='__main__':
    for f in (test_parse_v2_segments_and_offsets,test_rejects_version1_for_right_scale_audit,test_rejects_trailing_bytes,test_signed_zero_report_is_not_bitwise_identity): f()
    print('PASS 4 tests')
