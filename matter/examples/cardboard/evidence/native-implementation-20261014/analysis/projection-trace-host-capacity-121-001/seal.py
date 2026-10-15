from pathlib import Path
import hashlib,json,difflib,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');S=B/'snapshots/source-padded-trial-summary-120';O=B/'analysis/projection-trace-host-capacity-121-001';O.mkdir(exist_ok=False)
r='matter/tools/cardboard_asymmetric_c1_paper_gpu_check.mm';old=(S/r).read_text();assert old.count('kProjectionTraceIterationCapacity=16u;')==1
new=old.replace('kProjectionTraceIterationCapacity=16u;','kProjectionTraceIterationCapacity=32u;')
p=O/'overlay'/r;p.parent.mkdir(parents=True);p.write_text(new)
(O/'overlay.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+r,tofile='b/'+r)))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
common=(S/'matter/src/metal/common.metalinc').read_text();assert 'kMaterialProjectionTraceIterationCapacity = 32u;' in common
assert 'kTraceIterationCapacity=32u;' in (S/'matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm').read_text()
report=dict(schema='numi.cardboard.projection_trace_host_capacity_repair.v1',source_snapshot=str(S),base_sha256=sha(S/r),overlay_sha256=sha(p),path=r,defect='Common Metal trace capacity increased from16 to32 in source118, but C1 host retained16. The tangent diagnostic kernel unconditionally writes all11976 trace words when enabled; the old host allocated5992 trace words. Earlier full builds/CPU tests did not exercise that native trace and do not qualify it.',actual_metal_trace_words=11976,source120_host_trace_words=5992,repaired_host_trace_words=11976,physical_operators_changed=False,acceptance_thresholds_changed=False,native_validation='pending source121 C1 trace under Metal shader validation',other_trace_host='bilinear15 capture host already uses32 iterations/11976words',scope='only C1 diagnostic host allocation, copy and capacity reporting')
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(O/'seal.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file()},indent=2)+'\n')
print(json.dumps(report))
