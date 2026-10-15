from pathlib import Path
import difflib, gzip, hashlib, json
B=Path('/Users/home/cardboard-implementation-20261014')
O=B/'analysis/repeated-cauchy-selector-overlay-137-001'
p=O/'candidate/runtime.mm'
raw=p.read_bytes()
(O/'initial-selector-draft.mm.gz').write_bytes(gzip.compress(raw,mtime=0))
s=raw.decode()
old='''                                femBaseCauchyFallbackRoot);
                        }
                        std::fprintf(stderr,
                            "fem_base_cauchy_fallback_enabled='''
new='''                                femBaseCauchyFallbackRoot);
                        } else std::fprintf(stderr,
                            "fem_base_cauchy_fallback_enabled='''
assert s.count(old)==1
s=s.replace(old,new)
old='''                    femResidualArmijoTraceRecords >= 64u) return;'''
new='''                    femResidualArmijoTraceRecords >=
                        (femBaseCauchyFallbackSelectedRoot &&
                         femBaseCauchyFallbackAllNewtonIterations
                            ? 512u : 64u)) return;'''
assert s.count(old)==1
s=s.replace(old,new)
old='''                constexpr std::uint32_t maximumTraceRecords = 64u;'''
new='''                const std::uint32_t maximumTraceRecords =
                    femBaseCauchyFallbackSelectedRoot &&
                    femBaseCauchyFallbackAllNewtonIterations ? 512u : 64u;'''
assert s.count(old)==1
s=s.replace(old,new)
p.write_text(s)
base=(O/'base/runtime.mm').read_text()
(O/'selector.patch').write_text(''.join(difflib.unified_diff(
    base.splitlines(True),s.splitlines(True),
    fromfile='a/matter/src/runtime.mm',tofile='b/matter/src/runtime.mm')))
j=json.loads((O/'integration.json').read_text())
j['candidate_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
j['scope']+=' Repeated-fallback selected control raises only its bounded diagnostic record ceiling from64 to512, retaining both direction passes for up to256 Newton iterations. The exact-state enable event is emitted only for exact selectors; all selectors have a separate explicit event.'
(O/'integration.json').write_text(json.dumps(j,indent=2)+'\n')
(O/'refine.py').write_bytes(Path(__file__).read_bytes())
print(j['candidate_sha256'])
