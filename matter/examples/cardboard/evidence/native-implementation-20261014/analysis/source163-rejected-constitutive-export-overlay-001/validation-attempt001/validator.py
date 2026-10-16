from pathlib import Path
import hashlib
import json
import shlex
import subprocess

B=Path('/Users/home/cardboard-implementation-20261014')
S=B/'snapshots/source-right-grip-telemetry-157'
O=B/'analysis/source163-rejected-constitutive-export-overlay-001'
P=O/'candidate/matter/tools/cardboard_probe.mm'
V=O/'validation'
V.mkdir(exist_ok=False)
source=P.read_text()

def definition(start, next_start=None):
    first=source.index(start)
    last=source.index(next_start,first) if next_start else source.index('\n}\n',first)+3
    return source[first:last]

writer_start=source.index('        const auto writeArena = ',source.index('const auto writeRejectedRegionalConditioning'))
writer_end=source.index('        writeArena("P",',writer_start)
writer=source[writer_start:writer_end]
harness='''#include "numi/matter/matter.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <type_traits>
#include <limits>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
using namespace numi::matter;
'''
harness+=definition('struct Digest {','std::string jsonEscape')
harness+=definition('Digest sha256(const void*')
harness+=definition('std::string hexDigest(const Digest& digest) {')
harness+=definition('template<class T>\nstd::string sha256Arena')
harness+='''int main(int argc,char** argv) {
 require(argc==3,"output directory and empty/full case required");
 const std::filesystem::path directory=argv[1];
 std::filesystem::create_directories(directory);
 RuntimeStateSnapshot snapshot;
 if(std::string(argv[2])=="full") {
   NMFEMNodeStateGPU n{};
   n.positionAndMass={1.0f,2.0f,3.0f,0.125f};
   n.velocityAndInverseMass={-0.0f,1.0f/3.0f,-2.5f,8.0f};
   n.restAndFixed={4.0f,5.0f,6.0f,2.0f};
   n.referenceDisplacementAndMode={1.0e-8f,-2.0e-8f,3.0e-8f,1.0f};
   snapshot.femRegionalGalerkinLinearizationNodes={n};
   snapshot.femRegionalGalerkinLinearizationMaterialState={-0.0f,1.0f/3.0f,1.0e-30f,-1.0e20f};
   snapshot.femRegionalGalerkinLinearizationSchedulerExponent={0u,1u,3u};
   snapshot.femRegionalGalerkinLinearizationLocalFHigh={{1.0f,0.0f,0.0f,0.0f},{0.0f,1.0f,0.0f,0.0f},{0.0f,0.0f,1.0f,0.0f}};
   snapshot.femRegionalGalerkinLinearizationLocalFLow={{1.0e-8f,-2.0e-8f,3.0e-8f,0.0f},{-4.0e-8f,5.0e-8f,-6.0e-8f,0.0f},{7.0e-8f,-8.0e-8f,9.0e-8f,-0.0f}};
 }
 std::ofstream metadata(directory/"manifest.json");
 require(metadata.good(),"cannot create fixture metadata");
 metadata<<"{\\\"files\\\":[";
 bool first=true;
'''+writer+'''
 metadata<<"]}\\n";
 require(metadata.good(),"cannot close fixture metadata");
 return 0;
}
'''
(V/'actual-writer.cpp').write_text(harness)
commands=[]
def run(args,name):
    q=subprocess.run(args,capture_output=True,text=True)
    (V/(name+'.stdout')).write_text(q.stdout)
    (V/(name+'.stderr')).write_text(q.stderr)
    commands.append({'argv':args,'exit_code':q.returncode})
    assert q.returncode==0,(name,q.stderr)
    return q
checks=[]
for opt in ['0','3']:
    exe=V/('actual-writer-O'+opt)
    run(['clang++','-std=c++20','-O'+opt,'-Wall','-Wextra','-Wpedantic','-Werror','-Wno-deprecated-declarations',
         '-I'+str(S/'matter/include'),'-I'+str(S/'include'),str(V/'actual-writer.cpp'),'-o',str(exe)],'compile-O'+opt)
    for kind in ['full','empty']:
        directory=V/('output-O'+opt+'-'+kind)
        run([str(exe),str(directory),kind],'run-O'+opt+'-'+kind)
        metadata=json.loads((directory/'manifest.json').read_text())
        assert [r['name'] for r in metadata['files']]==[
            'candidate_nodes.bin','linearization_accepted_material_history.bin',
            'linearization_scheduler_exponents.bin','linearization_F_high.bin','linearization_F_low.bin']
        sizes=[80,16,12,48,48] if kind=='full' else [0]*5
        for row,expected in zip(metadata['files'],sizes):
            payload=(directory/row['name']).read_bytes()
            assert len(payload)==row['bytes']==expected
            assert hashlib.sha256(payload).hexdigest()==row['sha256']
        checks.append({'optimization':'O'+opt,'fixture':kind,'arrays':5,'all_bytes_and_sha256_verified':True})
    for name in ['candidate_nodes.bin','linearization_accepted_material_history.bin',
                 'linearization_scheduler_exponents.bin','linearization_F_high.bin','linearization_F_low.bin']:
        if opt=='3':
            for kind in ['full','empty']:
                assert (V/('output-O0-'+kind)/name).read_bytes()==(V/('output-O3-'+kind)/name).read_bytes()
cc=json.loads(Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-157/compile_commands.json').read_text())
rows=[r for r in cc if r['file'].endswith('/matter/tools/cardboard_probe.mm')]
assert len(rows)==1
argv=shlex.split(rows[0]['command'])
clean=[]
skip=False
for arg in argv:
    if skip:
        skip=False
        continue
    if arg=='-o':
        skip=True
    elif arg=='-c':
        clean.append('-fsyntax-only')
    elif arg==rows[0]['file']:
        clean.append(str(P))
    else:
        clean.append(arg)
run(clean,'probe-syntax')
run(['patch','--dry-run','--batch','-p1','-i',str(O/'overlay.patch'),'-d',str(S)],'patch-dry-run')
report={'schema':'numi.cardboard.rejected_constitutive_export_cpu_validation.v1',
        'checks':checks,'writer_extracted_from_candidate':True,
        'probe_syntax_exit_code':0,'patch_dry_run_exit_code':0,
        'extra_gpu_readback_bytes':0,'native_physics_transactions':0,
        'limits':['Actual writer exercised with synthetic bytes only; no native rejected-state capture or trajectory parity was run.',
                  'Exported material history is accepted history at AP linearization, not current projected constitutive state.',
                  'No exact Newton correction or perturbation states are introduced.']}
(V/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(V/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
(O/'validate.py').write_bytes(Path(__file__).read_bytes())
(O/'README.md').write_text('# Preserve rejected AP constitutive inputs (source163)\n\n'
    'This probe-only overlay writes four arrays already present in the bounded RuntimeStateSnapshot: '
    'accepted history at the AP linearization, scheduler rate exponents, and material-local F high/low words. '
    'The existing export retained candidate nodes and AP/RAP while dropping these arrays at process exit. '
    'The new metadata labels accepted history explicitly and marks exact Newton corrections and perturbed projected states unavailable.\n\n'
    'No solver, target generation, constitutive program or Metal source changes; zero added GPU readback bytes. '
    'The existing regional 64 MiB snapshot bound governs availability. '
    'CPU O0/O3 tests use the extracted actual production writer and verify all emitted bytes and SHA-256 values for full and empty arrays. '
    'Probe syntax and source157-relative patch dry-run pass. No native trajectory or capture parity is claimed. '
    'Old failure files cannot be retroactively filled.\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'schema':'numi.cardboard.rejected_constitutive_export_overlay.v1',
          'base_snapshot':str(S),'base_probe_sha256':sha(S/'matter/tools/cardboard_probe.mm'),
          'candidate_probe_sha256':sha(P),'patch_sha256':sha(O/'overlay.patch'),
          'physics_changed':False,'native_run':False,
          'files':{str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file()}}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'checks':len(checks),'manifest_sha256':sha(O/'manifest.json')}))
