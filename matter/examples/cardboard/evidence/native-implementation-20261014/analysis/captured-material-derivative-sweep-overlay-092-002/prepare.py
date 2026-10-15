from pathlib import Path
import difflib, hashlib, json, shlex, subprocess

base = Path('/Users/home/cardboard-implementation-20261014')
source = base / 'snapshots/source-private-trial-and-material-sweep-092'
relative = Path('matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm')
out = base / 'analysis/captured-material-derivative-sweep-overlay-092-002'
out.mkdir(exist_ok=False)
original = (source / relative).read_text()
text = original.replace('#include <cmath>\n', '#include <cmath>\n#include <cerrno>\n#include <fcntl.h>\n#include <unistd.h>\n')
guard = '''// Reserve evidence exclusively before dispatch; failures retain partial evidence.
struct ExclusiveSweepOutput {
    int descriptor = -1;
    explicit ExclusiveSweepOutput(const std::filesystem::path& path) {
        descriptor=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
        need(descriptor>=0,"cannot exclusively create controlled sweep output: "+path.string());
    }
    ~ExclusiveSweepOutput(){if(descriptor>=0)::close(descriptor);}
    ExclusiveSweepOutput(const ExclusiveSweepOutput&)=delete;
    ExclusiveSweepOutput& operator=(const ExclusiveSweepOutput&)=delete;
    void writeAll(const std::string& text) {
        need(text.size()<=16u*1024u*1024u,"controlled sweep output exceeds bounded writer");
        std::size_t written=0u;
        while(written<text.size()) {
            const auto count=::write(descriptor,text.data()+written,text.size()-written);
            if(count<0&&errno==EINTR)continue;
            need(count>0,"controlled sweep write failed; partial evidence retained");
            written+=static_cast<std::size_t>(count);
        }
        const int saved=descriptor;descriptor=-1;
        need(::close(saved)==0,"controlled sweep close failed; evidence retained");
    }
};
'''
needle = '// Instrument-only controlled directions around immutable captured centers.\n'
assert text.count(needle) == 1
text = text.replace(needle, guard + needle)
needle = '    const auto samples=derivativeSweepRecords(capture);\n'
assert text.count(needle) == 1
text = text.replace(needle, '    ExclusiveSweepOutput evidence(outdir/("paper-derivative-sweep-"+policy+".jsonl"));\n' + needle)
needle = '    std::ofstream file(outdir/("paper-derivative-sweep-"+policy+".jsonl"));\n    need(file.good(),"cannot write controlled derivative sweep");\n'
assert text.count(needle) == 1
text = text.replace(needle, '    std::ostringstream file;\n')
needle = '        <<"\\\"policy\\\":\\\""<<policy<<"\\\",\\\"capture_sha256\\\":\\\""<<captureHash\n'
assert text.count(needle) == 1
text = text.replace(needle, '''        <<"\\"policy\\":\\""<<policy<<"\\",\\"compiled_local_iterations_liner\\":"<<world.materials[0].localNewtonIterations
        <<",\\"compiled_local_iterations_medium\\":"<<world.materials[1].localNewtonIterations
        <<",\\"capture_sha256\\":\\""<<captureHash
''')
needle = '    need(file.good(),"controlled derivative sweep write failed");\n'
assert text.count(needle) == 1
text = text.replace(needle, '    need(file.good(),"controlled derivative sweep formatting failed");\n    evidence.writeAll(file.str());\n')
target = out / relative.name
target.write_text(text)
(out / 'changes.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='a/'+str(relative),tofile='b/'+str(relative))))
(out / 'prepare.py').write_bytes(Path(__file__).read_bytes())
flags = Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-091/matter/CMakeFiles/numi-matter-cardboard-bilinear15-capture-diagnostic-check.dir/flags.make').read_text()
values = {line.split(' = ',1)[0]: line.split(' = ',1)[1] for line in flags.splitlines() if ' = ' in line}
command = ['/usr/bin/clang++','-fsyntax-only']
for key in ('OBJCXX_DEFINES','OBJCXX_INCLUDES','OBJCXX_FLAGS'):
    command.extend(shlex.split(values[key]))
command.append(str(target))
result = subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(out / 'syntax.log').write_text(result.stdout)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = dict(schema='numi.cardboard.observational-overlay.v1',base_snapshot=str(source),
    destination=str(relative),base_sha256=sha(source/relative),replacement_sha256=sha(target),
    patch_sha256=sha(out/'changes.patch'),syntax_exit=result.returncode,syntax_command=command,
    supersedes='captured-material-derivative-sweep-overlay-091-001',
    reason='Read-only review found truncating output streams in the standalone checker. Source-bound fresh directories protected prior runs; no native sweep had executed. Preserve the original overlay and frozen092 source.',
    exclusive_create=True,header_compiled_budgets=True,production_changes=False,native_executed=False)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(package=str(out),syntax_exit=result.returncode,replacement_sha256=sha(target))))
raise SystemExit(result.returncode)
