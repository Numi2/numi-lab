"""Execute the production telemetry helper in a small CPU C++ harness."""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile

HERE = Path(__file__).parent
SOURCE = Path('/Users/home/numi-cardboard-20261006/matter/tools/cardboard_probe.mm')
text = SOURCE.read_text()
start = text.index('bool commandedRightGripConstrained(')
opening = text.index('{', start)
depth = 0
for end in range(opening, len(text)):
    depth += (text[end] == '{') - (text[end] == '}')
    if depth == 0:
        break
helper = text[start:end + 1]
harness = r'''
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cmath>
struct nm_float4 { float x=0, y=0, z=0, w=0; };
struct MeshSource { std::vector<nm_float4> restNodes; std::vector<std::uint32_t> rightGripNodes; };
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
HELPER
int main() {
    int checks = 0;
    auto check = [&](bool value) { if (!value) throw std::runtime_error("assertion"); ++checks; };
    MeshSource mesh{{{}, {}, {}}, {0, 2}};
    std::vector<nm_float4> targets(6);
    for (int n=0; n<3; ++n) targets[n].w=1;
    for (int n=3; n<6; ++n) targets[n].w=2;
    check(commandedRightGripConstrained(targets, mesh, 0));
    check(!commandedRightGripConstrained(targets, mesh, 1));
    auto rejected = [&](auto work) {
        bool threw = false;
        try { work(); } catch (const std::runtime_error&) { threw = true; }
        check(threw);
    };
    rejected([&] { commandedRightGripConstrained(targets, mesh, 2); });
    targets[2].w=2;
    rejected([&] { commandedRightGripConstrained(targets, mesh, 0); });
    targets[2].w=1;
    targets[0].w=0;
    rejected([&] { commandedRightGripConstrained(targets, mesh, 0); });
    targets[0].w=std::numeric_limits<float>::quiet_NaN();
    rejected([&] { commandedRightGripConstrained(targets, mesh, 0); });
    targets[0].w=1;
    mesh.rightGripNodes={0, 3};
    rejected([&] { commandedRightGripConstrained(targets, mesh, 0); });
    mesh.rightGripNodes={};
    rejected([&] { commandedRightGripConstrained(targets, mesh, 0); });
    mesh.rightGripNodes={0};
    mesh.restNodes.clear();
    rejected([&] { commandedRightGripConstrained(targets, mesh, 0); });
    check(checks == 9);
}
'''.replace('HELPER', helper)
(HERE / 'compiled-grip-reader.cpp').write_text(harness)
with tempfile.TemporaryDirectory(prefix='cardboard-grip-reader-') as directory:
    binary = Path(directory) / 'check'
    command = ['clang++', '-std=c++20', '-Wall', '-Wextra', '-Werror',
               str(HERE / 'compiled-grip-reader.cpp'), '-o', str(binary)]
    build = subprocess.run(command, capture_output=True)
    (HERE / 'compiled-grip-reader-build.log').write_bytes(build.stdout + build.stderr)
    assert build.returncode == 0, build.stderr.decode()
    run = subprocess.run([str(binary)], capture_output=True)
    (HERE / 'compiled-grip-reader-run.log').write_bytes(run.stdout + run.stderr)
    assert run.returncode == 0, run.stderr.decode()
report = {'schema': 'numi.cardboard.compiled-grip-reader-check.v1',
          'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'extracted_helper_sha256': hashlib.sha256(helper.encode()).hexdigest(),
          'checks': 10, 'build_exit_code': build.returncode, 'run_exit_code': run.returncode,
          'scope': 'Actual production helper, two environments plus seven rejection cases; no physical solver trajectory.'}
(HERE / 'compiled-grip-reader-report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
