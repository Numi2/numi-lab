#!/usr/bin/env python3
"""Exercise the K1 camera writer without requiring external robot assets."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
a = p.parse_args()
text = (a.source / 'apps/ai_sapiens_compile.cpp').read_text()
start = text.index('    const auto writeObservation = [&](bool windowCamera) {')
end = text.index('    writeObservation(true);', start) + len('    writeObservation(true);')
writer = text[start:end]
scene = text[text.index('void writeWindowScene('):text.index('void writeVisualPresentation(')]
if 'k1-visual-observation-window.json' not in scene:
    raise SystemExit('Window scene does not select its explicit camera artifact')
with tempfile.TemporaryDirectory(prefix='numi-k1-camera-') as temporary:
    root = Path(temporary)
    cpp = root / 'fixture.cpp'
    exe = root / 'fixture'
    cpp.write_text('#include <filesystem>\n#include <fstream>\n#include <vector>\n#include <stdexcept>\nint main(int argc,char** argv) {\n'
                   'std::filesystem::path output=argv[1];\n'
                   'std::vector<std::filesystem::path> packPaths={"visual/fixture.mrvpack"};\n'
                   + writer + '\n}\n')
    subprocess.run(['clang++', '-std=c++20', str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe), str(root)], check=True)
    old = json.loads((root / 'k1-visual-observation.json').read_text())
    window = json.loads((root / 'k1-visual-observation-window.json').read_text())
    assert old['id'] == 'robotis_ai_sapiens_k1_presentation_v1'
    assert old['camera']['position'] == [0.0, 0.0, 0.0]
    assert old['camera']['orientation'] == [0.0, 0.0, 0.0, 1.0]
    assert window['id'] == 'robotis_ai_sapiens_k1_window_v1'
    assert window['camera']['position'] == [1.05, -1.30, 0.3048154]
    assert window['camera']['orientation'] == [-0.73858354, -0.26102070, 0.20711737, 0.58605883]
    assert old['packs'] == window['packs'] and old['id'] != window['id']
    print('legacy and Window camera artifacts: PASS')
