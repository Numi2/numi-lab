from pathlib import Path
import hashlib,json
E=Path('/Users/n/numi-human-resting-evidence-20261005')
p=E/'run-resting-release-qualification-026.py'
t=p.read_text()
replacements={
"source = Path('/Users/n/numi-human-resting-passive-release-source-016')":"source = Path('/Users/n/numi-human-resting-hand-reduction-source-018')",
"build = Path('/Users/n/numi-human-resting-passive-release-build-016')":"build = Path('/Users/n/numi-human-resting-hand-reduction-build-018')",
"argv[0] = str(build / 'bin/numi-human-native')":"argv += ['--resting-rigid-hands']\nargv[0] = str(build / 'bin/numi-human-native')",
"files = [":"files = [\n    'include/metalrobo/NumiHumanRestingHandReduction.hpp',\n    'tests/numi_human_resting_hand_reduction_test.cpp',\n    'CMakeLists.txt',",
}
for old,new in replacements.items():
    assert t.count(old)==1,old
    t=t.replace(old,new)
out=E/'run-resting-hand-reduction-032.py'
assert not out.exists()
out.write_text(t)
print(hashlib.sha256(out.read_bytes()).hexdigest())
