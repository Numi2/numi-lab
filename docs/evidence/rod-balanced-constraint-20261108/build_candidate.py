#!/usr/bin/env python3
"""Build an isolated diagnostic metallib using the recorded build commands."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess

root = Path.cwd()
build = Path('/Users/numi/numi-reconciliation-build-20261108')
scratch = Path('/Users/numi/rod-balanced-candidate-20261108')
scratch.mkdir(exist_ok=True)
source = root / 'src/metal/DiscreteElasticRod.metal'
original = source.read_text()
needle = '''        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((dispatch.flags &
             MR_ROD_GPU_FLAG_SELF_COLLISION) != 0u) {
            recordPendingSelfContactCorrection('''
insert = '''        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Research-only: estimate the next stretch projection from the
        // post-sweep compliant-constraint stationarity residual.
        for (uint edge = lane; edge < dispatch.edgeCount; edge += laneCount) {
            const float currentLength = length(positions[edge + 1u] - positions[edge]);
            const float h = dispatch.gravityAndTimestep.w;
            const float alpha = restLengths[edge] / stretchStiffness[edge] / (h * h);
            const float inverseA = inverseMasses[edge];
            const float inverseB = inverseMasses[edge + 1u];
            const float denominator = inverseA + inverseB + alpha;
            const float residual = currentLength - restLengths[edge] +
                alpha * stretchMultipliers[edge];
            const float predicted = abs(residual) / denominator * max(inverseA, inverseB);
            if (!(currentLength > kGeometryFloor) || !(denominator > 0.0f) ||
                !isfinite(currentLength) || !isfinite(denominator) || !isfinite(predicted)) {
                recordFailure(failure, MR_ROD_GPU_NONFINITE_RESULT);
            } else {
                recordPositiveMaximum(maximumCorrectionBits, predicted);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((dispatch.flags &
             MR_ROD_GPU_FLAG_SELF_COLLISION) != 0u) {
            recordPendingSelfContactCorrection('''
if original.count(needle) != 1:
    raise SystemExit('unexpected source shape')
candidate = scratch / 'DiscreteElasticRod.metal'
candidate.write_text(original.replace(needle, insert))
commands = subprocess.check_output(['ninja','-C',str(build),'-t','commands','metalrobo_shaders'], text=True)
parts = [p.strip() for p in commands.split(' && ')]
compile_cmd = next(shlex.split(p) for p in parts if '/src/metal/DiscreteElasticRod.metal' in p and ' -c ' in p)
link_cmd = next(shlex.split(p) for p in parts if p.startswith('xcrun -sdk macosx metallib '))
air = scratch / 'MetalRoboDiscreteElasticRod.air'
metallib = scratch / 'MetalRobo.metallib'
old_source = str(source)
old_air = str(build / 'shaders/MetalRoboDiscreteElasticRod.air')
old_lib = str(build / 'shaders/MetalRobo.metallib')
compile_cmd = [str(candidate) if a == old_source else str(air) if a == old_air else a for a in compile_cmd]
link_cmd = [str(air) if a == old_air else str(metallib) if a == old_lib else a for a in link_cmd]
for cmd in (compile_cmd, link_cmd):
    subprocess.run(cmd, check=True)
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
inputs = [a for a in link_cmd if a.endswith('.air')]
report = dict(compile=compile_cmd, link=link_cmd, source_sha256=digest(source),
              candidate_sha256=digest(candidate), inputs_sha256={p:digest(p) for p in inputs},
              metallib_sha256=digest(metallib))
(scratch / 'build.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
print(json.dumps({'candidate':str(candidate),'metallib':str(metallib),
                  'candidate_sha256':report['candidate_sha256'],
                  'metallib_sha256':report['metallib_sha256']},indent=2))
