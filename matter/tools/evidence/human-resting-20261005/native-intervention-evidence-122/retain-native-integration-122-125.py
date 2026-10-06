"""Copy compact existing evidence; retain large native assets on the Mini."""
from pathlib import Path
import hashlib, json, shutil

E = Path('/Users/n/numi-human-resting-evidence-20261005')
sha = lambda p: hashlib.file_digest(Path(p).open('rb'), 'sha256').hexdigest()


def bundle(name, scope, retained, remote):
    out = E / name
    out.mkdir(exist_ok=False)
    rows = []
    for source, relative in retained:
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        rows.append({'source': str(source), 'source_sha256': sha(source),
                     'retained': relative, 'retained_sha256': sha(destination)})
    large = [{'source': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in remote]
    manifest = {'scope': scope, 'retention_script_sha256': sha(__file__),
                'retained_files': rows, 'remote_artifacts': large}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'bundle': str(out), 'manifest_sha256': sha(out / 'manifest.json'),
                      'retained_files': len(rows), 'remote_artifacts': len(large)}), flush=True)


study = E / 'native-drive-study-100'
files = [(study / name, name) for name in ('analysis.json', 'verification.json',
          'baseline-invocation.json', 'paired-post-run-diagnostics-120.json')]
files += [(study / 'study/registration.json', 'study/registration.json'),
          (E / 'analyze-native-study-120.py', 'post-run/analyze-native-study-120.py'),
          (E / 'respiratory-mechanical-consistency-117/resting_intervention_study.py',
           'post-run/resting_intervention_study.py'),
          (E / 'native-drive-study-098/resting_intervention_study.py', 'instrument/resting_intervention_study.py'),
          (E / 'native-layer-admission-002/inspect_resting_movie.swift', 'inspect_resting_movie.swift'),
          (Path(__file__), 'retain-native-integration-122-125.py')]
remote = []
for arm in ('resting-baseline', 'resting-drive-half'):
    root = study / 'study/trials' / arm / 'output/scene'
    for name in ('invocation.json', 'native.log', 'resting-coupled.csv', 'resting-surface-audit.csv',
                 'intervention-observation.json', 'movie-inspection.txt', 'frame-initial.png', 'frame-final.png'):
        files.append((root / name, arm + '/' + name))
    remote += [root / 'native-viewer.mov', root / 'resting-human.mrvpack']
    remote += sorted((root / 'accepted-geometry').glob('*.mrvpack'))
for name in ('source-hashes.json', 'source-revisions.json'):
    files.append((E / 'native-drive-study-096' / name, name))
bundle('native-intervention-evidence-122',
       'Frozen exploratory 160-second native paired study100. Known anatomy defects and incomplete preregistered implicit-library binding remain; not final five-minute acceptance.', files, remote)

root = E / 'native-source-reconciliation-123'
files = [(root / name, name) for name in ('invocation.json', 'native.log', 'source.patch',
          'resting-coupled.csv', 'resting-surface-audit.csv', 'frozen026-comparison.json')]
files += [(E / 'run-source-reconciliation-123.py', 'run-source-reconciliation-123.py')]
for name in ('configure.log', 'build.log', 'frozen026-scoped-source-reconciliation.json'):
    files.append((E / 'final-runtime-reconciliation-121' / name, name))
remote = [root / 'native-viewer.mov', root / 'resting-human.mrvpack']
remote += sorted((root / 'accepted-geometry').glob('*.mrvpack'))
bundle('cardiac-runtime-source-evidence-125',
       'Current-main plus five scoped cardiac files built and run on Mini. Same input trajectory and retained geometry bytes as frozen026 run114. Cardiac/respiratory neighbor defects remain; no whole-anatomy acceptance.', files, remote)
