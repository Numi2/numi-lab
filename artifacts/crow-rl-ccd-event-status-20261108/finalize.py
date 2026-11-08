from pathlib import Path
import hashlib,json,os,shutil,stat,subprocess
R=Path.cwd();A=R/'artifacts/crow-rl-ccd-event-status-20261108';B=R/'build/crow-ccd-event-status-build-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
shutil.copy2(B/'run.py',A/'run.py');shutil.copy2(__file__,A/'finalize.py')
for src in (Path('/private/tmp/crow-ccd-event-status-prepare.stdout'),Path('/private/tmp/crow-ccd-event-status-prepare.stderr')):
    if src.exists():shutil.copy2(src,A/src.name)
(A/'README.md').write_text('''# First CCD event failure preservation

The CCD initializer cleared event counters after an environment had already
failed, while later statically encoded substeps skipped its physical work.
It now returns immediately for non-success status. The accepted dynamics,
rollback and public failure stamp are unchanged on the retained clamp failure.

The production native probe checks both hybrid and disabled CCD, all thirteen
non-success codes, two generations and an out-of-range lane: 56 complete-buffer
preservation checks pass. All four healthy initializer records match the old
shader exactly; a repeat is exact. A complete successful open-jaw control has
identical traces, final q/v/scene and scene trajectory. The native contact
regression passes. The initial build invocation lacked CMake on PATH; its log
is retained, followed by the successful full Release build and execution.

With first counters preserved, clamp lane 0 fails at tick 32/substep 19 after
three zero-time event replays, with no time consumed. Lane 2 fails at substep14
after four event passes, with 0.000514183 seconds remaining. The earlier zero
counters did not demonstrate a contact-solver failure. These are event-budget
failures; their responsible projected candidate still requires protected
capture before changing event timing. No event budget or tolerance was raised.

This is diagnostic correctness, not completed physical beak qualification.
The learned policy and protected deployment are unchanged.
''')
pa=R/'artifacts/crow-rl-beak-hold-ccd-fix-20261108'
(pa/'README.md').write_text('''# Four-pose beak clamp/lift/release trial

The sealed six-arm study is inconclusive: clamp and partitioned clamp reject
at tick32; timestep refinement rejects at tick96. Open-jaw, no-contact and
zero-friction controls each complete 640 accepted transitions without native
rejections. Every arm, typed failure, accepted prefix and exact runtime is kept.
No successful full gesture or learned manipulation is claimed.

The later CCD initializer was erasing event counters in failed lanes. The
separate event-status repair reproduces the exact clamp trace/rollback while
preserving the first zero-time and advance-budget failures. Therefore neither
the old zero counters nor absent pair-distance captures identify the original
failure mechanism. See ../crow-rl-ccd-event-status-20261108/results.json.

Object poses vary deterministically, but are not biological replicates. The
no-contact arm disables ground contact too. This diagnostic begins with a rigid
surrogate between the beaks; it does not test retinal control, ground pickup,
food transfer, swallowing or biological calibration.
''')
doc=R/'docs/NUMERICS.md';s=doc.read_text();anchor='## Current accuracy boundary'
assert s.count(anchor)==1
s=s.replace(anchor,'''The MetalWorld CCD event initializer preserves non-success records across
later encoded substeps. This retains first-failure event counters through
transaction rollback. The [native preservation regression](../artifacts/crow-rl-ccd-event-status-20261108/README.md)
checks complete failed buffers, exact healthy output, a repeated clamp failure,
a successful physical control and the contact owner regression. Failure
counters before this correction can have been cleared; zero values alone
cannot rule out a CCD event-budget failure.

'''+anchor);doc.write_text(s)
# These are rebuildable intermediates, not evidence needed for executed recovery.
files=[]
for root in (R/'build/crow-ccd-translation-build-02',R/'build/crow-ccd-event-status-build-01'):
    for p in root.rglob('*'):
        if not p.is_file() or not (p.suffix in ('.air','.pyc')):continue
        assert subprocess.check_output(['git','ls-files','--',str(p.relative_to(R))])==b''
        opened=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        assert opened.returncode in (0,1) and not opened.stdout
        st=p.stat();files.append({'path':str(p.relative_to(R)),'sha256':sha(p),'bytes':st.st_size,'mode':stat.S_IMODE(st.st_mode),'mtime_ns':st.st_mtime_ns})
before=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
for f in files:
    p=R/f['path'];st=p.stat();assert (sha(p),st.st_size,stat.S_IMODE(st.st_mode),st.st_mtime_ns)==(f['sha256'],f['bytes'],f['mode'],f['mtime_ns'])
for f in files:(R/f['path']).unlink()
write(A/'compiler-cache-cleanup.json',{'scope':'Closed untracked rebuildable compiler intermediates only; source, executable shaders and native runtime retained.',
 'files':files,'removed_bytes':sum(x['bytes'] for x in files),'open_handles':0,'df_before':before,'df_after':subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)})
print(json.dumps({'removed_intermediate_bytes':sum(x['bytes'] for x in files),'source_diff_check':subprocess.run(['git','diff','--check']).returncode}))
