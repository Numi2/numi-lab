from pathlib import Path
import hashlib,importlib.util,json,os,stat,subprocess
R=Path.cwd();A=R/'artifacts/crow-rl-ccd-128-integration-20261108';B=R/'build/crow-ccd-articulated-sweep-build-01'
G=R/'artifacts/crow-rl-git-recovery-ccd-sweep-20261108';G.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('archive',R/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
report=json.loads((A/'results.json').read_text());residual=json.loads((A/'residual-replay-results.json').read_text())
assert report['native_object_push_regression_passed'] and not report['refinement5mm_pass']
assert all(arm['physical_replay_exact'] for arm in residual['arms'].values())
old=json.loads((A/'release-analysis.json').read_text())
# Correct an analysis label without altering the retained observations. A
# control-boundary height can miss the instant of impact and sample a bounce.
for arm in old['arms'].values():
    for lane in arm['lanes']:
        lane['post_separation_until_first_sample_near_ground_acceleration_range_m_s2']=lane.pop('free_flight_vertical_acceleration_range_m_s2')
old['interpretation_correction']='Earlier free_flight label was too strong: a discrete height sample can miss first ground contact and include a bounce. These are post-separation acceleration ranges, not certified free-flight intervals. The originally labeled report and original analysis code remain recoverable from Git commit3e68f251.'
(A/'release-analysis.json').write_text(json.dumps(old,indent=2,sort_keys=True)+'\n')
report['diagnostic_replays']={'physical32_and64_all_original_trace_final_state_scene_trajectory_and_summary_bytes_exact':True,'additional_published_final_substep_solver_and_joint_velocity_observations':'residual-replay-results.json','interpretation':'Final-substep diagnostics alone do not establish a cause. Low sampled release residuals coexist with cadence-sensitive jaw path and separation; aggregate residuals do not localize jaw rows and omit earlier microsteps.'}
report['integration_decision']='Contact-time and projection correctness repairs integrated on research branch; no crow profile or protected policy promotion. Full release trajectory fails unchanged numerical convergence gate. No further substep escalation or solver iteration tuning from these data.'
a.write_json(A/'results.json',report)
(A/'README.md').write_text('''# Integrated crow contact correctness and open convergence gate

The native translating sphere/box contact-time correction and articulated
hybrid future-FK/motion-margin repairs are integrated on the research branch.
The full Release build passes. All4096 primitive records and all128 projection
conditions match the qualified candidates. The native contact owner rejects
substeps0/129, executes all128, and passes its existing controls. Integrated32
and64 full crow traces, final q/v/scene and scene trajectories are byte exact
to the candidate.

The actual128 run accepts640 transitions with zero rejection. All four poses
pass the sampled grasp/lift/release gates: minimum lift23.85mm, maximum head
relative slip0.160mm, opposed jaw impulses32/32 hold samples, and sampled jaw
penetration below0.040mm. These are prescribed controls on a rigid surrogate,
not ingestion, learned manipulation or biological calibration.

The unchanged full-trajectory numerical gate FAILS:64vs128 differs by62.607mm
(limit5mm);32vs64 previously differed by118.168mm. The largest error is during
release/landing. Sampled jaw separation starts at ticks117,114..115 and113 for
32,64 and128; joint paths differ before release. Some control-boundary height
samples miss an impact and include bounce. See the corrected labels in
release-analysis.json; no certified continuous release/impact time is claimed.

Two output-only diagnostic replays (32/64) reproduce every original physical
trace/final/trajectory byte. Published final-substep residuals and joint
velocities are retained in the payload. They do not establish solver iteration
count or CCD split handling as the cause; aggregate residuals omit earlier
microsteps and do not isolate jaw rows. Source inspection also finds that the
hybrid event rematerializer documents an explicit-drive assumption while this
fixture uses implicit drives. That is an untested owning-source question, not
an explanation proven by these runs.

The first object-push invocation failed before physics because the retained
runtime omitted RetinalStream.metallib. Its failure and the packaging preflight
mistake are preserved. After adding the exact built retinal shader, one
unchanged native owner execution passes7296 transitions, solved contact/miss
controls, RGB visibility, FP64 reward checks, exact partition and world/retina
continuation. No physical failure was silently retried.

No crow profile or protected policy was promoted, no event budget or tolerance
was relaxed, and no further cadence escalation is selected. Full mechanical
trajectory qualification remains open. Git payload recovery and scoped native
reexecution are required before raw evidence is pruned; their receipts live in
artifacts/crow-rl-git-recovery-ccd-sweep-20261108.
''')
# Remove only rebuildable inactive task-local compiler/cache intermediates.
roots=[R/'build'/name for name in ('crow-ccd-event-capture-build-01','crow-ccd-contact-time-build-01','crow-ccd-articulated-sweep-build-01')]
discard=[p for root in roots for p in root.rglob('*') if p.is_file() and p.suffix in ('.air','.pyc')]
inventory=[{'path':str(p.relative_to(R)),'sha256':a.file_hash(p),'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode),'mtime_ns':p.stat().st_mtime_ns} for p in discard]
assert not a.git('ls-files','--',*[f['path'] for f in inventory])
checks=[]
for root in roots:
    h=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(root)],capture_output=True,text=True,timeout=60)
    assert h.returncode in (0,1) and not h.stdout and not h.stderr
    checks.append({'root':str(root.relative_to(R)),'open_handles':0})
before=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
for f in inventory:
    p=R/f['path'];s=p.stat();assert (a.file_hash(p),s.st_size,stat.S_IMODE(s.st_mode),s.st_mtime_ns)==(f['sha256'],f['bytes'],f['mode'],f['mtime_ns'])
for f in inventory:(R/f['path']).unlink()
a.write_json(G/'compiler-cache-cleanup.json',{'files':inventory,'removed_bytes':sum(f['bytes'] for f in inventory),'open_handle_checks':checks,'df_before':before,'df_after':subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True),'retained':'All source inputs, plans, final native binaries, metallibs, toolchain/link receipts and physical outputs remain. Ordinary production build is untouched.'})
(G/'README.md').write_text('''# Git recovery of the crow contact sweep studies

This closes storage for selected-event capture, translating primitive contact
time, articulated projection/margin and128 integration studies. Negative and
inconclusive observations are retained. Verification reads actual Git objects,
including prior deduplicated backing blobs. Scoped executed recovery must pass
before pruning raw roots. This is evidence recovery, not full scientific or
biological environment qualification.
''')
subprocess.run(['git','add',str(A),str(G)],check=True)
subprocess.run(['git','commit','-m','Record failed release convergence and preserve native diagnostic evidence'],check=True)
subprocess.run(['git','push','origin','HEAD'],check=True)
a.capture(A.relative_to(R)/'payload')
subprocess.run(['git','add',str(A/'payload')],check=True)
subprocess.run(['git','commit','-m','Archive crow contact sweep raw evidence with Git deduplication'],check=True)
subprocess.run(['git','push','origin','HEAD'],check=True)
a.verify(A.relative_to(R)/'payload/manifest.json')
