from pathlib import Path
import importlib.util,json,shutil,subprocess
R=Path.cwd();B=R/'build/crow-ccd-event-capture-build-01';B.mkdir(exist_ok=True)
A=R/'artifacts/crow-rl-ccd-event-capture-20261108';A.mkdir(exist_ok=True);assert not list(A.iterdir())
spec=importlib.util.spec_from_file_location('archive',R/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
assert not a.git('status','--porcelain')
entries=a.tree()
with a.Objects() as objects:
    m=json.loads(objects.read(entries['artifacts/crow-rl-ccd-event-status-20261108/payload/manifest.json'][1]))
    fby={f['path']:f for f in m['files']}
    sources={'probe':'build/crow-beak-hold-build-02/probe-with-runtime',
        'cases-probe':'build/crow-ccd-translation-build-02/cases_probe',
        'cases.bin':'build/crow-ccd-translation-build-02/cases.bin',
        'expected-grid.bin':'build/crow-ccd-translation-build-02/integrated-grid.bin',
        'old.ccdsweep':'build/crow-ccd-translation-build-02/input/clamp.ccdsweep',
        'old-replay.bin':'build/crow-ccd-translation-build-02/integrated-clamp.bin'}
    for arm,root in [('clamp','build/crow-ccd-event-status-build-01'),('refinement','build/crow-beak-hold-build-02'),('open','build/crow-beak-hold-build-02')]:
        suffixes=['trace.tsv']+(['failure-q.f32','failure-v.f32','failure-scene.bin','failure-world-status.bin','failure.json'] if arm!='open' else ['final-q.f32','final-v.f32','final-scene.bin','scene.bin','summary.json'])
        if arm=='clamp':suffixes+=['failure-contact-status.bin']
        for name in suffixes:sources[f'expected/{arm}/{name}']=f'{root}/{arm}/{name}'
    for target,source in sources.items():
        f=fby[source];p=B/target;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as out:
            for block in a.content(f,entries,objects):out.write(block)
        p.chmod(f['mode']);assert a.file_hash(p)==f['sha256']
T=R/'.numi/retained-runtime/ccd-event-capture-20261108/baseline'
(T/'metalrobo').mkdir(parents=True,exist_ok=False)
for src,dst in [('build/lib/libmetalrobo.dylib','libmetalrobo.dylib'),('build/shaders/MetalRobo.metallib','metalrobo/MetalRobo.metallib'),('build/shaders/CCDSweepProbe.metallib','CCDSweepProbe.metallib'),('build/bin/metalrobo_ccd_sweep_probe','sweep-probe')]:shutil.copy2(R/src,T/dst)
# Protect the selected candidate just before a deterministic event-budget
# failure. This is inspection only; no status or physical work is modified.
p=R/'src/metal/Collision.metal';s=p.read_text();marker='''    statuses[environment] = status;
}

// Presence admits the opt-in protected-tail allocation contract.'''
assert s.count(marker)==1
code='''    if ((dispatch.flags & MR_METAL_WORLD_CONTACT_CAPTURE_CCD_FAILURE) != 0u &&
        status.code == MR_STEP_SUCCESS && storedEventCount != 0u) {
        const MRCCDPairGPU selected = candidates[candidateBase];
        const MRCCDEventStateGPU cursor = eventStates[environment];
        const float timeTolerance = max(dispatch.ccdEventParameters.y, dispatch.ccdParameters.y);
        const float duration = clamp(selected.intervalAndDistance.x, 0.0f, cursor.time.y);
        const bool zeroTimeLimit = duration <= timeTolerance &&
            cursor.zeroTimeReplayCount >= dispatch.maxCCDZeroTimeReplays;
        const bool advanceLimit = cursor.splitCount + 1u >= dispatch.maxCCDAdvanceSolvePasses &&
            cursor.time.y - duration > timeTolerance;
        if (zeroTimeLimit || advanceLimit) {
            const uint base = dispatch.environmentCount * dispatch.ccdCandidateCapacity +
                environment * MR_CCD_FAILURE_SNAPSHOT_CANDIDATES;
            device MRCCDFailureSnapshotGPU& saved =
                *reinterpret_cast<device MRCCDFailureSnapshotGPU*>(candidates + base);
            if (saved.validAndStamp.x == 0u) {
                saved.candidate = selected;
                saved.sourceA = shapes[selected.colliderA];
                saved.sourceB = shapes[selected.colliderB];
                saved.startA = projectedColliders[projectionBase + selected.colliderA];
                saved.startB = projectedColliders[projectionBase + selected.colliderB];
                saved.endA = futureProjectedColliders[projectionBase + selected.colliderA];
                saved.endB = futureProjectedColliders[projectionBase + selected.colliderB];
                saved.event = cursor;
                saved.validAndStamp = uint4(MR_CCD_EVENT_FAILURE_SNAPSHOT_TAG,
                    status.controlStep, status.physicsSubstep, selected.compiledPair);
            }
        }
    }
'''
s=s.replace(marker,code+marker);p.write_text(s)
p=R/'include/metalrobo/engine_types.h';s=p.read_text();s=s.replace('    MR_CCD_FAILURE_SNAPSHOT_TAG = 0x43434631u,','    MR_CCD_FAILURE_SNAPSHOT_TAG = 0x43434631u,\n    MR_CCD_EVENT_FAILURE_SNAPSHOT_TAG = 0x43434632u,');p.write_text(s)
p=R/'src/metal/CCDFailureCapture.inc';s=p.read_text()
old='''            return status.code == MR_STEP_DID_NOT_CONVERGE &&
                status.firstFailingPair != MR_INVALID_INDEX;'''
new='''            const bool pairFailure = status.code == MR_STEP_DID_NOT_CONVERGE &&
                status.firstFailingPair != MR_INVALID_INDEX;
            const bool eventFailure = (status.code == MR_STEP_DID_NOT_CONVERGE ||
                status.code == MR_STEP_CCD_EVENT_BUDGET_EXHAUSTED) &&
                status.firstFailingPair == MR_INVALID_INDEX &&
                status.firstFailingEventKeyLow != MR_INVALID_INDEX &&
                status.firstFailingEventKeyHigh == status.environment;
            return pairFailure || eventFailure;'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('''        const auto& dispatch = context.boundContactDispatch;''','''        const auto& dispatch = context.boundContactDispatch;
        const bool eventFailure = status.firstFailingPair == MR_INVALID_INDEX;
        const auto failedPair = eventFailure ? status.firstFailingEventKeyLow : status.firstFailingPair;''')
s=s.replace('status.firstFailingPair >= dispatch.eligiblePairCount','failedPair >= dispatch.eligiblePairCount')
old='''        if (saved.validAndStamp.x != MR_CCD_FAILURE_SNAPSHOT_TAG ||
            saved.candidate.environment != status.environment ||
            saved.candidate.compiledPair != status.firstFailingPair ||
            (saved.candidate.flags & MR_CCD_PAIR_UNRESOLVED) == 0u ||
            (saved.candidate.flags & MR_CCD_PAIR_SPECULATIVE_FALLBACK) != 0u) {'''
new='''        const bool matchingKind = eventFailure
            ? saved.validAndStamp.x == MR_CCD_EVENT_FAILURE_SNAPSHOT_TAG &&
                (saved.candidate.flags & MR_CCD_PAIR_HAS_IMPACT) != 0u &&
                saved.candidate.status == MR_STEP_SUCCESS
            : saved.validAndStamp.x == MR_CCD_FAILURE_SNAPSHOT_TAG &&
                (saved.candidate.flags & MR_CCD_PAIR_UNRESOLVED) != 0u &&
                (saved.candidate.flags & MR_CCD_PAIR_SPECULATIVE_FALLBACK) == 0u;
        if (!matchingKind || saved.candidate.environment != status.environment ||
            saved.candidate.compiledPair != failedPair ||
            world.failingIndex != status.firstFailingPair) {'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('''const std::uint32_t header[] = {2u, world.environment, world.controlStep,
            world.failingSubstep, status.firstFailingPair,''','''const std::uint32_t header[] = {eventFailure ? 3u : 2u, world.environment, world.controlStep,
            world.failingSubstep, failedPair,''');p.write_text(s)
p=R/'apps/ccd_sweep_probe.mm';s=p.read_text()
s=s.replace('(h[0]==2&&h[5]==11)','((h[0]==2||h[0]==3)&&h[5]==11)')
s=s.replace('h[4]==s.firstFailingPair&&','h[4]==(h[0]==3?s.firstFailingEventKeyLow:s.firstFailingPair)&&')
s=s.replace('||h[0]==2),"Failure header differs"','||h[0]>=2),"Failure header differs"')
s=s.replace('if(h[0]==2){','if(h[0]>=2){')
s=s.replace('world[0].failingIndex==h[4]','world[0].failingIndex==s.firstFailingPair')
s=s.replace('p.validAndStamp.x==MR_CCD_FAILURE_SNAPSHOT_TAG&&','p.validAndStamp.x==(h[0]==3?MR_CCD_EVENT_FAILURE_SNAPSHOT_TAG:MR_CCD_FAILURE_SNAPSHOT_TAG)&&')
old='''        require((p.candidate.flags&MR_CCD_PAIR_UNRESOLVED)!=0&&(p.candidate.flags&MR_CCD_PAIR_SPECULATIVE_FALLBACK)==0,"Protected record is not unsafe unresolved CCD");'''
new='''        if(h[0]==3) {
            require(s.firstFailingPair==MR_INVALID_INDEX&&s.firstFailingEventKeyHigh==h[1]&&
                (s.code==MR_STEP_DID_NOT_CONVERGE||s.code==MR_STEP_CCD_EVENT_BUDGET_EXHAUSTED)&&
                (p.candidate.flags&MR_CCD_PAIR_HAS_IMPACT)!=0&&p.candidate.status==MR_STEP_SUCCESS,
                "Protected record is not the reported event-budget failure");
        } else {
            require((p.candidate.flags&MR_CCD_PAIR_UNRESOLVED)!=0&&(p.candidate.flags&MR_CCD_PAIR_SPECULATIVE_FALLBACK)==0,"Protected record is not unsafe unresolved CCD");
        }'''
assert s.count(old)==1;s=s.replace(old,new).replace('(h[0]==2?"protected-first-failure":"post-submission")','(h[0]==3?"protected-selected-event":h[0]==2?"protected-first-failure":"post-submission")');p.write_text(s)
shutil.copy2(__file__,A/'prepare.py');a.write_json(A/'preparation.json',{'base':a.git('rev-parse','HEAD').decode().strip(),'restored_sources':sources,
 'baseline_runtime':{str(p.relative_to(R)):a.file_hash(p) for p in T.rglob('*') if p.is_file()},
 'purpose':'Extend bounded optional failure inspection to the selected zero-time/advance-budget event; no physics, capacity or tolerance changes.'})
print(json.dumps({'raw_root':str(B),'source_diff_check':subprocess.run(['git','diff','--check']).returncode}))
