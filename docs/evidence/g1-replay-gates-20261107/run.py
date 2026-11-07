"""Preregistered real-pack extraction and unmodified replay; no perturbation."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
from gatecheck import compare

root = Path('/Users/numi/numi-heldout-g1-20261107/replay-gates-4081010')
owner = Path('/Users/numi/numi-g1-policy-origin-20261107')
build = Path('/Users/numi/numi-g1-policy-origin-build-20261107')
policy = owner/'artifacts/g1-legs-locomotion-60min-20260805/retained-policy/candidate.policypack'
probe = Path('/Users/numi/numi-heldout-g1-20261107/contract-fingerprint')
extractor = Path('/Users/numi/numi-heldout-g1-20261107/extract-actions')
files = {'policy': policy, 'rollout': build/'bin/metalrobo_task_rollout',
         'native': build/'lib/libmetalrobo.dylib', 'metal': build/'shaders/MetalRobo.metallib',
         'probe': probe, 'extractor': extractor}
expected = {'policy': '23930cbf1af7f40b3a68a9456e865ef88986426f7847e6a42669e2582ff3b3d3',
 'rollout': '571a7c461b698dfa0e2d4df48ef6e8a28515d4432c18fcea677b1a3fa5f9921f',
 'native': '80663b5c54b80798c7cfcfeb67dfb344de6992e2ecc9a48db531ecdaedd2209a',
 'metal': '32d9ed14031ab83f87dfcf8281b1d3c65c6315a02e04ce15ab2728772ef35fab'}
root.mkdir(parents=False, exist_ok=False)
def save(name, data): (root/name).write_text(data)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(name, args, seconds):
    save(name+'.command.json', json.dumps([str(v) for v in args], indent=2)+'\n')
    with (root/(name+'.stdout')).open('wb') as out, (root/(name+'.stderr')).open('wb') as err:
        proc = subprocess.Popen([str(v) for v in args], stdout=out, stderr=err, start_new_session=True)
        try:
            rc=proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            rc=124
    save(name+'.exit', str(rc)+'\n')
    print(name, 'exit', rc, flush=True)
    if rc: raise RuntimeError(name+' failed or timed out')

def gate(name, first, second, full_trace):
    report=compare(root,first,second,full_trace)
    save(name+".json",json.dumps(report, indent=2)+"\n")
    if report["differences"] or report["trace_mismatch_count"]: raise RuntimeError(name+": mismatch")

try:
    identities={name:sha(p) for name,p in files.items()}
    save('identities.json',json.dumps(identities,indent=2)+'\n')
    if any(identities[k] != v for k,v in expected.items()): raise RuntimeError('historical binary/input identity mismatch')
    revision=subprocess.check_output(['git','-C',str(owner),'rev-parse','HEAD'],text=True).strip()
    save('owner.revision',revision+'\n')
    save('owner.status',subprocess.check_output(['git','-C',str(owner),'status','--short'],text=True))
    with (root/'owner.patch').open('wb') as f: subprocess.run(['git','-C',str(owner),'diff','--binary'],stdout=f,check=True)
    episode_hash=sha(owner/'apps/EpisodeForwardEvidence.swift')
    save('owner.EpisodeForwardEvidence.swift.sha256',episode_hash+'\n')
    if sha(root/'owner.patch') != 'e1f95de19103c9324b9aacc503b70b251ebb2ff30640fbb0d961706a9f6f79da' or episode_hash != '330c8b43c2f4b9c4c9bff4ab0a1b3e2414fd934dd802ad3c72b0c3460643b000': raise RuntimeError('historical source patch mismatch')
    if revision != '791fb586934f8222fcf8269c411c17eb2a774087': raise RuntimeError('historical revision mismatch')
    run('fingerprint',[probe,policy],30)
    fingerprint=(root/'fingerprint.stdout').read_text()
    if 'policy_program_fingerprint=17940125103817627925\n' not in fingerprint or 'PASS exact policy contract\n' not in fingerprint: raise RuntimeError('compiled policy identity mismatch')
    common=[files['rollout'],'--envs','8','--steps','1000','--repeats','1','--chunk','1','--physics-substeps','4','--velocity-iterations','4','--final-velocity-iterations','2','--scene','ground','--task','g1-legs-locomotion','--metallib',files['metal'],'--no-scheduled-resets','--seed','4081010','--state-trace-environment','0']
    run('closed',common+['--state-trace',root/'closed.trace.tsv','--policy-pack',policy],120)
    run('recorded',common+['--state-trace',root/'recorded.trace.tsv','--policy-pack',policy,'--rollout-pack',root/'recorded.rolloutpack'],120)
    gate('recording-parity','closed','recorded',False)
    run('extract',[extractor,policy,root/'recorded.rolloutpack','17940125103817627925',root/'actions.f32'],30)
    save('captured.sha256',json.dumps({'rolloutpack':sha(root/'recorded.rolloutpack'),'actions':sha(root/'actions.f32')},indent=2)+'\n')
    run('replay',common+['--state-trace',root/'replay.trace.tsv','--action-stream',root/'actions.f32'],120)
    gate('replay-parity','recorded','replay',True)
    save('result.txt','PASS: extraction and exact unmodified replay gates\n')
    print('PASS extraction and replay gates',flush=True)
except Exception as e:
    save('result.txt','STOP: '+str(e)+'\n')
    print('STOP:',e,flush=True)
    raise SystemExit(1)
