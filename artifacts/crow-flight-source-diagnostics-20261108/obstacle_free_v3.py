from pathlib import Path
import hashlib,json,os,re,subprocess,time
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-flight-source-diagnostics-20261108';B=R/'build/crow-flight-source-diagnostics-01'
lib=B/'obstacle-free-lib/libmetalrobo.dylib'
runs=json.load(open(A/'runs.json'));old=B.parent/'crow-navigation-feedback-qualification-01/versioned.policypack';new=B/'obstacle-free-policy.policypack'
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(lib.parent)}
probe=runs[1]['argv'][:]
probe[probe.index('--metallib')+1]=str(B/'filtered-action.metallib')
probe[probe.index('--steps')+1]='1'
p=subprocess.run(probe,cwd=R,env=env,text=True,capture_output=True,timeout=120)
(A/'v3-policy-rejection.stderr').write_text(p.stderr);(A/'v3-policy-rejection.stdout').write_text(p.stdout)
m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',p.stderr)
if p.returncode==0 or m is None:raise RuntimeError('Expected contract rejection; got '+p.stderr[-2500:])
world,task,obs,act=map(int,m.groups())
src=read_policy_pack(old,library_path=str(lib))
# Fingerprints include the world identity and all change with collision flags.
# Tensor identity and matched pre-contact trajectory are checked below.
def layers(xs):return [PolicyDenseLayerArtifact(weights=x.weights,bias=x.bias,activation=x.activation) for x in xs]
write_policy_pack(new,policy_id='crow_v15_course_collision_disabled_diagnostic',revision=1,contract_version=src.contract_version,
 world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,
 layers=layers(src.layers),critic_layers=layers(src.critic_layers),observation_mean=src.observation_mean,
 observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,critic_observation_mean=src.critic_observation_mean,
 critic_observation_inverse_standard_deviation=src.critic_observation_inverse_standard_deviation,
 action_log_standard_deviation=src.action_log_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,
 observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(lib))
results=[]
for source_idx in [1,3]:
 base=runs[source_idx];cmd=base['argv'][:];name=base['name']
 cmd[cmd.index('--metallib')+1]=str(B/'filtered-action.metallib')
 cmd[cmd.index('--policy-pack')+1]=str(new)
 target=B/('obstacle-free-v3-'+name+'.tsv');cmd[cmd.index('--state-trace')+1]=str(target)
 t=time.monotonic();p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=300)
 (A/('obstacle-free-v3-'+name+'.stdout')).write_text(p.stdout)
 (A/('obstacle-free-v3-'+name+'.stderr')).write_text(p.stderr)
 if p.returncode:raise RuntimeError(p.stderr[-4000:])
 data=json.loads(p.stdout);events=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_CONTACT_TRACE ')]
 a=(B/(name+'.tsv')).read_text().splitlines()[1:];b=target.read_text().splitlines()[1:]
 same=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
 unexpected=[(x['step'],x['lane'],c) for x in events for c in x['contacts'] if c['a'] in range(18,23) or c['b'] in range(18,23)]
 result={'name':name,'argv':cmd,'seconds':time.monotonic()-t,'result':data['first_episode_navigation'],
 'failed_environment_steps':data['failed_environment_steps'],'fatal_contacts':events,'obstacle_contact_violations':unexpected,
 'exact_trace_prefix_steps':same,'diagnostic_library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),
 'policy_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'old_contract_rejected':True,
 'scope':'Scene course shapes have simulation disabled; targets and visual meshes remain; ground collision unchanged. Diagnostic policy has identical tensors, explicitly rebound to changed task/world contract.'}
 results.append(result);(A/'obstacle-free-v3-results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(name,'same prefix',same,'failures',data['first_episode_navigation']['physical_failure_count'],
       'completion',data['first_episode_navigation']['safe_completion_count'],'waypoints',data['first_episode_navigation']['maximum_waypoints_by_environment'],
       'obstacle violations',len(unexpected),flush=True)
 if unexpected:raise RuntimeError('Disabled obstacles still appear as fatal collision candidates')
