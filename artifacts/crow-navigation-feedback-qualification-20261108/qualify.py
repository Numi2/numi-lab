from pathlib import Path
import hashlib,json,os,re,subprocess
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack,read_policy_rollout_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();B=R/'build/crow-navigation-feedback-qualification-01';A=R/'artifacts/crow-navigation-feedback-qualification-20261108'
lib=R/'build/lib/libmetalrobo.dylib'
srcfile=R/'build/crow-active-navigation-transfer-01/trainable.policypack'; dstfile=B/'versioned.policypack'
cmd=[str(R/'build/bin/metalrobo_task_rollout'),'--birdflow-american-crow-journey','--birdflow-journey-variant','v15',
 '--birdflow-navigation-course','training','--envs','1','--steps','180','--repeats','1','--chunk','1','--physics-substeps','4',
 '--seed','20261111','--no-scheduled-resets','--minimum-difficulty-band','10','--maximum-difficulty-band','10',
 '--policy-pack',str(srcfile),'--visual-observation-config','assets/crow_navigation_course/crow-navigation.active-head.visual-observation.json']
def run(name,cmd):
 p=subprocess.run(cmd,cwd=R,text=True,capture_output=True,timeout=240,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 (B/(name+'.stdout')).write_text(p.stdout);(B/(name+'.stderr')).write_text(p.stderr)
 return p
p=run('old-policy-rejection',cmd)
m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',p.stderr)
assert p.returncode and 'incompatible_contract' in p.stderr and m,p.stderr
world,task,observation,action=map(int,m.groups())
src=read_policy_pack(srcfile,library_path=lib)
assert src.world_fingerprint==world and src.observation_fingerprint==observation and src.action_fingerprint==action
def convert(layers):return [PolicyDenseLayerArtifact(weights=l.weights,bias=l.bias,activation=l.activation) for l in layers]
write_policy_pack(dstfile,policy_id='crow_v15_feedback_qualification',revision=1,contract_version=src.contract_version,
 world_fingerprint=world,task_fingerprint=task,observation_fingerprint=observation,action_fingerprint=action,
 layers=convert(src.layers),critic_layers=convert(src.critic_layers),
 observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,
 critic_observation_mean=src.critic_observation_mean,critic_observation_inverse_standard_deviation=src.critic_observation_inverse_standard_deviation,
 action_log_standard_deviation=src.action_log_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,
 observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=lib)
cmd[cmd.index('--policy-pack')+1]=str(dstfile)
out=[]
for name,teach in [('new-autonomous',False),('new-teacher-labels',True)]:
 c=cmd+['--state-trace',str(B/(name+'.tsv'))]
 if teach:c+=['--birdflow-journey-teacher','--birdflow-journey-student-authority','1','--rollout-pack',str(B/(name+'.rolloutpack'))]
 res=run(name,c);out.append({'name':name,'argv':c,'returncode':res.returncode,'stderr':res.stderr[-2000:]})
 assert res.returncode==0,res.stderr
old=(B/'old-autonomous.tsv').read_bytes();new=(B/'new-autonomous.tsv').read_bytes()
corrected=(B/'fixed-teacher-labels.tsv').read_bytes();teacher=(B/'new-teacher-labels.tsv').read_bytes()
a=read_policy_rollout_pack(B/'old-teacher-labels.rolloutpack',library_path=lib)
b=read_policy_rollout_pack(B/'fixed-teacher-labels.rolloutpack',library_path=lib)
c=read_policy_rollout_pack(B/'new-teacher-labels.rolloutpack',library_path=lib)
oldlabels=a.teacher_actions.reshape(-1,18);fixedlabels=b.teacher_actions.reshape(-1,18);newlabels=c.teacher_actions.reshape(-1,18)
delta=np.max(np.abs(oldlabels-fixedlabels),axis=0)
results={'old_policy_rejected':True,'new_contract':{'world':world,'task':task,'observation':observation,'action':action},
 'physical_autonomous_trace_byte_identical':old==new,'teacher_authority_one_trace_byte_identical':corrected==teacher,
 'teacher_labels_match_fixed_diagnostic':bool(np.array_equal(fixedlabels,newlabels)),
 'old_vs_fixed_max_label_differences':delta.tolist(),'head_teacher_labels_preserved':bool(np.array_equal(oldlabels[:,15:],fixedlabels[:,15:])),
 'teacher_label_changed':bool(np.any(delta[:15]>1e-6)),
 'policy_sha256':hashlib.sha256(dstfile.read_bytes()).hexdigest(),'runs':out}
(A/'result.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
print(json.dumps(results,sort_keys=True))
assert all(results[k] for k in ('physical_autonomous_trace_byte_identical','teacher_authority_one_trace_byte_identical','teacher_labels_match_fixed_diagnostic','head_teacher_labels_preserved','teacher_label_changed'))
