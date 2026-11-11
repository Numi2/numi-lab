from metalrobo.mlx_policy_learning import read_policy_rollout_pack
import numpy as np,json
from pathlib import Path
A=Path("artifacts/crow-v15-dagger-isolated-labels-control-20261111")
r=read_policy_rollout_pack(Path("build/crow-v15-dagger-isolated-labels-control-20261111/label-20261126-rollout.bin"),library_path="build/crow-v15-hip-roll-journey-integration-20261111/libmetalrobo.dylib")
labels=r.teacher_actions.reshape(r.control_step_count,r.environment_count,r.action_count)
latents=r.latents.reshape(r.control_step_count,r.environment_count,r.action_count)
x=labels[400:1400,:,:4];y=latents[400:1400,:,:4]
o={"steps":r.control_step_count,"environments":r.environment_count,"action_count":r.action_count,"finite":bool(np.isfinite(x).all()),"minimum":float(x.min()),"maximum":float(x.max()),"nonzero_fraction":float(np.mean(np.abs(x)>1e-6)),"different_from_student_latent_fraction":float(np.mean(np.abs(x-y)>1e-6)),"scope":"Diagnostic teacher labels after eight seconds, on student-visited states; a nonidentity label does not establish an expert correction, safe demonstration, or valid post-failure samples."}
(A/"label-inspection.json").write_text(json.dumps(o,indent=2)+"\n")
print(json.dumps(o,indent=2))
