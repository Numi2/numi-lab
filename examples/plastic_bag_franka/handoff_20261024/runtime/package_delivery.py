"""Preserve only a completed handoff as the primary result; keep failures labelled."""
from pathlib import Path
import json,shutil,hashlib,subprocess,tarfile
ROOT=Path('/Users/home/Documents/Codex/2026-10-21/i-wa');W=ROOT/'work/plastic_bag_franka_handoff_20261024';O=ROOT/'outputs'/W.name;P=ROOT/'work/plastic_bag_handoff_preservation_20261024';D=P/'handoff_20261024';REPO=ROOT/'work/numi_lab_bag_publication_20261024';PARENT=ROOT/'outputs/plastic_bag_empty_franka_20261024'
def copy(a,b):
 b.parent.mkdir(parents=True,exist_ok=True)
 if a.is_dir():shutil.copytree(a,b,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 else:shutil.copy2(a,b)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while data:=f.read(1024*1024):h.update(data)
 return h.hexdigest()
v=json.loads((O/'video_record.json').read_text());a=json.loads((O/'assessment.json').read_text());assert v['decoded_without_errors'] and v['native_states']==1921 and a['last_grip_nodes'][0]==0 and a['last_grip_nodes'][1]>0
assert json.loads((O/'robot_film_full.json').read_text())['intersection_state_count']==0
assert json.loads((O/'interrobot_full.json').read_text())['intersection_state_count']==0
copy(W,D/'solver');copy(O,D/'evidence');copy(ROOT/'work/plastic_bag_franka/franka_assets',D/'franka_assets')
# Bind the exact approved prefix and its interval energy/solver measurements.
copy(PARENT/'trajectory.json',D/'parent_approved/trajectory.json')
for directory in ['prefix','combined']:
 for file in ['solver_accuracy.csv','energy_balance.csv','energy_history.csv','run_summary.json']:
  if (PARENT/directory/file).exists():copy(PARENT/directory/file,D/'parent_approved'/directory/file)
copy(ROOT/'work/plastic_bag_empty_franka_20261024/source',D/'parent_approved/source')
files=sorted(x for x in D.rglob('*') if x.is_file() and x.name!='manifest_sha256.json');manifest={str(x.relative_to(D)):sha(x) for x in files};(D/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2))
archive=P/'numi_plastic_bag_franka_handoff_20261024.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as t:t.add(D,arcname=D.name)
video=O/'empty_bag_two_franka_handoff_8s_normal.mp4'
(P/'SHA256SUMS').write_text(sha(archive)+'  '+archive.name+'\n'+sha(video)+'  '+video.name+'\n')
E=REPO/'examples/plastic_bag_franka/handoff_20261024'
for n in ['source','include','build_only.sh','build_fk.sh','grip.csv','grip2.csv','robot.json','robot2.json','initial_state.json','scene_plan.json','donor_base.txt','receiver_base.txt','donor_upright_targets.csv','receiver_upright_targets.csv']:copy(W/n,E/n)
for n in ['render.py','render_dual.py','render_studio.py','robot_clearance.py','robot_film_overlap.py','interrobot_overlap.py','grip_alignment_audit.py','grip_alignment_single.py','handoff_metrics.py','plan_upright_handoff.py','watch_capture.py','continue_handoff.py','watch_finish.py','contact_watch_only.py','finalize_video.py','package_delivery.py','verify_release.py','bind_handoff_fixture.py']:copy(W/'runtime'/n,E/'runtime'/n)
copy(W/'HANDOFF_SCENE.md',E/'README.md');copy(W/'EVIDENCE_INDEX.md',E/'EVIDENCE_INDEX.md')
for n in ['assessment.json','handling_plan.json','video_record.json','handoff_metrics.json','grip_alignment.json','robot_film_full.json','interrobot_full.json','solver_preservation.json','scene_geometry_binding.json','two_hand_support_before_release.json','direct_launch_identity_sha256.txt','capture_launch_identity_sha256.txt','donor_upright_ik.log','receiver_upright_ik.log','donor_table_clearance.json','receiver_table_clearance.json','native_fk_source_identity.json','film_kernel_identity.json','gripper_rotation.patch','direct_runtime_binding.json','direct_machine_identity.txt','checkpoint_join_equivalence.json','audit_reuse_receipt.json','completion_protocol.json']:copy(O/n,E/'evidence'/n)
copy(video,REPO/'docs/media/plastic-bag-franka-handoff-8s.mp4')
gif=REPO/'docs/media/plastic-bag-franka-handoff.gif';subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-v','error','-i',str(video),'-filter_complex','fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a','-loop','0',str(gif)],check=True)
p=REPO/'README.md';s=p.read_text();assert '### Two Franka handoff' not in s
s+='''

### Two Franka handoff — empty bag

[![Empty pouch passed between two Franka grippers](docs/media/plastic-bag-franka-handoff.gif)](https://github.com/Numi2/numi-lab/blob/numisolver/docs/media/plastic-bag-franka-handoff-8s.mp4)

**[Watch the complete normal-speed handoff](docs/media/plastic-bag-franka-handoff-8s.mp4)**

The first arm picks up the empty pouch by its opening corner. The second grips the opposite opening corner, the first releases and retreats, then the second holds the bag. Eight simulated seconds, 1,921 native Metal states at 240 Hz, zero coins and zero fixed film nodes. Both finite fingertip contacts contribute friction, spring energy and actuator work; rotating-pad material velocity is included. Blender only presents the saved states.

'''
s+=f"Measured second-hand contact begins at **{a['receiver_first_contact_s']:.3f} s**; the last first-hand contact is at **{a['donor_last_contact_s']:.3f} s**. The receiving hand supports the bag alone for **{a['receiver_only_supported_duration_s']:.3f} s**. Every saved state passes the authored solid-housing/film and inter-arm surface-crossing checks.\n\n"
s+='''The receiver uses a gentle pinch of approximately **0.4 N per jaw**. The receiver has an explicitly authored **12 mm forward pad extension**; native FK and visible contact faces match. Articulated poses were rebound to this fixture with the exact world pad input bytes retained; film forces, work and motion are unchanged. Its central rigid carrier has a leading tip recessed by 10 mm (4 × 8 × 4 mm), also included in the visible geometry and crossing checks. Robot joints are prescribed, and saved-state clearance is not continuous whole-robot collision dynamics. The approved film material and energy tolerances are preserved; the finite pad edge-entry model remains a documented limitation, and gentle preload is a scene choice rather than a repair of that discontinuity; its material constants remain uncalibrated and air drag is absent. The first five seconds retain the approved older gripper-model prefix; corrected rotating-pad dynamics continue its exact state thereafter.

[Source, measurements and limits](examples/plastic_bag_franka/handoff_20261024) · [Full native replay, checkpoints and preserved rejected attempts](https://github.com/Numi2/numi-lab/releases/tag/plastic-bag-franka-handoff-2026-10-24)
''';p.write_text(s)
r={'archive':str(archive),'archive_sha256':sha(archive),'manifest_files':len(manifest),'video':str(video),'video_sha256':sha(video),'example':str(E),'gif':str(gif)};(P/'package_record.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
