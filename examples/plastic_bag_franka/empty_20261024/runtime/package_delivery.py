"""Package only a completed, audited empty-bag run, preserving rejected evidence separately."""
from pathlib import Path
import json,shutil,hashlib,subprocess,tarfile
ROOT=Path('/Users/home/Documents/Codex/2026-10-21/i-wa');W=ROOT/'work/plastic_bag_empty_franka_20261024';O=ROOT/'outputs/plastic_bag_empty_franka_20261024';P=ROOT/'work/plastic_bag_empty_preservation_20261024';D=P/'empty_20261024';REPO=ROOT/'work/numi_lab_bag_publication_20261024';APPROVED=ROOT/'work/plastic_bag_preservation_20261024/approved_20261024'
def copy(a,b):
 b.parent.mkdir(parents=True,exist_ok=True)
 if a.is_dir():shutil.copytree(a,b,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 else:shutil.copy2(a,b)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((O/'delivery_receipt.json').read_text());assert a['coin_count_from_gpu']==0 and a['native_states']==1921 and a['actual_corner_lift_m']>.45
p=json.loads((O/'performance_followup.json').read_text());assert p['all_saved_state_and_checkpoint_bytes_equal']
assert json.loads((O/'side_video_record.json').read_text())['decode_pass']
for name in ['source','include','runtime','native_runtime','build_only.sh','grip.csv','robot.json','initial_state.json','pickup_request.txt','scene_plan.json','EMPTY_SCENE.md']:copy(W/name,D/'solver'/name)
for name in ['franka_assets','owner_fk']:copy(APPROVED/name,D/name)
for name in ['first_drop','prefix','combined','performance','rejected_topdown','rejected_fixed_tilt']:copy(O/name,D/'evidence'/name)
for path in O.iterdir():
 if path.is_file() and path.suffix in ['.json','.csv','.txt','.log','.patch','.mp4'] and path.name not in ['static_landed_approach_geometry.json','first_drop_progress_raw.json','package_record.json']:copy(path,D/'evidence'/path.name)
files=sorted(x for x in D.rglob('*') if x.is_file() and x.name!='manifest_sha256.json');manifest={str(x.relative_to(D)):sha(x) for x in files};(D/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2))
archive=P/'numi_plastic_bag_empty_franka_20261024.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:tar.add(D,arcname=D.name)
(P/'SHA256SUMS').write_text(sha(archive)+'  '+archive.name+'\n')
# Small reviewable repository payload. Full native states and rejected runs remain in the release archive.
E=REPO/'examples/plastic_bag_franka/empty_20261024'
for name in ['source','include','runtime','build_only.sh','grip.csv','robot.json','initial_state.json','pickup_request.txt','scene_plan.json']:copy(W/name,E/name)
copy(W/'EMPTY_SCENE.md',E/'README.md')
for name in ['delivery_receipt.json','aggregate_metrics.json','assessment.json','video_record.json','side_video_record.json','performance_comparison.json','performance_followup.json','performance_followup_plan.json','robot_table_clearance.json','planner.log','launch_identity_sha256.txt','complete_source_identity_sha256.txt','empty_host.patch','empty_metal.patch']:copy(O/name,E/'evidence'/name)
for src,dst in [('empty_bag_air_drop_high_pickup_drop_8s_normal.mp4','plastic-bag-franka-empty-8s.mp4'),('empty_bag_side_8s_normal.mp4','plastic-bag-franka-empty-side-8s.mp4')]:copy(O/src,REPO/'docs/media'/dst)
gif=REPO/'docs/media/plastic-bag-franka-empty.gif';subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-v','error','-i',str(O/'empty_bag_air_drop_high_pickup_drop_8s_normal.mp4'),'-filter_complex','fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a','-loop','0',str(gif)],check=True)
readme=REPO/'README.md';text=readme.read_text();heading='### Empty bag comparison'
assert heading not in text
text+='''

### Empty bag comparison

[![Empty pouch: air drop, high corner pickup and release](docs/media/plastic-bag-franka-empty.gif)](https://github.com/Numi2/numi-lab/blob/numisolver/docs/media/plastic-bag-franka-empty-8s.mp4)

**[Watch the complete normal-speed empty-bag run](docs/media/plastic-bag-franka-empty-8s.mp4)** · [Side view](docs/media/plastic-bag-franka-empty-side-8s.mp4)

The same film material and contact solver, with zero active coins: air drop, finite fingertip corner pickup, a 0.50 m commanded lift, then release and table impact. Its folds and hanging shape emerge from the unloaded film dynamics. The eight-second run records 1,921 native Metal states at 240 Hz. Blender only renders the saved states.

'''
text+=f"Measured film-corner lift: **{a['actual_corner_lift_m']:.3f} m**. Omitting empty coin-only dispatches reduced median GPU time by **{p['median_gpu_reduction_percent']:.2f}%** across six paired eight-interval measurements; saved states and final checkpoint bytes matched in every pair. This timing is scoped to those intervals. Film, self-contact, table and gripper passes and solver tolerances were retained.\n\n"
text+='''[Source, exact identities, measurements and limits](examples/plastic_bag_franka/empty_20261024) · [Complete native replay and preserved rejected paths](https://github.com/Numi2/numi-lab/releases/tag/plastic-bag-empty-franka-2026-10-24)

Air drag remains absent for this material comparison. Robot joints are prescribed; saved-state housing clearance is not continuous whole-robot contact dynamics. Stable replay is not experimental material validation.
''';readme.write_text(text)
print(json.dumps({'archive':str(archive),'archive_sha256':sha(archive),'manifest_files':len(manifest),'example':str(E),'gif':str(gif)},indent=2))
