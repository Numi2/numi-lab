from pathlib import Path
import subprocess,time,json,hashlib
ROOT=Path('/Users/home/Documents/Codex/2026-10-21/i-wa');W=ROOT/'work/plastic_bag_empty_franka_20261024';OUT=ROOT/'outputs/plastic_bag_empty_franka_20261024';REMOTE='/Users/n/numi-plastic-bag-empty-franka-20261024'
def run(a,**kw):return subprocess.run(a,check=True,**kw)
while True:
 x=subprocess.run(['ssh','macmini',f'test -f {REMOTE}/first_drop/run_summary.json'],capture_output=True)
 if x.returncode==0:break
 progress=run(['ssh','macmini',f'tail -1 {REMOTE}/first_drop/solver_accuracy.csv'],capture_output=True,text=True).stdout.strip();print('FIRST DROP '+progress,flush=True);time.sleep(45)
run(['rsync','-a',f'macmini:{REMOTE}/first_drop/',str(OUT/'first_drop')+'/']);run(['rsync','-a',f'macmini:{REMOTE}/first_drop.log',str(OUT/'first_drop.log')])
f=json.loads((OUT/'first_drop/trajectory.json').read_text())['frames'][-1];assert f['step']==240 and not f['coins'];corner=f['front'][-1]
old=json.loads((W/'robot.json').read_text());park=old['frames'][240];hover=park['grip_center'];seed=old['frames'][768]['q']
request=corner+park['q']+hover+seed;(W/'pickup_request.txt').write_text(' '.join(map(repr,request))+'\n')
(OUT/'grasp_target.json').write_text(json.dumps({'step':240,'node':'front[-1], welded opening corner','actual_position_m':corner,'hover_center_m':hover,'requested_lift_m':.5,'closing_axis':'approved tilted closing axis; actual native FK pads'},indent=2))
run(['rsync','-a',str(W/'pickup_request.txt'),f'macmini:{REMOTE}/pickup_request.txt'])
x=run(['ssh','macmini',f'cd {REMOTE} && ./planner new_grip.csv new_robot.json pickup_request.txt'],capture_output=True,text=True);print(x.stdout,flush=True);(OUT/'planner.log').write_text(x.stdout)
for name in ['new_grip.csv','new_robot.json']:run(['rsync','-a',f'macmini:{REMOTE}/{name}',str(W/name)])
new=json.loads((W/'new_robot.json').read_text());assert len(new['frames'])==1681
# Keep every executed pose, including the exact join state; only future inputs change.
robot={'frames':old['frames'][:241]+new['frames'][1:]};assert len(robot['frames'])==1921
(W/'robot.json').write_text(json.dumps(robot,separators=(',',':')))
oldrows=(W/'grip.csv').read_text().splitlines();newrows=(W/'new_grip.csv').read_text().splitlines();(W/'grip.csv').write_text('\n'.join(oldrows[:242]+newrows[2:])+'\n')
x=run(['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(W/'runtime/robot_clearance.py'),str(W/'robot.json')],capture_output=True,text=True);(OUT/'robot_table_clearance.json').write_text(x.stdout);clear=json.loads(x.stdout);assert clear['minimum_gap_m']>=-1e-7,clear
# Geometry-only planning check, with the actual landed film frozen. This is not a simulated trajectory or video.
geometry=[]
for step in range(240,625):
 x=dict(f);x['step']=step;x['time_s']=step/240.;x['robot']=robot['frames'][step];geometry.append(x)
gp=OUT/'static_landed_approach_geometry.json';gp.write_text(json.dumps({'frames':geometry},separators=(',',':')))
x=run(['/Applications/Blender.app/Contents/MacOS/Blender','-b','--python-exit-code','1','--python',str(W/'runtime/robot_film_overlap.py'),'--',str(gp)],capture_output=True,text=True);(OUT/'static_landed_approach_geometry.log').write_text(x.stdout)
check=json.loads(next(l.split('ROBOT_FILM_AUDIT ',1)[1] for l in x.stdout.splitlines() if l.startswith('ROBOT_FILM_AUDIT ')));assert check['intersection_state_count']==0,check
for name in ['grip.csv','robot.json']:run(['rsync','-a',str(W/name),f'macmini:{REMOTE}/{name}'])
print('PLANNED '+json.dumps({'corner':corner,'robot_table_min_gap_m':clear['minimum_gap_m']}),flush=True)
# This PID is solely owned by the empty-bag task.
run(['ssh','macmini',f'cd {REMOTE} && nohup ./build/numi_classic_bag_drop --empty-bag --output combined --resume first_drop/latest_checkpoint.bin --steps 1920 --checkpoint-every 120 > combined.log 2>&1 < /dev/null &'])
# Capture the PID by unique cwd/output invocation; never act on another simulation.
time.sleep(1)
x=run(['ssh','macmini',f'pgrep -f "^./build/numi_classic_bag_drop --empty-bag --output combined --resume first_drop/latest_checkpoint.bin --steps 1920 --checkpoint-every 120$"'],capture_output=True,text=True);pids=x.stdout.split();assert len(pids)==1,pids
run(['ssh','macmini',f'printf %s {pids[0]} > {REMOTE}/native.pid']);print('NATIVE LAUNCHED PID '+pids[0],flush=True)
