import gzip,json
from pathlib import Path
p=json.load(gzip.open('artifacts/crow-v15-landing-descent-dev-20261111/trial/replay.json.gz','rt'))['payload'];names=p['body_names'];expected=['locomotion_ground','crow_course_gate_left','crow_course_gate_right','crow_course_slalom_a','crow_course_slalom_b','crow_course_perch'];assert names[33:39]==expected
scene=names[33:];course=scene.index('crow_course_gate_left');perch=33+course+4
assert perch==38 and names[perch]=='crow_course_perch'
Path('artifacts/crow-v15-perch-toe-world-index-dev-20261111/scene-mapping-check.json').write_text(json.dumps({'source':'retained collision-enabled development replay','robot_end':33,'scene_local_course_start':course,'contact_perch_body_index':perch,'body_names':expected},indent=2)+'\n')
