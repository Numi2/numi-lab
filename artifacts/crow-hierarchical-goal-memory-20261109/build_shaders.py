from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-hierarchical-goal-memory-20261109';B=R/'build/crow-hierarchical-goal-memory-20261109'
hashes={}
for mode in ['actor','oracle','dagger']:
 base=(R/'build/crow-hierarchical-contract-20261109'/(mode+'.metal')).read_text()
 old='''        } else if (routeComponent < 13u) {
            value = routeComponent - 8u == waypoint ? 1.0f : 0.0f;'''
 new='''        } else if (routeComponent >= 8u && routeComponent <= 10u &&
                   (program.schedule.w & (1u << 28u)) != 0u) {
            // Memory of the previously issued goal, not obstacle geometry.
            const uint previous = waypoint > 0u ? min(waypoint-1u,kCrowNavigationWaypointCount-1u) : 0u;
            const float3 relative = rotateInverse(
                orientation,
                crowNavigationWaypointTarget(sceneBodies,operation.source.y,previous) -
                    rootWorldPosition(program,q));
            value = waypoint > 0u ? 0.25f*relative[routeComponent-8u] : 0.0f;
        } else if (routeComponent == 11u &&
                   (program.schedule.w & (1u << 28u)) != 0u) {
            value = 0.5f*state.threatTeacher.w;
        } else if (routeComponent < 13u) {
            value = routeComponent - 8u == waypoint ? 1.0f : 0.0f;'''
 assert base.count(old)==1
 s=base.replace(old,new)
 old2='''            phase=max(phase,1.0f);
            uint course=MR_INVALID_INDEX;'''
 assert s.count(old2)==1
 s=s.replace(old2,'''            uint course=MR_INVALID_INDEX;''')
 old3='''                const float3 finalGoal=crowNavigationWaypointTarget(sceneState+sceneBase,course,4u);
                uint loadedFeet=0u;'''
 new3='''                const float3 finalGoal=crowNavigationWaypointTarget(sceneState+sceneBase,course,4u);
                if (phase < 0.5f && length(finalGoal.xy-pos.xy)<0.08f &&
                    length(yawFrameLinear.xy)<0.08f) phase=1.0f;
                uint loadedFeet=0u;'''
 assert s.count(old3)==1
 s=s.replace(old3,new3)
 src=B/(mode+'.metal');src.write_text(s)
 (A/(mode+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/hierarchical-'+mode+'.metal',tofile='b/goal-memory-'+mode+'.metal')))
 air=B/(mode+'.air');shader=B/(mode+'.metallib')
 subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
 other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
 subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),'-o',str(shader)],check=True)
 hashes[mode]={'source':hashlib.sha256(src.read_bytes()).hexdigest(),'shader':hashlib.sha256(shader.read_bytes()).hexdigest()}
(A/'shaders.json').write_text(json.dumps(hashes,indent=2)+'\n')
