from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-clearance-observer-20261109';B=R/'build/crow-clearance-observer-20261109';B.mkdir(parents=True,exist_ok=True)
base=(R/'build/crow-hierarchical-goal-memory-20261109/c_api.cpp').read_text()
s=base.replace('#include "metalrobo/RetinalObserver.hpp"','#include "box_distance.hpp"\n#include "metalrobo/RetinalObserver.hpp"',1)
s=s.replace('struct MRTaskRolloutHandle {','''struct CrowClearanceRecord {
    double distance = std::numeric_limits<double>::infinity();
    unsigned long long step = 0;
    std::size_t birdShape = 0, obstacleShape = 0;
    bool closed = false;
};
struct MRTaskRolloutHandle {
    std::vector<CrowClearanceRecord> crowClearance;''',1)
needle='''        if (handle->visualRuntime && handle->visualRuntime->retinal) {'''
assert s.count(needle)==1
insert=r'''
        if (std::getenv("CROW_CLEARANCE_TRACE") != nullptr &&
            control_step_count == 1u) {
            if (handle->crowClearance.empty())
                handle->crowClearance.resize(environmentCount);
            if (handle->crowClearance.size()!=environmentCount)
                throw std::runtime_error("clearance observer lane count changed");
            std::vector<MRBodyStateGPU> composed;
            std::string reason;
            if (!metalrobo::composeVisualBodyStates(handle->model,environmentCount,
                    published.finalQ,published.finalV,published.finalSceneBodies,
                    composed,&reason))
                throw std::runtime_error("clearance body composition failed: "+reason);
            const auto& model=handle->model;
            const std::size_t bodyCount=model.bodies.size();
            std::vector<std::size_t> birds,obstacles;
            for (std::size_t i=0;i<model.shapes.size();++i) {
                const auto& shape=model.shapes[i];
                const auto& name=model.bodyNames.at(shape.bodyIndex);
                if (name.starts_with("crow_course_gate_") ||
                    name.starts_with("crow_course_slalom_")) obstacles.push_back(i);
                else if (name.starts_with("crow_") &&
                         !name.starts_with("crow_course_")) birds.push_back(i);
            }
            if (birds.empty() || obstacles.size()!=4u)
                throw std::runtime_error("clearance observer collider contract mismatch");
            const auto makeBox=[&](std::size_t lane,std::size_t index) {
                const auto& shape=model.shapes[index];
                if (shape.shapeType!=MR_SHAPE_BOX)
                    throw std::runtime_error("clearance observer requires box colliders");
                const auto& body=composed[lane*bodyCount+shape.bodyIndex];
                const auto &p=body.position,&q=body.orientation,
                           &lp=shape.localPosition,&lq=shape.localRotation,&d=shape.dimensions;
                return crowclear::box({p.x,p.y,p.z},{q.x,q.y,q.z,q.w},
                    {lp.x,lp.y,lp.z},{lq.x,lq.y,lq.z,lq.w},{d.x,d.y,d.z});
            };
            const auto step=static_cast<unsigned long long>(handle->submittedControlSteps+1u);
            for (std::size_t lane=0;lane<environmentCount;++lane) {
                auto& record=handle->crowClearance[lane];
                if(record.closed)continue;
                for (const auto j:obstacles) {
                    const auto obstacle=makeBox(lane,j);
                    for (const auto i:birds) {
                        const auto bird=makeBox(lane,i);
                        const double bound=std::sqrt(crowclear::norm2(bird.c-obstacle.c))-
                            std::sqrt(bird.h[0]*bird.h[0]+bird.h[1]*bird.h[1]+bird.h[2]*bird.h[2])-
                            std::sqrt(obstacle.h[0]*obstacle.h[0]+obstacle.h[1]*obstacle.h[1]+obstacle.h[2]*obstacle.h[2]);
                        if (bound>=record.distance)continue;
                        const auto distance=crowclear::distance(bird,obstacle);
                        if (distance<record.distance) {
                            record.distance=distance;record.step=step;
                            record.birdShape=i;record.obstacleShape=j;
                        }
                    }
                }
                const bool terminated=lane<published.transitions.size() &&
                   published.transitions[lane].termination.w!=MR_TASK_TERMINATION_CONTINUING;
                if(terminated || step%500u==0u) {
                    record.closed=terminated;
                    const auto& bs=model.shapes[record.birdShape];
                    const auto& os=model.shapes[record.obstacleShape];
                    std::fprintf(stderr,
                        "CROW_CLEARANCE_SAMPLE {\"lane\":%zu,\"end_step\":%llu,\"minimum_step\":%llu,\"minimum_m\":%.12g,\"bird_shape\":\"%s\",\"obstacle_shape\":\"%s\",\"bird_body\":\"%s\",\"obstacle_body\":\"%s\",\"sampling\":\"accepted_control_steps\"}\n",
                        lane,step,record.step,record.distance,
                        model.shapeNames[record.birdShape].c_str(),model.shapeNames[record.obstacleShape].c_str(),
                        model.bodyNames[bs.bodyIndex].c_str(),model.bodyNames[os.bodyIndex].c_str());
                }
            }
        }
'''
s=s.replace(needle,insert+needle,1)
src=B/'c_api.cpp';src.write_text(s)
(A/'api.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/goal-memory-c_api.cpp',tofile='b/clearance-c_api.cpp')))
lines=subprocess.check_output(['ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines()
line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x)
args=shlex.split(line);obj=B/'c_api.cpp.o'
for key,val in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(key)+1]=val
args[-1]=str(src);args.insert(1,'-I'+str(A));subprocess.run(args,cwd=R/'build',check=True)
link=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x)
args=shlex.split(link[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib)
args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj)
args[args.index('CMakeFiles/metalrobo.dir/src/core/TaskProgram.cpp.o')]=str(R/'build/crow-hierarchical-goal-memory-20261109/TaskProgram.cpp.o')
subprocess.run(args,cwd=R/'build',check=True)
(A/'build.json').write_text(json.dumps({'api_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'box_distance_sha256':hashlib.sha256((A/'box_distance.hpp').read_bytes()).hexdigest(),'sampling':'accepted control-step endpoints only; not a claim of continuous substep minimum','scope':'four gate/slalom obstacle colliders; landing perch is the support target, not an obstacle'},indent=2)+'\n')
